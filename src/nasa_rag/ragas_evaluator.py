import json
import os
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_openai import OpenAIEmbeddings
from openai import AsyncOpenAI

from nasa_rag.rag_client import discover_chroma_backends, retrieve_documents

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms.base import llm_factory
    from ragas.metrics import NonLLMContextPrecisionWithReference, ResponseRelevancy
    from ragas.metrics.collections import (
        BleuScore,
        ContextPrecision,
        ContextRecall,
        ContextRelevance,
        Faithfulness,
        RougeScore,
    )

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False


def evaluate_response_quality(
    question: str,
    answer: str,
    contexts: List[str],
    reference: Optional[str],
) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}

    # Note: guard logic to avoid crashes
    if not question or not answer or not contexts:
        return {"error": "Incomplete triple"}

    if not isinstance(contexts, list):
        return {"error": "contexts needs to be a list"}

    if len(contexts) == 0:
        return {"error": "contexts needs to contain at least one element"}

    if any([not isinstance(con, str) for con in contexts]):
        return {"error": "All elements in contexts needs to be of type string"}

    openai_api_key = os.getenv("OPENAI_API_KEY")
    eval_model_name = "gpt-3.5-turbo"
    # Note:  Create evaluator LLM with model gpt-3.5-turbo
    llm = llm_factory(eval_model_name, client=AsyncOpenAI())

    # Note:  Create evaluator_embeddings with model test-embedding-3-small
    langchain_openai_embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    evaluator_embeddings = LangchainEmbeddingsWrapper(langchain_openai_embeddings)

    # Note:  Define an instance for each metric to evaluate
    faithfulness = Faithfulness(llm=llm)
    context_recall = ContextRecall(llm=llm)
    context_precision = ContextPrecision(llm=llm)
    context_relevance = ContextRelevance(llm=llm)

    # Note: metrics requiring reference
    response_relevacy = ResponseRelevancy(llm=llm, embeddings=evaluator_embeddings)
    bleu = BleuScore()
    non_llm_context = NonLLMContextPrecisionWithReference()
    rouge = RougeScore()

    # Note:  Evaluate the response using the metrics
    results = {
        "faithfulness": faithfulness.score(
            **{
                "user_input": question,
                "retrieved_contexts": contexts,
                "response": answer,
            }
        ).value,
        "context_relevance": context_relevance.score(
            **{
                "user_input": question,
                "retrieved_contexts": contexts,
            }
        ).value,
    }

    if reference is not None:
        eval_inputs = {
            "user_input": question,
            "retrieved_contexts": contexts,
            "reference_contexts": contexts,
            "response": answer,
            "reference": reference,
        }
        sample = SingleTurnSample(**eval_inputs)

        # Note: computing context recall with error handling
        try:
            results["context_recall"] = context_recall.score(
                **{
                    "user_input": question,
                    "retrieved_contexts": contexts,
                    "reference": reference,
                }
            ).value
        except Exception:
            results["context_recall"] = 0.0

        # Note: computing context precision with error handling
        try:
            results["context_precision"] = context_precision.score(
                **{
                    "user_input": question,
                    "retrieved_contexts": contexts,
                    "reference": reference,
                }
            ).value
        except Exception:
            results["context_precision"] = 0.0

        # Note: computing response relevancy with error handling
        try:
            results["response_relevancy"] = float(response_relevacy.single_turn_score(sample))
        except Exception:
            results["response_relevancy"] = 0.0

        # Note: computing non llm context score with error handling
        try:
            results["non_llm_context_score"] = float(non_llm_context.single_turn_score(sample))
        except Exception:
            results["non_llm_context_score"] = 0.0

        # Note: computing Bleu score with error handling
        try:
            results["bleu_score"] = bleu.score(
                **{
                    "response": answer,
                    "reference": reference,
                }
            ).value
        except Exception:
            results["bleu_score"] = 0.0

        # Note: computing Rouge score with error handling
        try:
            results["rouge_score"] = rouge.score(
                **{
                    "response": answer,
                    "reference": reference,
                }
            ).value
        except Exception:
            results["rouge_score"] = 0.0

    # Note:  Return the evaluation results
    return results


if __name__ == "__main__":
    print("INFO: --- RAGAS evaluation ---")

    evaluation_dataset = None
    try:
        with open("evaluation_dataset.json", "r") as eval_dataset_file:
            evaluation_dataset = json.load(eval_dataset_file)
    except Exception as e:
        print(f"Error while loading evaluation data for RAGAS: {e}")
        exit(1)

    evaluation_dataset = [(item["q"], item["a"], item["r"]) for item in evaluation_dataset]

    backends = discover_chroma_backends()
    collection_name = backends["chroma_db_openai_nasa_space_missions_text"]["collection_name"]

    chroma_client = chromadb.PersistentClient(
        path="chroma_db_openai",
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )
    collection = chroma_client.get_collection(
        name=collection_name,
        embedding_function=OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small",
        ),
    )

    ragas_scores = []
    metric_names = set()
    for question, answer, reference in evaluation_dataset:
        query_result = retrieve_documents(
            query=question,
            collection=collection,
            n_results=3,
        )

        eval_scores = evaluate_response_quality(
            question,
            answer,
            contexts=query_result["documents"][0],
            reference=reference,
        )

        ragas_scores.append(eval_scores)
        metric_names = metric_names.union(set(eval_scores.keys()))

    print("-" * 30)
    for metric_name in metric_names:
        values = [item[metric_name] for item in ragas_scores]
        avg = sum(values) / max(1, len(values))
        print(f"RAGAS evaluation: average {metric_name} is {avg:.3}")
