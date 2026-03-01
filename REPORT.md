# Report - NASA RAG Chat Project

## Overview
There is a `Makefile` in the root of the repo that can be used to run the most used commands.

All the "TODO" in the code were addressed and replaced with a "Note".

- `python3 -m nasa_rag.llm_client` : tests the functionality of the LLM client
- `python3 -m nasa_rag.embedding_pipeline --data-path ./data_text/ --openai-key $OPENAI_API_KEY --chroma-dir ./chroma_db_openai --chunk-size 1024 --chunk-overlap 64 --batch-size 100 --update-mode skip` : was used to index the dataset
- `python3 -m nasa_rag.rag_client` : tests the functionality of the RAG client
- `python3 -m nasa_rag.ragas_evaluator` : runs `ragas` evaluation on the evaluation dataset created in `evaluation_dataset.json`
- `python3 -m streamlit run app/chat.py` : runs the `streamlit` app on a local IP address (edit `./.streamlit/config.toml` to change this address)

## Python virtual environment
To run this project solution we need to install the depenencies listed in the `requirements_*.txt` file obtained with `make lock` in a Python 3.13 virtual environemnt.

We also need to install the `nasa_rag` Python package provided by this repo (for example with `pip install --no-deps .`)

## Comments about OpenAI models used
Giving the user the option to choose between `gpt-4.1-mini`, `gpt-5-nano` and `git-5-mini`. This had to be changes as the default options in `chat.py`.

## Comments about embedding Pipeline
- `get_embedding()` changed to `get_embeddings()` (proceessing chunks in batches) to speed up the process of indexing a new dataset.

## Comments about RAG client
- The `rag_client.py` need the `OpenAIEmbeddingFunction()` which was not imported. Instead it was placed in `embedding_pipeline.py`

## Comments about RAGAS evaluation
- Using `gpt-5` here.
- Need to install extra depedencies for for `ragas` evaluation, e.g. "sacrebleu".
- Needed to pass inputs to the different metrics in a different way for each metric.