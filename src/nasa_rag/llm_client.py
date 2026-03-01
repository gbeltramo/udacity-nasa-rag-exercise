import copy
import os
from typing import Dict, List

from openai import OpenAI


def generate_response(
    openai_key: str,
    user_message: str,
    context: str,
    conversation_history: List[Dict],
    model: str = "gpt-5-mini",
) -> str:
    """Generate response using OpenAI with context."""

    # Note:  Define system prompt
    # Note: this is done after the line `if "messages" not in st.session_state:` in the chat.py file

    # Note:  Set context in messages
    # Note:  Add chat history
    # Note:  Create OpenAI Client
    # Note:  Send request to OpenAI
    # Note:  Return response

    client = OpenAI(
        api_key=openai_key,
    )

    # Note: copying conversation history to avoid adding RAG messages in it
    # Moreover the input conversation_history is st.session_state.messages[:-1]
    # which is excluding the last {"role": "user", "content": user_message} from the copy
    rag_messages = copy.deepcopy(conversation_history)

    rag_query = f"""You are answering questions using ONLY information from the provided documents.

CRITICAL RULES:
1. Every fact in your answer MUST come directly from the documents below
2. If the documents don't contain enough information to answer the question, you MUST say so
3. DO NOT use any knowledge from your training data
4. DO NOT make inferences beyond what is explicitly stated
5. DO NOT fill in gaps with plausible-sounding information

If you encounter conflicting information in the documents:

1. DO NOT choose which source is correct
2. DO NOT blend contradictory information
3. INSTEAD, clearly state that sources conflict:
   "The documents contain conflicting information on this point:
   - Source A states: [exact quote]
   - Source B states: [exact quote]
   Both are provided as documented facts; clarification is needed."


When the provided documents don't fully answer the question:

1. Provide whatever information IS available from the documents
2. Explicitly state what information is MISSING
3. DO NOT fill gaps with plausible assumptions

Use this format:

BASED ON PROVIDED DOCUMENTS:
[What you can answer from the documents]

INFORMATION WHICH IS CONTRADICTORY IN DOCUMENTS:
[Details about contradictions]

INFORMATION NOT FOUND IN DOCUMENTS:
[Specific details needed but missing]

-----

{context}

-----

QUESTION:
{user_message}

Provide your answer following the rules above.
"""

    # Note: adding RAG query (containig the user_message) to the
    # temporary variable rag_messages. None of the retrieved documents are saved
    # in the conversation_history variable
    rag_messages.append(
        {
            "role": "system",
            "content": rag_query,
        }
    )

    response = client.chat.completions.create(
        messages=rag_messages,
        model=model,
        temperature=1.0,  # Note: gpt-5-mini is restriced to temperature 1.0
        max_completion_tokens=8192,
    )
    assistant_message = response.choices[0].message.content

    # Note: We only append the assistant message to the conversation history
    # This is done in the file `chat.py` in the line
    # st.session_state.messages.append({"role": "assistant", "content": response})

    return assistant_message


if __name__ == "__main__":
    api_key = os.getenv("OPENAI_API_KEY")
    conversation_history = []

    # Note: this is only for testing. It is not used in the real RAG application
    conversation_history.append(
        {
            "role": "system",
            "content": """You are a NASA Subject Matter Expert assistant.
Use authoritative, evidence-based knowledge about NASA programs, aerospace engineering, space science, mission operations, and related policy.
Prioritize and explicitly cite any retrieved documents before relying on your internal knowledge.

""",
        }
    )
    # Note: this is only for testing. It is not used in the real RAG application
    print(
        generate_response(
            openai_key=api_key,
            user_message="What is the capital of Italy?",
            context="France -> Paris\nGermany -> Berlin\nItaly -> Rome\nSpain -> Madrid",
            conversation_history=conversation_history,
            model="gpt-5-mini",
        )
    )
