import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("chroma_rag_client.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


def discover_chroma_backends() -> Dict[str, Dict[str, str]]:
    """Discover available ChromaDB backends in the project directory"""
    backends = {}
    current_dir = Path(".")

    # Look for ChromaDB directories
    # Note:  Create list of directories that match specific criteria (directory type and name pattern)
    chroma_db_dirs = [
        dir for dir in current_dir.iterdir() if dir.is_dir() if str(dir).startswith("chroma_db")
    ]

    # Note:  Loop through each discovered directory
    # Note:  Wrap connection attempt in try-except block for error handling
    for db_dir in chroma_db_dirs:
        try:
            # Note:  Initialize database client with directory path and configuration settings
            chroma_client = chromadb.PersistentClient(
                path=db_dir,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            # Note:  Retrieve list of available collections from the database
            all_collections = chroma_client.list_collections()
            # Note:  Loop through each collection found
            for collection in all_collections:
                # Note:  Create unique identifier key combining directory and collection names

                # Note:  Build information dictionary containing:
                # Note:  Store directory path as string
                # Note:  Store collection name
                # Note:  Create user-friendly display name
                # Note:  Get document count with fallback for unsupported operations
                # Note:  Add collection information to backends dictionary
                collection_unique_id = f"{db_dir}_{collection.name}"
                collection_information = {
                    "directory": str(db_dir),
                    "collection_name": collection.name,
                    "display_name": collection.name.replace("_", " "),
                    "document_count": collection.count(),
                }

                backends[collection_unique_id] = collection_information
        except Exception as e:
            # Note:  Handle connection or access errors gracefully

            # Note:  Set appropriate fallback values for missing information
            logger.exception(f"Error in discover_chroma_backends(): {e}")

            # Note:  Create fallback entry for inaccessible directories
            collection_unique_id = f"{db_dir}_unknown"
            collection_information = {
                "directory": str(db_dir),
                "collection_name": "unknown",
                # Note:  Include error information in display name with truncation
                "display_name": f"Error {db_dir}: {str(e)[:10]}",
                "document_count": 0,
            }

            backends[collection_unique_id] = collection_information

    # Note:  Return complete backends dictionary with all discovered collections
    return backends


def initialize_rag_system(chroma_dir: str, collection_name: str):
    """Initialize the RAG system with specified backend (cached for performance)"""

    try:
        # Note:  Create a chomadb persistentclient
        chroma_client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
        # Note:  Return the collection with the collection_name
        collection = chroma_client.get_collection(
            name=collection_name,
            embedding_function=OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small",
            ),
        )
        return collection, True, None
    except Exception as e:
        return None, False, e


def retrieve_documents(
    collection: chromadb.Collection,
    query: str,
    n_results: int = 3,
    mission_filter: Optional[str] = None,
) -> Optional[Dict]:
    """Retrieve relevant documents from ChromaDB with optional filtering"""

    # Note:  Initialize filter variable to None (represents no filtering)
    filter = None
    # Note:  Check if filter parameter exists and is not set to "all" or equivalent
    if (mission_filter is not None) and (mission_filter != "all"):
        # Note:  If filter conditions are met, create filter dictionary with appropriate field-value pairs
        filter = {"mission": mission_filter}

    # Note:  Execute database query with the following parameters:
    # Note:  Pass search query in the required format
    # Note:  Set maximum number of results to return
    # Note:  Apply conditional filter (None for no filtering, dictionary for specific filtering)
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=filter,
    )
    # Note:  Return query results to caller
    return results


def format_context(documents: List[str], metadatas: List[Dict], ids: List[str]) -> str:
    """Format retrieved documents into context"""
    if not documents:
        return ""

    # Note:  Initialize list with header text for context section
    context_parts_list = ["DOCUMENTS"]
    seen_ids = set()

    for i in range(len(documents)):
        doc_id = ids[i]

        # Note: avoid adding duplicate documents
        if doc_id in seen_ids:
            continue

        seen_ids.add(doc_id)
        # Note:  Loop through paired documents and their metadata using enumeration
        for idx, (text, metadata) in enumerate(zip(documents, metadatas)):
            # Note:  Extract mission information from metadata with fallback value
            # Note:  Clean up mission name formatting (replace underscores, capitalize)
            # Note:  Extract source information from metadata with fallback value
            # Note:  Extract category information from metadata with fallback value
            # Note:  Clean up category name formatting (replace underscores, capitalize)
            mission = metadata.get("mission", "unknown").replace("_", " ").lower()
            source = metadata.get("source", "unknown")
            document_category = (
                metadata.get("document_category", "unknown").replace("_", " ").lower()
            )

            # Note: deduplicating added texts by checking if they are already present
            # We do this by checking if the text_id was already added to the context parts
            text_id = f"[DOC {mission} {source} {document_category} {idx}] "
            # Note:  Create formatted source header with index number and extracted information
            # Note:  Add source header to context parts list
            context_parts_list.append(text_id)
            # Note:  Check document length and truncate if necessary
            # Note:  Add truncated or full document content to context parts list
            context_parts_list.append(text)

        # Note:  Join all context parts with newlines and return formatted string
        return "\n".join(context_parts_list)


if __name__ == "__main__":
    print("INFO: Test discover backends")
    backends = discover_chroma_backends()
    print(f"INFO: {backends=}")
    print()

    print("INFO: Test retrieve documents")
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
    s = time.time()
    query_result = retrieve_documents(
        query="Tell me what you know about the Apollo NASA missions? How many there were?",
        collection=collection,
        n_results=3,
    )
    e = time.time()
    print(f"INFO: ChromaDB query time: {e - s:.2f} seconds")

    print(f"INFO: {type(query_result)=}")
    for idx, doc in enumerate(query_result["documents"][0]):
        print(f"INFO: {idx}\n{doc}")
        print("=" * 60)
