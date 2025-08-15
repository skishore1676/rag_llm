import os
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.postprocessor import SentenceTransformerRerank
import chromadb
from app.core.config import load_config

# Load configuration
config = load_config()

def query_index(query_text: str, llm_type: str = None, rerank_enable: bool = None, chat_history: list = None) -> str:
    """
    Queries the existing index with the given text.

    Args:
        query_text (str): The question to ask the index.

    Returns:
        str: The response from the language model.
    """
    storage_path = config["indexing"]["storage_path"]
    
    if not os.path.exists(storage_path):
        return "Index not found. Please run the indexing process first."

    print("Loading the index...")
    
    # 1. Configure LLM and embedding model
    current_llm_type = llm_type if llm_type else config["llm"]["type"]

    if current_llm_type == "ollama":
        Settings.llm = Ollama(
            model=config["llm"]["ollama_model"],
            base_url=config["llm"]["ollama_base_url"],
            temperature=config["llm"]["temperature"],
        )
    else: # Default to openai
        Settings.llm = OpenAI(
            model=config["llm"]["model"],
            temperature=config["llm"]["temperature"],
            api_key=config["llm"]["api_key"],
        )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config["embedding"]["model"]
    )

    # 2. Load the index from storage
    db = chromadb.PersistentClient(path=storage_path)
    chroma_collection = db.get_or_create_collection("default_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    doc_store = SimpleDocumentStore.from_persist_dir(storage_path)
    index_store = SimpleIndexStore.from_persist_dir(storage_path)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        docstore=doc_store,
        index_store=index_store
    )
    index = load_index_from_storage(storage_context)

    # 3. Create a query engine
    print("Creating query engine...")
    
    query_engine_kwargs = {
        "similarity_top_k": config["retrieval"]["similarity_top_k"]
    }

    if rerank_enable is not None:
        current_rerank_enable = rerank_enable
    else:
        current_rerank_enable = config["rerank"]["enable"]

    if current_rerank_enable:
        print(f"Enabling re-ranking with model: {config["rerank"]["model"]}")
        rerank_processor = SentenceTransformerRerank(
            model=config["rerank"]["model"],
            top_n=config["rerank"]["top_n"],
        )
        query_engine_kwargs["node_postprocessors"] = [rerank_processor]

    query_engine = index.as_query_engine(**query_engine_kwargs)

    # 4. Execute the query
    print(f"Executing query: {query_text}")
    response = chat_engine.chat(query_text)

    source_nodes = []
    for node in response.source_nodes:
        source_nodes.append({
            "text": node.text,
            "score": node.score,
            "metadata": node.metadata
        })

    return {"answer": str(response), "sources": source_nodes}
