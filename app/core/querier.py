import os
import logging
from llama_index.core import (
    Settings,
    VectorStoreIndex,
    PromptTemplate,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
import chromadb

logger = logging.getLogger(__name__)

def query_index(query_text: str, index_name: str, config: dict, llm_type: str = None, rerank_enable: bool = None, chat_history: list = None) -> dict:
    """
    Queries the existing index with the given text.

    Args:
        query_text (str): The question to ask the index.
        index (VectorStoreIndex): The pre-loaded LlamaIndex object.
        config (dict): The application configuration dictionary.
        llm_type (str, optional): The LLM to use ('openai' or 'ollama'). Defaults to config.
        rerank_enable (bool, optional): Whether to enable re-ranking. Defaults to config.
        chat_history (list, optional): The chat history. Currently unused.

    Returns:
        dict: A dictionary containing the answer and source documents.
    """
    # 1. Configure embedding model. This must be done BEFORE the LLM is configured
    # to prevent LlamaIndex from defaulting to the LLM's embedding model.
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=config["embedding"]["model"]
    )

    # 2. Configure LLM
    current_llm_type = llm_type if llm_type else config["llm"]["type"]

    if current_llm_type == "ollama":
        Settings.llm = Ollama(
            model=config["llm"]["ollama_model"],
            base_url=config["llm"]["ollama_base_url"],
            temperature=config["llm"]["temperature"],
            request_timeout=120.0,  # Set a longer timeout (in seconds)
        )
    else: # Default to openai
        Settings.llm = OpenAI(
            model=config["llm"]["model"],
            temperature=config["llm"]["temperature"],
            api_key=config["llm"]["api_key"],
        )

    try:
        # Build chat history string
        chat_history_str = ""
        if chat_history:
            chat_history_lines = []
            for item in chat_history:
                chat_history_lines.append(f"{item['role'].capitalize()}: {item['content']}")
            chat_history_str = "Conversation History:\n" + "\n".join(chat_history_lines) + "\n"

        # Load the index
        logger.info(f"Loading index '{index_name}'")
        path = os.path.join(config["indexing"]["storage_path"], index_name)
        db = chromadb.PersistentClient(path=path)
        chroma_collection = db.get_or_create_collection(index_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=path)
        index = load_index_from_storage(storage_context)

        # 3. Create a query engine from the loaded index
        logger.info("Creating query engine...")

        # --- Enhanced Prompt Template ---
        QA_PROMPT_TMPL_STR = (
            "You are a helpful expert assistant. Your job is to answer questions about the provided context.\n"
            "The context contains documents from a user's project.\n"
            "Use only the information from the context below to answer the question.\n"
            "If the context does not contain the answer, say 'I do not have enough information in the context to answer this question.'\n"
            "Do not use any of your prior knowledge.\n\n"
            "Context Information:\n"
            "---------------------\n"
            "{context_str}\n"
            "---------------------\n\n"
            "{chat_history}\n"
            "User's Question: {query_str}\n\n"
            "Answer: "
        )
        qa_template = PromptTemplate(QA_PROMPT_TMPL_STR).partial_format(chat_history=chat_history_str)

        query_engine_kwargs = {
            "similarity_top_k": config["retrieval"]["similarity_top_k"],
            "text_qa_template": qa_template,
        }

        if rerank_enable is not None:
            current_rerank_enable = rerank_enable
        else:
            current_rerank_enable = config["rerank"]["enable"]

        if current_rerank_enable:
            logger.info(f"Enabling re-ranking with model: {config["rerank"]["model"]}")
            rerank_processor = SentenceTransformerRerank(
                model=config["rerank"]["model"],
                top_n=config["rerank"]["top_n"],
            )
            query_engine_kwargs["node_postprocessors"] = [rerank_processor]

        query_engine = index.as_query_engine(**query_engine_kwargs)

        # 4. Execute the query
        logger.info(f"Executing query: {query_text}")
        response = query_engine.query(query_text)

        source_nodes = []
        for node in response.source_nodes:
            source_nodes.append({
                "text": node.text,
                "score": node.score,
                "metadata": node.metadata
            })

        return {"answer": str(response), "sources": source_nodes}
    except Exception as e:
        logger.error(f"An error occurred during querying: {str(e)}")
        raise
