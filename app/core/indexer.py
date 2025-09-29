import os
import logging
from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from .parser import ParserFactory

logger = logging.getLogger(__name__)
 
def create_index(project_path: str, config: dict, index_name: str):
    """
    Creates a vector index for the documents in the given project path.

    Args:
        project_path (str): The absolute path to the project directory.
        config (dict): The application configuration dictionary.
    """
    logger.info("Starting the indexing process...")

    # 1. Configure embedding model
    logger.info(f"Initializing embedding model: {config['embedding']['model']}")
    embed_model = HuggingFaceEmbedding(model_name=config["embedding"]["model"])
    Settings.embed_model = embed_model
    Settings.chunk_size = config["indexing"]["chunk_size"]
    logger.debug(f"Setting chunk size to: {Settings.chunk_size}")

    # 2. Load documents
    logger.info(f"Loading documents from: {project_path}")
    if not os.path.isdir(project_path):
        raise ValueError(f"Directory not found at {project_path}")

    try:
        # Get parser configuration and instantiate parser
        parser_config = config.get("parser", {"type": "unstructured"})
        parser = ParserFactory.get_parser(parser_config)
        logger.info(f"Using parser: {type(parser).__name__}")

        # Define supported file extensions for document processing
        supported_extensions = {
            '.txt', '.md', '.pdf', '.docx', '.doc', '.xlsx', '.xls',
            '.pptx', '.ppt', '.csv', '.html', '.htm'
        }

        documents = []
        processed_files_count = 0
        # Walk through directory recursively to find files
        for root, dirs, files in os.walk(project_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = os.path.splitext(file)[1].lower()

                # Skip temporary Office files (start with ~$)
                if file.startswith('~$'):
                    logger.debug(f"Skipping temporary Office file: {file_path}")
                    continue

                # Process only supported file types
                if file_ext in supported_extensions:
                    try:
                        logger.debug(f"Parsing file: {file_path}")
                        file_documents = parser.parse(file_path)
                        documents.extend(file_documents)
                        processed_files_count += 1
                    except Exception as e:
                        logger.warning(f"Failed to parse {file_path}: {str(e)}")
                        continue

        logger.info(f"Loaded {len(documents)} documents from {processed_files_count} files.")

        # Filter out documents with missing content and clean invalid metadata
        valid_documents = []
        for doc in documents:
            if doc.text and doc.text.strip():  # Skip empty documents
                # Clean metadata to only allow str, int, float, None
                if hasattr(doc, 'metadata') and doc.metadata:
                    cleaned_metadata = {}
                    for key, value in doc.metadata.items():
                        if isinstance(value, (str, int, float, type(None))):
                            cleaned_metadata[key] = value
                        else:
                            cleaned_metadata[key] = str(value)  # Convert to string if invalid type
                    doc.metadata = cleaned_metadata
                valid_documents.append(doc)
        documents = valid_documents
        logger.info(f"After filtering, {len(documents)} valid documents remain.")
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise


    # 3. Setup ChromaDB vector store
    logger.info("Setting up ChromaDB vector store...")
    base_storage_path = config["indexing"]["storage_path"]
    storage_path = os.path.join(base_storage_path, index_name)

    # Ensure the storage directory exists for ChromaDB
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
        logger.debug(f"Created storage directory: {storage_path}")

    try:
        db = chromadb.PersistentClient(path=storage_path)
        chroma_collection = db.get_or_create_collection(index_name)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

        # 4. Configure storage and build index
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        logger.info("Creating the vector store index...")
        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
        )

        # 5. Persist index to disk (ChromaDB handles its own persistence)
        logger.info(f"Persisting index to: {storage_path}")
        # No explicit persist call needed for ChromaDB here, as it's persistent by nature.
        # However, we still need to persist the LlamaIndex metadata.
        index.storage_context.persist(persist_dir=storage_path)

        logger.info(f"Number of documents in ChromaDB collection: {chroma_collection.count()}")
        logger.info("Indexing complete.")
    except Exception as e:
        logger.error(f"Error during indexing: {str(e)}")
        raise
