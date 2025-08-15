import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb

from app.core.config import load_config

# Load configuration
config = load_config()

def create_index(project_path: str):
    """
    Creates a vector index for the documents in the given project path.

    Args:
        project_path (str): The absolute path to the project directory.
    """
    print("Starting the indexing process...")

    # 1. Configure embedding model
    print(f"Initializing embedding model: {config['embedding']['model']}")
    embed_model = HuggingFaceEmbedding(model_name=config["embedding"]["model"])
    Settings.embed_model = embed_model

    # 2. Load documents
    print(f"Loading documents from: {project_path}")
    # LlamaIndex's SimpleDirectoryReader requires the input directory to exist.
    if not os.path.isdir(project_path):
        print(f"Error: Directory not found at {project_path}")
        return
        
    reader = SimpleDirectoryReader(project_path, recursive=True)
    documents = reader.load_data()
    print(f"Loaded {len(documents)} documents.")

    if not documents:
        print("No documents found to index. Exiting.")
        return

    # 3. Setup ChromaDB vector store
    print("Setting up ChromaDB vector store...")
    storage_path = config["indexing"]["storage_path"]
    
    # Ensure the storage directory exists for ChromaDB
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)

    db = chromadb.PersistentClient(path=storage_path)
    chroma_collection = db.get_or_create_collection("default_collection")
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # 4. Configure storage and build index
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    print("Creating the vector store index...")
    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
    )

    # 5. Persist index to disk (ChromaDB handles its own persistence)
    print(f"Persisting index to: {storage_path}")
    # No explicit persist call needed for ChromaDB here, as it's persistent by nature.
    # However, we still need to persist the LlamaIndex metadata.
    index.storage_context.persist(persist_dir=storage_path)

    print(f"Number of documents in ChromaDB collection: {chroma_collection.count()}")
    print("Indexing complete.")
