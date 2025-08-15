import argparse
import os
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
)
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from app.core.indexer import create_index
from app.core.querier import query_index
from app.core.config import load_config

def main():
    parser = argparse.ArgumentParser(description="RAG LLM CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Indexing command
    index_parser = subparsers.add_parser("index", help="Create or update the index")
    index_parser.add_argument("--path", type=str, required=True, help="Absolute path to the project directory to index")

    # Querying command
    query_parser = subparsers.add_parser("query", help="Ask a question to the indexed documents")
    query_parser.add_argument("question", type=str, help="The question to ask")

    args = parser.parse_args()

    config = load_config()

    if args.command == "index":
        if not os.path.isabs(args.path):
            print("Error: Please provide an absolute path for indexing.")
            return
        create_index(args.path, config)
    elif args.command == "query":
        storage_path = config["indexing"]["storage_path"]
        if not os.path.exists(storage_path):
            print("Error: Index not found. Please run the 'index' command first.")
            return

        print("Loading index for CLI query...")
        db = chromadb.PersistentClient(path=storage_path)
        chroma_collection = db.get_or_create_collection("default_collection")
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store, persist_dir=storage_path)
        index = load_index_from_storage(storage_context)
        print("Index loaded.")

        response = query_index(args.question, index, config)
        print("\nAnswer:")
        print(response["answer"])

if __name__ == "__main__":
    main()
