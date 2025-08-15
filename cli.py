import argparse
import os
from app.core.indexer import create_index
from app.core.querier import query_index

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

    if args.command == "index":
        if not os.path.isabs(args.path):
            print("Error: Please provide an absolute path for indexing.")
            return
        create_index(args.path)
    elif args.command == "query":
        answer = query_index(args.question)
        print("\nAnswer:")
        print(answer)

if __name__ == "__main__":
    main()
