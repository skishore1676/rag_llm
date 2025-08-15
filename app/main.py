from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from llama_index.core import (
    StorageContext,
    load_index_from_storage,
    VectorStoreIndex,
    Settings,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from app.core.config import load_config, save_config
from app.core.indexer import create_index
from app.core.querier import query_index

config = load_config()
index: VectorStoreIndex | None = None # Global variable to hold the loaded index

app = FastAPI()

# Mount static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.on_event("startup")
def startup_event():
    """Load the index into memory when the application starts."""
    global index
    storage_path = config["indexing"]["storage_path"]
    storage_path = config["indexing"]["storage_path"]
    if os.path.exists(storage_path):
        print("Loading index from disk at startup...")
        try:
            # 1. Configure embedding model
            print(f"Initializing embedding model: {config['embedding']['model']}")
            embed_model = HuggingFaceEmbedding(model_name=config["embedding"]["model"])
            Settings.embed_model = embed_model

            # 2. Load the index from storage
            db = chromadb.PersistentClient(path=storage_path)
            chroma_collection = db.get_or_create_collection("default_collection")
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store, persist_dir=storage_path
            )
            index = load_index_from_storage(storage_context)
            print("Index loaded successfully.")
        except Exception as e:
            print(f"Error loading index at startup: {e}")
            index = None

templates = Jinja2Templates(directory="app/templates")

# CORS middleware to allow requests from the frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def deep_update(source: dict, overrides: dict) -> dict:
    """
    Recursively update a dictionary.
    Modifies `source` in place.
    """
    for key, value in overrides.items():
        if isinstance(value, dict) and key in source and isinstance(source.get(key), dict):
            deep_update(source[key], value)
        else:
            source[key] = value
    return source

@app.get("/config")
async def get_config():
    return config

@app.post("/config")
async def update_config(new_config: dict):
    global config
    deep_update(config, new_config)
    save_config(config)
    return {"status": "success", "message": "Configuration updated successfully."}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    # Reload config to ensure UI has the latest project paths
    global config
    config = load_config()
    template_vars = {
        "request": request,
        "project_paths": config["indexing"]["project_paths"],
        "default_llm_type": config["llm"]["type"],
        "openai_models": config["llm"].get("openai_models", []),
        "ollama_models": config["llm"].get("ollama_models", [])
    }
    return templates.TemplateResponse("index.html", template_vars)

@app.post("/index")
def index_project(project_path: str = Form(...)):
    if not os.path.isabs(project_path):
        return {"status": "error", "message": "Please provide an absolute path for indexing."}
    try:
        create_index(project_path, config)
        # Reload the index after creating it
        startup_event()
        return {"status": "success", "message": f"Indexing complete for {project_path}."}
    except Exception as e:
        return {"status": "error", "message": f"An error occurred during indexing: {str(e)}"}

@app.post("/query")
async def query_project(question: str = Form(...), llm_type: str = Form(None), rerank_enable: bool = Form(False), chat_history: str = Form("[]")):
    if index is None:
        return {"answer": "Index is not loaded. Please index a project and ensure the server has started correctly.", "sources": []}

    import json
    chat_history_list = json.loads(chat_history)
    # Pass the pre-loaded index object for high performance
    answer_data = query_index(question, index, config, llm_type, rerank_enable, chat_history_list)
    return answer_data

if __name__ == "__main__":
    # Ensure the 'static' and 'templates' directories exist
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)