from fastapi import FastAPI, Request, Form, BackgroundTasks, HTTPException
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
import logging

from app.core.config import load_config, save_config
from app.core.indexer import create_index
from app.core.querier import query_index
from app.core.agent.master_agent import MasterAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = load_config()
indexing_status: str = "ready" # Status for UI feedback

app = FastAPI()

# Mount static files (CSS, JS, etc.)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

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

async def index_in_background_and_update_status(project_path: str, index_name: str):
    global indexing_status
    try:
        indexing_status = "indexing"
        logger.info(f"Indexing in background for {project_path} as index {index_name}")
        create_index(project_path, config, index_name)
        logger.info(f"Indexing in background complete for {project_path}")
    except Exception as e:
        logger.error(f"An error occurred during background indexing for {project_path}: {str(e)}")
    finally:
        indexing_status = "ready"

@app.get("/index/status")
async def get_index_status():
    global indexing_status
    return {"status": indexing_status}

@app.get("/indexes")
async def get_indexes():
    storage_path = config["indexing"]["storage_path"]
    if not os.path.exists(storage_path):
        return {"indexes": []}
    # List subdirectories, which are the named indexes
    subdirs = [d for d in os.listdir(storage_path) if os.path.isdir(os.path.join(storage_path, d))]
    return {"indexes": subdirs}

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
async def index_project(background: BackgroundTasks, project_path: str = Form(...), index_name: str = Form(...)):
    if not os.path.isabs(project_path):
        logger.error("Please provide an absolute path for indexing.")
        return {"status": "error", "message": "Please provide an absolute path for indexing."}
    background.add_task(index_in_background_and_update_status, project_path, index_name)
    return {"status": "info", "message": "Indexing started in the background. Please wait a moment and then try querying."}

@app.post("/query")
async def query_project(question: str = Form(...), index_name: str = Form(...), llm_type: str = Form(None), rerank_enable: bool = Form(False), chat_history: str = Form("[]")):
    try:
        import json
        chat_history_list = json.loads(chat_history)
        # Load the correct index based on index_name
        storage_path = os.path.join(config["indexing"]["storage_path"], index_name)
        if not os.path.exists(storage_path):
            return {"answer": f"Index '{index_name}' does not exist. Please select a valid index.", "sources": []}

        logger.info(f"Query received: {question} for index {index_name}")
        answer_data = query_index(question, index_name, config, llm_type, rerank_enable, chat_history_list)
        logger.info("Query completed successfully")
        return answer_data
    except Exception as e:
        logger.error(f"An error occurred during querying: {str(e)}")
        return {"answer": "An error occurred while processing your query. Please try again.", "sources": []}

@app.post("/agent-query")
async def agent_query(question: str = Form(...), index_name: str = Form(...), excel_path: str = Form(...)):
    if not os.path.isfile(excel_path):
        raise HTTPException(status_code=400, detail="Excel file not found.")

    try:
        master_agent = MasterAgent(config, index_name, excel_path)
        response = master_agent.query(question)
        logger.info("Agent query completed successfully")
        return {"answer": response}
    except Exception as e:
        logger.error(f"An error occurred during agent querying: {str(e)}")
        return {"answer": "An error occurred while processing your agent query. Please try again."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
