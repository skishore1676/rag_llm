from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import os

from app.core.config import load_config, save_config

config = load_config()

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

@app.get("/config")
async def get_config():
    return config

@app.post("/config")
async def update_config(new_config: dict):
    global config # Declare config as global to modify the module-level variable
    config.update(new_config)
    save_config(config)
    return {"status": "success", "message": "Configuration updated successfully."}

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "project_paths": config["indexing"]["project_paths"], "default_llm_type": config["llm"]["type"]})

@app.post("/index")
async def index_project(project_path: str = Form(...)):
    if not os.path.isabs(project_path):
        return {"status": "error", "message": "Please provide an absolute path for indexing."}
    create_index(project_path)
    return {"status": "success", "message": f"Indexing started for {project_path}"}

@app.post("/query")
async def query_project(question: str = Form(...), llm_type: str = Form(None), rerank_enable: bool = Form(False), chat_history: str = Form("[]")):
    import json
    chat_history_list = json.loads(chat_history)
    answer_data = query_index(question, llm_type, rerank_enable, chat_history_list, config)
    return answer_data

if __name__ == "__main__":
    # Ensure the 'static' and 'templates' directories exist
    os.makedirs("static", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)