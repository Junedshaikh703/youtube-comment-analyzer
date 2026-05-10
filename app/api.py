from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import uuid

from src.inference.inference_service import analyze_comments
from src.services.yt_fetcher import extract_video_id

app = FastAPI()
templates = Jinja2Templates(directory="app/templates")

# 🔥 GLOBAL TASK STORE
tasks = {}


# 🔹 Home Page
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


# 🔹 START ANALYSIS
@app.post("/analyze")
async def start_analysis(request: Request, background_tasks: BackgroundTasks):
    data = await request.json()
    video_url = data["video_url"]

    video_id = extract_video_id(video_url)
    task_id = str(uuid.uuid4())

    tasks[task_id] = {
        "status": "processing",
        "progress": "Starting...",
        "processed": 0,
        "total": 0,
        "counts": {"positive": 0, "negative": 0, "question": 0, "neutral": 0},
        "partial_summary": None,
        "result": {
            "summary": "",
            "replies": []
        }
    }

    background_tasks.add_task(analyze_comments, video_id, task_id, tasks)

    return {"task_id": task_id}


# 🔹 STATUS API
@app.get("/status/{task_id}")
async def get_status(task_id: str):
    return tasks.get(task_id, {"error": "Invalid task"})