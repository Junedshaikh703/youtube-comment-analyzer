from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import asyncio

from src.inference.inference_service import analyze_comments
from src.services.youtube_fetcher import extract_video_id, fetch_comments

app = FastAPI()

templates = Jinja2Templates(directory="app/templates")


# 🔹 1. Home Page (GET)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request, "summary": None, "pairs": None}
    )


# 🔹 2. Form Submission (POST - HTML flow)
@app.post("/", response_class=HTMLResponse)
async def analyze(request: Request, video_url: str = Form(...)):

    video_id = extract_video_id(video_url)

    # Run blocking work in threads
    comments = await asyncio.to_thread(fetch_comments, video_id)
    summary, pairs = await asyncio.to_thread(analyze_comments, comments)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "summary": summary,
            "pairs": pairs
        }
    )


# 🔹 3. Pure API Endpoint (JSON - for ML / external use)
@app.post("/predict")
async def predict(video_url: str):

    video_id = extract_video_id(video_url)

    comments = await asyncio.to_thread(fetch_comments, video_id)
    summary, pairs = await asyncio.to_thread(analyze_comments, comments)

    return JSONResponse(
        content={
            "summary": summary,
            "pairs": pairs
        }
    )