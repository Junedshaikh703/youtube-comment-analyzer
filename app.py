from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request

from src.inference.inference_service import analyze_comments
from src.utils.youtube_fetcher import extract_video_id, fetch_comments

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "summary": None, "pairs": None}
    )


@app.post("/", response_class=HTMLResponse)
def analyze(request: Request, video_url: str = Form(...)):

    video_id = extract_video_id(video_url)

    comments = fetch_comments(video_id)

    summary, pairs = analyze_comments(comments)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "summary": summary,
            "pairs": pairs
        }
    )