import os
import re
from dotenv import load_dotenv

load_dotenv()


def extract_video_id(url: str) -> str:
    pattern = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract video ID from URL: {url}")


def fetch_comments_paginated(video_id: str, max_comments: int = 1000):

    if os.getenv("CI"):
        yield [
            "Great video!",
            "Can you explain more?",
            "This is wrong",
            "Loved it!",
            "What tool do you use?"
        ]
        return

    from googleapiclient.discovery import build

    youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))

    fetched = 0

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText",
    )

    while request and fetched < max_comments:
        response = request.execute()

        page_comments = []

        for item in response.get("items", []):
            text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            page_comments.append(text)
            fetched += 1

            if fetched >= max_comments:
                break

        if page_comments:
            yield page_comments

        request = youtube.commentThreads().list_next(request, response)


def fetch_transcript(video_id: str) -> str:
    if os.getenv("CI"):
        return "Dummy transcript"

    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        return " ".join([item["text"] for item in transcript])
    except Exception:
        return ""


def fetch_video_metadata(video_id: str) -> dict:
    return {"title": "", "description": ""}