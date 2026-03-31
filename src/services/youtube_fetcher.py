import os
from dotenv import load_dotenv

load_dotenv()


def extract_video_id(url):

    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]

    if "youtu.be/" in url:
        return url.split("youtu.be/")[1]

    return None


def fetch_comments(video_id, max_comments=60):

    # ✅ CI mode → return dummy data
    if os.getenv("CI"):
        return [
            "Great video!",
            "Nice explanation",
            "Can you explain more?",
            "Loved the content"
        ]

    from googleapiclient.discovery import build

    API_KEY = os.getenv("YOUTUBE_API_KEY")

    youtube = build("youtube", "v3", developerKey=API_KEY)

    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    response = request.execute()

    for item in response["items"]:
        comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
        comments.append(comment)

        if len(comments) >= max_comments:
            break

    return comments