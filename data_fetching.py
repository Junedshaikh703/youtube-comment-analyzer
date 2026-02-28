import os
import re
import pandas as pd
from dotenv import load_dotenv
from googleapiclient.discovery import build

# Load API key
load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")

youtube = build("youtube", "v3", developerKey=API_KEY)


def extract_video_id(url):
    """
    Extract video ID from YouTube URL
    """
    pattern = r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})"
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    else:
        raise ValueError("Invalid YouTube URL")


def fetch_comments(video_id, max_comments=60):
    """
    Fetch top-level comments for a given video ID
    """
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    while request and len(comments) < max_comments:
        response = request.execute()

        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

            if len(comments) >= max_comments:
                break

        request = youtube.commentThreads().list_next(request, response)

    return comments


if __name__ == "__main__":

    video_urls = [
        "https://youtu.be/eCjuoqUy8Is?si=UtGfYZQTIaDuPTZH",
        "https://youtu.be/FVfiX4Hi08o?si=jdKkZGoAbdHaOCtR",
        "https://youtu.be/lmjQ4ymSqnw?si=5BQUC_rtYperm5C-"
    ]

    all_data = []

    for url in video_urls:
        video_id = extract_video_id(url)
        print(f"Fetching comments for video: {video_id}")

        comments = fetch_comments(video_id, max_comments=60)

        for comment in comments:
            all_data.append({
                "video_id": video_id,
                "comment_text": comment
            })

    df = pd.DataFrame(all_data)
    df.to_csv("data/raw/experiment_dataset.csv", index=False)

    print("\nDataset saved as experiment_dataset.csv")
    print(f"Total comments fetched: {len(df)}")