from flask import Flask, render_template, request

from src.inference.inference_service import analyze_comments
from src.utils.youtube_fetcher import extract_video_id, fetch_comments  


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    summary = None
    pairs = None
    comments = None

    if request.method == "POST":

        video_url = request.form["video_url"]
        video_id = extract_video_id(video_url)
        comments = fetch_comments(video_id) if video_id else None
        
        if comments:
            summary, pairs = analyze_comments(comments)

    return render_template(
        "index.html",
        summary=summary,
        pairs=pairs
    )


if __name__ == "__main__":
    app.run(debug=True)