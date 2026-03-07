from flask import Flask, render_template, request

from src.inference.inference_service import analyze_comments


app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    summary = None
    replies = None

    if request.method == "POST":

        comments_text = request.form["comments"]

        comments = comments_text.split("\n")

        summary, replies = analyze_comments(comments)

    return render_template(
        "index.html",
        summary=summary,
        replies=replies
    )


if __name__ == "__main__":
    app.run(debug=True)