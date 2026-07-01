import os
import json
import pathlib
import requests
import praw
import pandas as pd
import functools
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, abort, request, render_template, url_for, session, redirect, flash
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from google_auth_oauthlib.flow import Flow
from google.oauth2 import id_token
from google.auth.transport.requests import Request as GoogleRequest
from bs4 import BeautifulSoup

# ── Google OAuth Config ────────────────────────────────────────────────────────
# Credentials come from environment variables in production (Vercel).
# Locally, fall back to reading client_secret.json (which is git-ignored).
GOOGLE_CLIENT_ID     = os.environ.get("GOOGLE_CLIENT_ID",     "739799506696-5m5gmv9ranho8ci2g7h8lcgcubeuamcc.apps.googleusercontent.com")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")

_LOCAL_SECRET_FILE = os.path.join(pathlib.Path(__file__).parent, "client_secret.json")

def _build_client_config():
    """
    Return a dict that Flow.from_client_config() accepts.
    In production GOOGLE_CLIENT_SECRET env var is set.
    Locally we read client_secret.json directly.
    """
    if GOOGLE_CLIENT_SECRET:
        # Production path — no file needed
        return {
            "web": {
                "client_id":                GOOGLE_CLIENT_ID,
                "client_secret":            GOOGLE_CLIENT_SECRET,
                "auth_uri":                 "https://accounts.google.com/o/oauth2/auth",
                "token_uri":                "https://oauth2.googleapis.com/token",
                "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
                "redirect_uris":            [OAUTH_REDIRECT_URI],
            }
        }
    # Local dev path — read from file
    with open(_LOCAL_SECRET_FILE) as f:
        return json.load(f)

# ── API Credentials ────────────────────────────────────────────────────────────
reddit = praw.Reddit(
    client_id='-mi82_n8q6dN9o4RiUGoKQ',
    client_secret='hkf_DBqNC0yqsfTEnhC6i-PBfYZqHA',
    user_agent='Product_Sentineo'
)
NEWS_API_KEY        = 'f1c1b88c5f55436596e8df23d2a7649b'
EVENT_REGISTRY_KEY  = '354393d0-e38d-4735-9b99-fd99a8edc17f'
TUMBLR_API_KEY      = 'aeFN2SHfTT0Wz4frlZNRKHFa9Zxo6tfgrRpjxJhXmwOCZvOR6R'

# ── Flask App ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "brand-sentineo-secret-key-2025")

# OAuth redirect URI — override via environment variable in Vercel dashboard:
#   OAUTH_REDIRECT_URI = https://your-app.vercel.app/callback
OAUTH_REDIRECT_URI = os.environ.get(
    "OAUTH_REDIRECT_URI",
    "http://127.0.0.1:5000/callback"   # local dev default
)

# Allow plain-HTTP OAuth only in local dev (not on Vercel — it's always HTTPS)
if not os.environ.get("VERCEL"):
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"

# ── Sentiment Analyzer ─────────────────────────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def login_required(f):
    @functools.wraps(f)
    def decorated(*args, **kwargs):
        if "google_id" not in session:
            return redirect(url_for("home"))
        return f(*args, **kwargs)
    return decorated


def _get_canonical_url():
    """
    Return the full request URL with the correct scheme.
    Vercel terminates TLS at its edge and forwards requests to the function
    over HTTP, but sets the X-Forwarded-Proto header to 'https'.
    """
    url = request.url
    proto = request.headers.get("X-Forwarded-Proto", "")
    if proto == "https" and url.startswith("http://"):
        url = "https://" + url[len("http://"):]
    return url


# ─────────────────────────────────────────────────────────────────────────────
# Sentiment Analysis Functions  (all run in parallel via ThreadPoolExecutor)
# ─────────────────────────────────────────────────────────────────────────────

def reddit_analysis(key):
    """Fetch Reddit posts and compute weighted sentiment score."""
    posts = []
    try:
        # limit=25 keeps the call well within Vercel's 60-second window
        for post in reddit.subreddit("all").search(key, limit=25):
            upvotes  = max(post.score, 0)
            total    = upvotes + abs(post.score)
            downvotes = int((1 - post.upvote_ratio) * total) if total else 0
            votes_sum = upvotes + downvotes
            if votes_sum == 0:
                sentiment = analyzer.polarity_scores(post.title)["compound"] * 0.2
            else:
                sentiment = (
                    0.2 * analyzer.polarity_scores(post.title)["compound"]
                    + (0.5 * upvotes) / votes_sum
                    - 0.3 * downvotes / votes_sum
                )
            posts.append({"sentiment": sentiment})
    except Exception as e:
        print(f"[Reddit] Error: {e}")
        return 0.0

    df = pd.DataFrame(posts)
    return round(float(df["sentiment"].mean()), 4) if not df.empty else 0.0


def news_analysis(key):
    """Fetch NewsAPI articles and compute average VADER compound score."""
    try:
        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={key}&apiKey={NEWS_API_KEY}&pageSize=50"
        )
        resp = requests.get(url, timeout=12)
        articles = resp.json().get("articles", [])
    except Exception as e:
        print(f"[News] Error: {e}")
        return 0.0

    scores = []
    for art in articles:
        text = " ".join(filter(None, [
            art.get("title") or "",
            art.get("description") or "",
            art.get("content") or "",
        ]))
        scores.append(analyzer.polarity_scores(text)["compound"])

    return round(float(sum(scores) / len(scores)), 4) if scores else 0.0


def event_registry_analysis(key):
    """Fetch EventRegistry articles and compute VADER score on body text."""
    try:
        url = (
            f"https://eventregistry.org/api/v1/article/getArticles"
            f"?query=%7B%22%24query%22%3A%7B%22keyword%22%3A%22{key}%22%2C%22lang%22%3A%22eng%22%7D"
            f"%2C%22%24filter%22%3A%7B%22forceMaxDataTimeWindow%22%3A%2231%22%7D%7D"
            f"&resultType=articles&articlesSortBy=date&articlesPage=1&articlesCount=5"
            f"&apiKey={EVENT_REGISTRY_KEY}"
        )
        resp    = requests.get(url, timeout=12)
        results = resp.json().get("articles", {}).get("results", [])
    except Exception as e:
        print(f"[EventRegistry] Error: {e}")
        return 0.0

    scores = [
        analyzer.polarity_scores(art.get("body", ""))["compound"]
        for art in results if art.get("body")
    ]
    return round(float(sum(scores) / len(scores)), 4) if scores else 0.0


def tumblr_analysis(key):
    """Fetch Tumblr text posts and compute VADER score."""
    try:
        url  = f"https://api.tumblr.com/v2/tagged?tag={key}&api_key={TUMBLR_API_KEY}"
        resp = requests.get(url, timeout=12)
        posts = resp.json().get("response", [])
    except Exception as e:
        print(f"[Tumblr] Error: {e}")
        return 0.0

    texts = []
    for post in posts:
        if post.get("type") == "text" and "body" in post:
            soup = BeautifulSoup(post["body"], "html.parser")
            for tag in soup(["figure", "img"]):
                tag.decompose()
            text = soup.get_text(separator=" ", strip=True)
            if text:
                texts.append(text)

    if not texts:
        return 0.0

    scores = [analyzer.polarity_scores(t)["compound"] for t in texts]
    return round(float(sum(scores) / len(scores)), 4)


def run_all_analyses(keyword):
    """
    Run all 4 API calls concurrently using threads.
    Total wall-clock time ≈ slowest individual call (~10-15 s)
    instead of the sum (~40-60 s) if run sequentially.
    """
    tasks = {
        "reddit": reddit_analysis,
        "news":   news_analysis,
        "event":  event_registry_analysis,
        "tumblr": tumblr_analysis,
    }
    results = {}
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(fn, keyword): name for name, fn in tasks.items()}
        for future in as_completed(futures):
            name = futures[future]
            try:
                results[name] = future.result()
            except Exception as e:
                print(f"[{name}] Unhandled error: {e}")
                results[name] = 0.0
    return results


def combined_score(reddit, news, event, tumblr):
    return round(0.30 * reddit + 0.30 * news + 0.25 * event + 0.15 * tumblr, 4)


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    if "google_id" in session:
        return redirect(url_for("search"))
    return render_template("login.html")


@app.route("/google_login")
def google_login():
    flow = Flow.from_client_config(
        _build_client_config(),
        scopes=[
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/userinfo.email",
            "openid",
        ],
        redirect_uri=OAUTH_REDIRECT_URI,
    )
    authorization_url, state = flow.authorization_url()
    session["state"] = state
    return redirect(authorization_url)


@app.route("/callback")
def callback():
    if session.get("state") != request.args.get("state"):
        abort(401)

    flow = Flow.from_client_config(
        _build_client_config(),
        scopes=[
            "https://www.googleapis.com/auth/userinfo.profile",
            "https://www.googleapis.com/auth/userinfo.email",
            "openid",
        ],
        redirect_uri=OAUTH_REDIRECT_URI,
    )

    # Use the canonical (https) URL so the token exchange matches the redirect_uri
    flow.fetch_token(authorization_response=_get_canonical_url())
    credentials = flow.credentials

    id_info = id_token.verify_oauth2_token(
        credentials.id_token,
        GoogleRequest(),
        GOOGLE_CLIENT_ID,
    )

    session["google_id"] = id_info.get("sub")
    session["email"]     = id_info.get("email")
    session["name"]      = id_info.get("name", "User")
    session["picture"]   = id_info.get("picture", "")

    return redirect(url_for("search"))


@app.route("/search", methods=["GET", "POST"])
@login_required
def search():
    if request.method == "POST":
        keyword = request.form.get("keyword", "").strip()
        if not keyword:
            flash("Please enter a brand or product name.", "warning")
            return redirect(url_for("search"))

        # All 4 APIs called in parallel
        scores  = run_all_analyses(keyword)
        overall = combined_score(
            scores["reddit"], scores["news"],
            scores["event"],  scores["tumblr"],
        )

        return render_template(
            "results.html",
            keyword=keyword,
            reddit=scores["reddit"],
            news=scores["news"],
            event=scores["event"],
            tumblr=scores["tumblr"],
            overall=overall,
            user_name=session.get("name", ""),
            user_picture=session.get("picture", ""),
        )

    return render_template(
        "search.html",
        user_name=session.get("name", ""),
        user_picture=session.get("picture", ""),
    )


@app.route("/logout")
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("home"))


if __name__ == "__main__":
    app.run(debug=True)
