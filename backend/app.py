import os, requests, time
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Backend is running"

@app.route("/sentiments")
def sentiments():
    ticker = request.args.get("ticker")
    
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={API_KEY}&pageSize=20"
    response = requests.get(url)
    data = response.json()
    
    articles = data.get("articles") or []
    headlines_fetched = len(articles)
    results = []
    analyzer = SentimentIntensityAnalyzer()
    t0 = time.perf_counter()
    for item in articles:
        if (item["title"] is None):
            continue
        
        report = analyzer.polarity_scores(item["title"])
        score = report["compound"]

        if score > .05:
            results.append({"headline" : item["title"], "sentiment" : "positive", "score" : score})
        elif score < -.05:
            results.append({"headline" : item["title"], "sentiment" : "negative", "score" : score})
        else:
            results.append({"headline" : item["title"], "sentiment" : "neutral", "score" : score})

    sentiment_analysis_ms = (time.perf_counter() - t0) * 1000
    headlines_analyzed = len(results)
    print(
        f"[sentiments] ticker={ticker!r} | headlines fetched: {headlines_fetched} | "
        f"headlines analyzed: {headlines_analyzed} | "
        f"sentiment analysis latency: {sentiment_analysis_ms:.2f} ms"
    )

    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)