# streamlit_market_trends_with_sentiment_and_groq.py
import os
import streamlit as st
import pickle
import re
import json
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

# NLP & embeddings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Optional: FAISS (used only to build index at runtime, not pickled)
try:
    import faiss
    HAVE_FAISS = True
except Exception:
    HAVE_FAISS = False

# Groq client
from groq import Groq

# ---------------------------
# --- Configuration / Keys -
# ---------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    st.warning("GROQ_API_KEY missing in environment. Groq features will fail until you set it.")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# File storage
EMBEDDINGS_PATH = "embeddings.npy"
ARTICLES_PATH = "articles.pkl"
INSIGHTS_PATH = "insights.txt"

# ---------------------------
# --- Streamlit UI Setup ---
# ---------------------------
st.set_page_config(layout="wide", page_title="Market Trend Analysis (BERT + Groq)")
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Graphs & Insights", "AI-Driven Analysis & Recommendations", "Research Paper"])

# ---------------------------
# --- Cached Resources -----
# ---------------------------
@st.cache_resource(show_spinner=False)
def load_sentence_transformer():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=False)
def load_sentiment_pipeline():
    # BERT sentiment model (cardiffnlp/twitter-roberta-base-sentiment)
    model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, return_all_scores=False)

@st.cache_resource(show_spinner=False)
def build_faiss_index(embeddings):
    if not HAVE_FAISS:
        return None
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings, dtype=np.float32))
    return index

# ---------------------------
# --- Utilities ------------
# ---------------------------
def is_valid_url(url):
    return url.startswith("http://") or url.startswith("https://")

def scrape_articles(urls, timeout=8):
    headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100 Safari/537.36"}
    articles = []
    sources = []
    for url in urls:
        if not url or not is_valid_url(url):
            continue
        try:
            r = requests.get(url, timeout=timeout, headers=headers)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            paragraphs = soup.find_all("p")
            text = " ".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            if not text.strip():
                article_tag = soup.find("article")
                if article_tag:
                    text = article_tag.get_text().strip()
            if text:
                articles.append(text)
                sources.append(url)
        except Exception as e:
            st.error(f"Error fetching {url}: {e}")
    return articles, sources

def save_embeddings_and_articles(embeddings, articles):
    np.save(EMBEDDINGS_PATH, embeddings)
    with open(ARTICLES_PATH, "wb") as f:
        pickle.dump({"articles": articles, "saved_at": datetime.utcnow().isoformat()}, f)

def load_embeddings_and_articles():
    if not (os.path.exists(EMBEDDINGS_PATH) and os.path.exists(ARTICLES_PATH)):
        return None, None
    embeddings = np.load(EMBEDDINGS_PATH)
    with open(ARTICLES_PATH, "rb") as f:
        data = pickle.load(f)
    return embeddings, data.get("articles", [])

def compute_keyword_freq(articles, top_n=20):
    vectorizer = CountVectorizer(stop_words="english", max_features=2000)
    X = vectorizer.fit_transform(articles)
    freqs = np.array(X.sum(axis=0)).flatten()
    vocab = vectorizer.get_feature_names_out()
    idx = np.argsort(freqs)[::-1][:top_n]
    return [(vocab[i], int(freqs[i])) for i in idx]

def compute_market_heat(sentiments, keyword_counts):
    # sentiment scores assumed in [-1,1] or [0,1] depending on pipeline; normalize here
    # Convert sentiments to range 0..1 (if -1..1 -> (s+1)/2, if 0..1 -> keep)
    s = np.array(sentiments, dtype=float)
    if s.min() < 0:
        s = (s + 1) / 2.0
    avg_sent = float(s.mean()) if len(s) else 0.5
    # Keyword signal: more high-frequency market words implies more 'activity'
    kw_signal = min(1.0, sum([c for _, c in keyword_counts[:5]]) / 50.0) if keyword_counts else 0.0
    heat = (0.7 * avg_sent + 0.3 * kw_signal) * 100.0
    return round(float(np.clip(heat, 0, 100)), 2)

# ---------------------------
# --- Core Functions -------
# ---------------------------
def generate_embeddings_for_articles(articles, embed_model):
    return embed_model.encode(articles, convert_to_numpy=True)

def bert_sentiment_scores(articles, sentiment_pipeline):
    """
    Returns numeric sentiment scores in range [-1,1] (negative->-1, neutral->0, positive->1)
    """
    scores = []
    labels_map = {"NEGATIVE": -1.0, "NEUTRAL": 0.0, "POSITIVE": 1.0}
    for a in articles:
        try:
            out = sentiment_pipeline(a[:512])  # limit to first 512 chars for speed
            # pipeline may return {'label': 'LABEL', 'score': 0.9}
            label = out[0]['label'].upper() if isinstance(out, list) else out['label'].upper()
            prob = out[0]['score'] if isinstance(out, list) else out['score']
            val = labels_map.get(label, 0.0) * prob  # weighted by confidence
            scores.append(float(val))
        except Exception:
            scores.append(0.0)
    return scores

def groq_trend_prediction(sentiment_scores, top_keywords, articles, groq_client):
    """
    Ask Groq LLM to predict trend (Uptrend/Downtrend/Neutral) and return text + label + confidence.
    """
    if not groq_client:
        return {"trend": "Unknown", "confidence": 0.0, "explanation": "Groq client not configured."}

    prompt = (
        "You are a market analyst. Given the following information, predict the short-term market trend "
        "(Uptrend / Downtrend / Neutral) and provide a confidence percentage (0-100) and a concise explanation.\n\n"
        "Sentiment scores for 3 articles (range -1 to 1):\n"
        f"{json.dumps(sentiment_scores)}\n\n"
        "Top keywords (word:count):\n"
        f"{json.dumps(top_keywords)}\n\n"
        "Also consider the main article contents when necessary. Provide the output in strict JSON like:\n"
        '{"trend":"Uptrend","confidence":85,"explanation":"..."}\n\n"
        "Articles (short snippets):\n"
    )
    for i, art in enumerate(articles):
        snippet = art.strip().replace("\n", " ")[:400]
        prompt += f"\nArticle {i+1} snippet: {snippet}\n"

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a concise market analyst."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=400
        )
        text = resp.choices[0].message.content.strip()
        # Extract JSON from response (best-effort)
        json_text = None
        try:
            # Try direct JSON parse
            json_text = json.loads(text)
        except Exception:
            # Try to find {...}
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if m:
                try:
                    json_text = json.loads(m.group(0))
                except Exception:
                    json_text = None
        if isinstance(json_text, dict):
            return {
                "trend": json_text.get("trend", "Unknown"),
                "confidence": float(json_text.get("confidence", 0.0)),
                "explanation": json_text.get("explanation", text)
            }
        else:
            # Fallback: put raw text into explanation
            return {"trend": "Unknown", "confidence": 0.0, "explanation": text}
    except Exception as e:
        return {"trend": "Unknown", "confidence": 0.0, "explanation": f"Groq error: {e}"}

# ---------------------------
# --- Home Page -----------
# ---------------------------
if page == "Home":
    st.title("📊 Market Trend Analysis - Data Collection")

    st.markdown("Enter up to 3 news URLs (news articles, blog posts). We'll scrape, embed, analyze sentiment (BERT), "
                "and ask Groq for a short-term trend prediction.")

    urls = [st.text_input(f"Enter News URL {i+1}", key=f"url_{i}") for i in range(3)]
    process_clicked = st.button("Process Articles and Generate Insights")

    if process_clicked:
        with st.spinner("Scraping articles..."):
            articles, sources = scrape_articles(urls)
        if not articles:
            st.error("No valid articles scraped. Check the URLs.")
        else:
            st.success(f"Scraped {len(articles)} articles.")

            # Load models
            embed_model = load_sentence_transformer()
            sentiment_pipe = load_sentiment_pipeline()

            with st.spinner("Generating embeddings..."):
                embeddings = generate_embeddings_for_articles(articles, embed_model)
                save_embeddings_and_articles(embeddings, articles)
                st.success("Embeddings saved.")

            # Build FAISS index in-memory if available (not pickled)
            faiss_idx = build_faiss_index(embeddings) if HAVE_FAISS else None

            # BERT sentiments
            with st.spinner("Computing BERT sentiment scores..."):
                sentiment_scores = bert_sentiment_scores(articles, sentiment_pipe)
            st.write("BERT Sentiment Scores (weighted confidence, -1 to 1):")
            for i, s in enumerate(sentiment_scores):
                st.write(f"Article {i+1}: {s:.3f}")

            # Keywords
            keywords = compute_keyword_freq(articles, top_n=20)
            st.write("Top Keywords:", keywords[:10])

            # Groq trend
            with st.spinner("Asking Groq for trend prediction..."):
                groq_result = groq_trend_prediction(sentiment_scores, keywords, articles, groq_client)
            st.write("Groq Trend Prediction:", groq_result)

            # Market heat
            heat = compute_market_heat(sentiment_scores, keywords)
            st.metric("Market Heat Score (0-100)", f"{heat}")

            # Save insights summary to file for other pages
            insights_payload = {
                "generated_at": datetime.utcnow().isoformat(),
                "sentiment_scores": sentiment_scores,
                "top_keywords": keywords,
                "groq_result": groq_result,
                "market_heat": heat,
                "sources": sources
            }
            with open(INSIGHTS_PATH, "w", encoding="utf-8") as f:
                f.write(json.dumps(insights_payload, indent=2))
            st.success("Insights saved. Go to 'Graphs & Insights' and 'AI-Driven Analysis & Recommendations'.")

# ---------------------------
# --- Graphs & Insights ----
# ---------------------------
elif page == "Graphs & Insights":
    st.title("📈 Market Trends & Graphical Analysis (BERT + Groq)")

    embeddings, articles = load_embeddings_and_articles()
    if not articles or not os.path.exists(INSIGHTS_PATH):
        st.warning("No data available. Go to 'Home' and input data first.")
    else:
        with open(INSIGHTS_PATH, "r", encoding="utf-8") as f:
            insights_payload = json.load(f)

        sentiment_scores = insights_payload.get("sentiment_scores", [])
        keywords = insights_payload.get("top_keywords", [])
        groq_result = insights_payload.get("groq_result", {})
        market_heat = insights_payload.get("market_heat", 0)
        sources = insights_payload.get("sources", [])

        st.header("🔍 Quick Insights")
        st.write(f"Generated at: {insights_payload.get('generated_at')}")
        st.write("Groq Trend:", groq_result.get("trend"), "Confidence:", groq_result.get("confidence"))
        st.info(groq_result.get("explanation"))

        # Sentiment Timeline
        st.subheader("📉 Sentiment Timeline (BERT scores)")
        x = np.arange(1, len(sentiment_scores) + 1)
        s = np.array(sentiment_scores, dtype=float)
        # normalize to -1..1 to plot
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(x, s, marker="o", linestyle="-", linewidth=2)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Article {i}" for i in x])
        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("BERT Sentiment Score (-1..1)")
        ax.set_title("Sentiment per Article")
        st.pyplot(fig)

        # Wordcloud
        st.subheader("🔤 Word Cloud of Combined Articles")
        wc = WordCloud(width=900, height=400, background_color="white")
        wc_img = wc.generate(" ".join(articles))
        st.image(wc_img.to_array(), use_column_width=True)

        # Top Keywords bar
        st.subheader("📊 Top Keywords Frequency")
        if keywords:
            words, counts = zip(*keywords[:15])
            fig, ax = plt.subplots(figsize=(9, 3))
            ax.barh(words[::-1], counts[::-1])
            ax.set_xlabel("Frequency")
            ax.set_title("Top Keywords")
            st.pyplot(fig)
        else:
            st.write("No keywords available.")

        # Market heat and trend
        st.subheader("🔥 Market Heat & Trend")
        st.metric("Market Heat Score (0-100)", f"{market_heat}")
        st.write("Groq Trend Prediction:")
        st.write(f"**{groq_result.get('trend')}** — Confidence: {groq_result.get('confidence')}%")
        st.info(groq_result.get("explanation"))

# ---------------------------
# --- AI Driven Recommendations
# ---------------------------
elif page == "AI-Driven Analysis & Recommendations":
    st.title("🤖 AI-Driven Analysis & Recommendations")

    if not os.path.exists(INSIGHTS_PATH):
        st.warning("No insights available. Please process news articles first.")
    else:
        with open(INSIGHTS_PATH, "r", encoding="utf-8") as f:
            insights_payload = json.load(f)

        sentiment_scores = insights_payload.get("sentiment_scores", [])
        keywords = insights_payload.get("top_keywords", [])
        groq_result = insights_payload.get("groq_result", {})
        market_heat = insights_payload.get("market_heat", 0)

        st.header("📢 AI Summary & Actionable Recommendations")
        st.write("Market Heat:", market_heat)
        st.write("Predicted Trend:", groq_result.get("trend"), "(", groq_result.get("confidence"), "% )")
        st.info(groq_result.get("explanation"))

        # Ask Groq for recommended actions (short list)
        if groq_client:
            with st.spinner("Generating actionable recommendations via Groq..."):
                prompt = (
                    "You are a market analyst. Based on the following summary, give 5 short actionable recommendations "
                    "for an investor or product manager. Keep each recommendation to one short sentence.\n\n"
                    f"Sentiment scores: {json.dumps(sentiment_scores)}\n"
                    f"Top keywords: {json.dumps(keywords[:10])}\n"
                    f"Predicted trend: {groq_result.get('trend')} (confidence {groq_result.get('confidence')})\n\n"
                    "Return as a numbered list."
                )
                try:
                    resp = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": "You are a concise market analyst."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=300
                    )
                    recs = resp.choices[0].message.content.strip()
                    st.subheader("🔄 Actionable Recommendations")
                    st.write(recs)
                except Exception as e:
                    st.error(f"Groq recommendation error: {e}")
        else:
            st.warning("Groq client not configured - cannot generate LLM-based recommendations.")

        # Simple rule-based suggestions using sentiment scores (fallback)
        st.subheader("⚙️ Rule-based Suggestions (quick)")
        avg_sent = np.mean(sentiment_scores) if sentiment_scores else 0
        if avg_sent > 0.2:
            st.success("Overall positive sentiment → Consider bullish strategies (monitor for confirmation).")
        elif avg_sent < -0.2:
            st.warning("Overall negative sentiment → Consider defensive strategies and risk mitigation.")
        else:
            st.info("Neutral/ambiguous sentiment → Gather more data and monitor trends closely.")

# ---------------------------
# --- Research Paper Page ---
# ---------------------------
elif page == "Research Paper":
    st.title("📚 Research Paper Integration")

    st.markdown("""
    **Selected Paper (suggestion to reference in your project):**
    - *Market Trend Prediction using Sentiment Analysis of News Articles* (Example/Template)
    
    **Unique feature added from paper:**  
    - Sentiment timeline visualization and short-term trend prediction using a hybrid approach: numeric BERT sentiment + Groq LLM reasoning.
    """)

    st.subheader("What to include in your report / README")
    st.markdown("""
    1. Briefly describe dataset (the 3 scraped articles) and preprocessing steps.  
    2. Mention your sentiment model: `cardiffnlp/twitter-roberta-base-sentiment` (BERT) to compute numeric sentiment.  
    3. Explain how you combined signals: average sentiment, top keywords, and Groq LLM trend prediction.  
    4. Show the Sentiment Timeline and Market Heat Score (0-100).  
    5. Add recommendations produced by Groq (or fallback rule-based suggestions).
    """)

    # Provide a downloadable research-summary text that the student can include in their writeup
    research_summary = f"""
    Title: Integrating Sentiment Analysis and LLM-based Trend Prediction for Market Insights

    Abstract:
    We implement a hybrid pipeline that scrapes news articles, computes BERT-based sentiment scores,
    extracts top keywords, and queries an LLM (Groq) to predict short-term market trends. We produce
    a sentiment timeline, a market heat score (0-100), and actionable recommendations.

    Methods:
    - Scrape up to 3 news articles provided by user.
    - Compute embeddings using SentenceTransformer (all-MiniLM-L6-v2).
    - Compute numeric sentiment using cardiffnlp/twitter-roberta-base-sentiment (BERT pipeline).
    - Extract top keywords using CountVectorizer.
    - Query Groq LLM for trend prediction, confidence, and recommendations.

    Unique contribution:
    The hybrid approach uses precise numeric sentiment (BERT) for reproducible metrics and LLM reasoning
    (Groq) for contextual interpretation and recommendations — combining the strengths of both.

    Usage:
    - Run the Streamlit app, paste article URLs, click "Process Articles".
    - View graphs, market heat, trend prediction, and download the summary.

    Generated at: {datetime.utcnow().isoformat()}
    """
    st.download_button("Download research summary (.txt)", research_summary, file_name="research_summary.txt")
