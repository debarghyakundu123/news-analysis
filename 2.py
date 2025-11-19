import os, json, pickle, re
import streamlit as st
import requests
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from transformers import pipeline
try:
    import faiss
    HAVE_FAISS = True
except:
    HAVE_FAISS = False
from groq import Groq
from langdetect import detect

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

st.set_page_config(layout="wide", page_title="MarketTrend+Research")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Graphs & Insights", "AI-Driven Analysis", "Research Summary"])

EMB_PATH = "embeddings.npy"
ART_PATH = "articles.pkl"
INS_PATH = "insights.json"

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def load_sentiment_pipeline():
    try:
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    except:
        return pipeline("sentiment-analysis")

def is_url(u):
    return u and (u.startswith("http://") or u.startswith("https://"))

def fetch_url(u, timeout=8):
    headers = {"User-Agent": "Mozilla/5.0"}
    return requests.get(u, timeout=timeout, headers=headers)

def scrape(urls):
    articles = []
    images = []
    for u in urls:
        if not is_url(u):
            articles.append("")
            images.append(None)
            continue
        try:
            r = fetch_url(u)
            r.raise_for_status()
            soup = BeautifulSoup(r.content, "html.parser")
            ps = soup.find_all("p")
            txt = " ".join([p.get_text().strip() for p in ps if p.get_text().strip()])
            if not txt.strip():
                a = soup.find("article")
                if a:
                    txt = a.get_text().strip()
            og = soup.find("meta", property="og:image")
            img = None
            if og and og.get("content"):
                img = og["content"]
            else:
                imgt = soup.find("img")
                if imgt and imgt.get("src"):
                    img = imgt["src"]
            articles.append(txt)
            images.append(img)
        except:
            articles.append("")
            images.append(None)
    return articles, images

def save_data(emb, arts):
    np.save(EMB_PATH, emb)
    with open(ART_PATH, "wb") as f:
        pickle.dump({"arts": arts, "ts": datetime.utcnow().isoformat()}, f)

def load_data():
    if os.path.exists(EMB_PATH) and os.path.exists(ART_PATH):
        emb = np.load(EMB_PATH)
        d = pickle.load(open(ART_PATH, "rb"))
        return emb, d.get("arts", [])
    return None, []

def build_index(emb):
    if not HAVE_FAISS:
        return None
    dim = emb.shape[1]
    idx = faiss.IndexFlatL2(dim)
    idx.add(np.array(emb, dtype=np.float32))
    return idx

def get_keywords(arts, top=25):
    v = CountVectorizer(stop_words="english", max_features=2000)
    X = v.fit_transform(arts)
    freqs = np.array(X.sum(axis=0)).flatten()
    vocab = v.get_feature_names_out()
    idx = np.argsort(freqs)[::-1][:top]
    return [(vocab[i], int(freqs[i])) for i in idx]

def get_topics(arts, n=3):
    v = TfidfVectorizer(stop_words="english", max_features=2000)
    X = v.fit_transform(arts)
    lda = LatentDirichletAllocation(n_components=n, random_state=42)
    lda.fit(X)
    topics = []
    for comp in lda.components_:
        terms = [v.get_feature_names_out()[i] for i in comp.argsort()[-8:][::-1]]
        topics.append(" ".join(terms))
    return topics

def bert_sentiment(arts, sent_pipe):
    scores = []
    labels = []
    for a in arts:
        t = a.strip()
        if not t:
            scores.append(0.0)
            labels.append("NEUTRAL")
            continue
        try:
            out = sent_pipe(t[:512])
        except:
            out = sent_pipe(t)
        if isinstance(out, list):
            out = out[0]
        lab = out.get("label", "NEUTRAL").upper()
        prob = float(out.get("score", 0.0))
        if lab == "POSITIVE":
            val = prob
        elif lab == "NEGATIVE":
            val = -prob
        else:
            val = 0.0
        scores.append(val)
        labels.append(lab)
    return scores, labels

def sentiment_image(img_url, groq_client):
    if not groq_client or not img_url:
        return {"label": "UNKNOWN", "confidence": 0.0, "explanation": "No image or Groq missing"}
    prompt = f"Describe the emotion in this image (Positive/Neutral/Negative) and give confidence. URL: {img_url}"
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "You are an emotion analyst."},
                      {"role": "user", "content": prompt}],
            max_tokens=200
        )
        txt = resp.choices[0].message.content.strip()
        j = None
        try:
            j = json.loads(txt)
        except:
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if m:
                try:
                    j = json.loads(m.group(0))
                except:
                    j = {"label": "UNKNOWN", "confidence": 0.0, "explanation": txt}
        return {"label": j.get("label", "UNKNOWN"), "confidence": float(j.get("confidence", 0.0)), "explanation": j.get("explanation", txt)}
    except Exception as e:
        return {"label": "ERROR", "confidence": 0.0, "explanation": str(e)}

def check_consistency(art, img, groq_client):
    if not groq_client or not img:
        return {"match": "Unknown", "confidence": 0.0, "reason": "No Groq or no image"}
    prompt = f"Text: '''{art[:800].replace(chr(10),' ')}''' Image URL: {img}. Does the image support the text claim? Yes or No. Provide confidence and short reason in JSON."
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "You are a factual analyst."},
                      {"role": "user", "content": prompt}],
            max_tokens=250
        )
        txt = resp.choices[0].message.content.strip()
        j = None
        try:
            j = json.loads(txt)
        except:
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if m:
                try:
                    j = json.loads(m.group(0))
                except:
                    j = {"match": "Unknown", "confidence": 0.0, "reason": txt}
        return {"match": j.get("match", "Unknown"), "confidence": float(j.get("confidence", 0.0)), "reason": j.get("reason", txt)}
    except Exception as e:
        return {"match": "Error", "confidence": 0.0, "reason": str(e)}

def compute_sim_matrix(emb):
    if emb is None:
        return None
    return cosine_similarity(emb)

def predict_trend(sent_scores, keywords, arts, groq_client):
    if not groq_client:
        return {"trend": "Unknown", "confidence": 0.0, "explanation": "No Groq configured"}
    p = "You are a market analyst. Given sentiment scores, top keywords, and article snippets, decide Uptrend/Downtrend/Neutral. JSON output.\n"
    p += "Sentiment: " + json.dumps(sent_scores) + "\n"
    p += "Keywords: " + json.dumps(keywords[:10]) + "\n"
    for i, a in enumerate(arts):
        p += f"Snippet {i+1}: " + a[:300].replace("\n", " ") + "\n"
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "You are concise."},
                      {"role": "user", "content": p}],
            max_tokens=300
        )
        txt = resp.choices[0].message.content.strip()
        j = None
        try:
            j = json.loads(txt)
        except:
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            if m:
                try:
                    j = json.loads(m.group(0))
                except:
                    j = {"trend": "Unknown", "confidence": 0.0, "explanation": txt}
        return {"trend": j.get("trend", "Unknown"), "confidence": float(j.get("confidence", 0.0)), "explanation": j.get("explanation", txt)}
    except Exception as e:
        return {"trend": "Error", "confidence": 0.0, "explanation": str(e)}

def compute_risk(sent_scores, keyword_counts, head_emotion=0.5):
    s = np.array(sent_scores)
    if_neg = np.mean(np.minimum(0, s))
    pos = np.mean(np.maximum(0, s))
    intensity = abs(pos) + abs(if_neg)
    kw_signal = (sum([c for _, c in keyword_counts[:5]]) / 50.0) if keyword_counts else 0
    score = (0.5 * (1 - abs(np.mean(s))) + 0.3 * kw_signal + 0.2 * head_emotion) * 100
    return round(float(np.clip(score, 0, 100)), 2)

def compute_tsi(sent_scores, sim):
    a = np.mean(sent_scores) if sent_scores else 0
    s = np.mean(sim) if sim is not None else 0.5
    return round(float(np.clip(a * 0.7 + s * 0.3, -1, 1)), 3)

def compute_volatility(sent_scores):
    return float(np.std(sent_scores)) if sent_scores else 0.0

def radar_categories(keywords):
    cats = {"economy": 0, "risk": 0, "growth": 0, "politics": 0, "health": 0}
    for w, c in keywords:
        lw = w.lower()
        if any(x in lw for x in ["econom", "market", "stock"]): cats["economy"] += c
        if any(x in lw for x in ["risk", "crash", "decline"]): cats["risk"] += c
        if any(x in lw for x in ["growth", "rise", "profit"]): cats["growth"] += c
        if any(x in lw for x in ["polit", "election", "policy"]): cats["politics"] += c
        if any(x in lw for x in ["health", "covid", "vaccine"]): cats["health"] += c
    vals = list(cats.values())
    maxv = max(vals) if max(vals) > 0 else 1
    return {k: round(v / maxv, 3) for k, v in cats.items()}

# ----------------- UI -------------------

if page == "Home":
    st.title("MarketTrend+ Input")
    urls = [st.text_input(f"URL {i+1}", key=f"url{i}") for i in range(3)]
    proc = st.button("Process")
    if proc:
        arts, imgs = scrape(urls)
        langs = []
        for a in arts:
            try:
                langs.append(detect(a[:200]))
            except:
                langs.append("en")
        embed_model = load_embed_model()
        emb = embed_model.encode(arts, convert_to_numpy=True)
        save_data(emb, arts)
        idx = build_index(emb)
        sent_pipe = load_sentiment_pipeline()
        scores, labs = bert_sentiment(arts, sent_pipe)
        keywords = get_keywords(arts, top=30)
        topics = get_topics(arts, n=3)
        sim = compute_sim_matrix(emb)
        trend = predict_trend(scores, keywords, arts, groq_client)
        img_sent = [sentiment_image(im, groq_client) if im else {"label": "NONE", "confidence": 0.0, "explanation": "No image"} for im in imgs]
        cons = [check_consistency(a, im, groq_client) if im else {"match": "NoImage", "confidence": 0, "reason": "No image"} for a, im in zip(arts, imgs)]
        risk = compute_risk(scores, keywords, head_emotion=0.5)
        tsi_v = compute_tsi(scores, sim)
        vol = compute_volatility(scores)
        heat = round((abs(np.mean(scores)) * 100 + risk) / 2, 2)
        insights = {
            "ts": datetime.utcnow().isoformat(),
            "arts": arts,
            "imgs": imgs,
            "langs": langs,
            "bert_scores": scores,
            "bert_labels": labs,
            "keywords": keywords,
            "topics": topics,
            "sim": sim.tolist() if sim is not None else None,
            "trend": trend,
            "img_sent": img_sent,
            "consistency": cons,
            "risk": risk,
            "tsi": tsi_v,
            "volatility": vol,
            "heat": heat
        }
        with open(INS_PATH, "w", encoding="utf-8") as f:
            json.dump(insights, f, indent=2)
        st.success("Done processing")
                st.markdown("---")
        st.subheader("Ask a Question Based on the Articles")

        user_q = st.text_input("Enter your question:", key="user_question_box")
        ask_btn = st.button("Generate Answer")

        if ask_btn and user_q.strip():
            try:
                # Build a context from the scraped articles
                context = ""
                for i, a in enumerate(arts):
                    context += f"Article {i+1}: {a[:800].replace(chr(10),' ')}\n"

                prompt_q = (
                    "You are a research assistant. Answer the user's question ONLY "
                    "based on the given articles. Provide a clear, factual answer.\n\n"
                    f"ARTICLES:\n{context}\n\n"
                    f"QUESTION: {user_q}\n"
                    "ANSWER:"
                )

                if groq_client:
                    resp = groq_client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": "You are helpful and factual."},
                                  {"role": "user", "content": prompt_q}],
                        max_tokens=350
                    )
                    answer = resp.choices[0].message.content.strip()
                else:
                    answer = "Groq not configured. Cannot generate answer."

                st.subheader("Answer:")
                st.write(answer)

            except Exception as e:
                st.error(f"Error generating answer: {e}")


elif page == "Graphs & Insights":
    st.title("Graphs & Insights")
    if not os.path.exists(INS_PATH):
        st.warning("Process first")
        st.stop()
    with open(INS_PATH, "r", encoding="utf-8") as f:
        ins = json.load(f)
    arts = ins["arts"]
    bert_scores = ins["bert_scores"]
    keywords = ins["keywords"]
    topics = ins["topics"]
    sim = np.array(ins["sim"]) if ins.get("sim") else None

    st.subheader("Sentiment Timeline")
    x = np.arange(1, len(bert_scores) + 1)
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, bert_scores, marker="o")
    ax.axhline(0, linestyle="--", color="gray")
    ax.set_xticks(x)
    ax.set_xticklabels([f"A{i}" for i in x])
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("BERT score")
    st.pyplot(fig)

    st.subheader("Similarity Heatmap")
    if sim is not None:
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(sim, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(x - 1)
        ax.set_yticks(x - 1)
        ax.set_xticklabels([f"A{i}" for i in x])
        ax.set_yticklabels([f"A{i}" for i in x])
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)
    else:
        st.info("Similarity not available")

    st.subheader("Top Keywords")
    w, c = zip(*keywords[:20])
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.barh(list(w)[::-1], list(c)[::-1])
    st.pyplot(fig)

    st.subheader("Word Cloud")
    wc = WordCloud(width=900, height=400, background_color="white").generate(" ".join(arts))
    st.image(wc.to_array(), use_column_width=True)

    st.subheader("Topics (LDA)")
    for i, t in enumerate(topics):
        st.write(f"Topic {i+1}: {t}")

    st.subheader("Keyword Radar")
    rad = radar_categories(keywords)
    labels = list(rad.keys())
    vals = list(rad.values())
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    vals += vals[:1]
    angles += angles[:1]
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals, marker="o")
    ax.fill(angles, vals, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    st.pyplot(fig)

    st.subheader("Image–Text Consistency & Sentiment")
    for i, (c, im) in enumerate(zip(ins["consistency"], ins["imgs"])):
        st.write(f"Article {i+1} consistency: {c.get('match')} (conf {c.get('confidence')}) Reason: {c.get('reason')}")
        if im:
            st.image(im, width=300)
    for i, isent in enumerate(ins["img_sent"]):
        st.write(f"Image {i+1} → {isent.get('label')} (conf {isent.get('confidence')}) expl: {isent.get('explanation')[:120]}")

    st.subheader("Risk / Heat / TSI / Volatility")
    st.metric("Risk Score", ins["risk"])
    st.metric("Market Heat", ins["heat"])
    st.metric("TSI", ins["tsi"])
    st.metric("Volatility", ins["volatility"])

elif page == "AI-Driven Analysis":
    st.title("AI Driven Analysis & Recommendations")
    if not os.path.exists(INS_PATH):
        st.warning("Process first")
        st.stop()
    with open(INS_PATH, "r", encoding="utf-8") as f:
        ins = json.load(f)
    st.write("Trend Prediction:", ins["trend"])
    avg_sent = np.mean(ins["bert_scores"]) if ins["bert_scores"] else 0
    final_score = 0.6 * ((avg_sent + 1) / 2) + 0.3 * (ins["risk"] / 100) + 0.1 * ((ins["tsi"] + 1) / 2)
    final_pct = round(final_score * 100, 2)
    if final_pct > 60:
        recommendation = "BUY"
    elif final_pct < 40:
        recommendation = "SELL"
    else:
        recommendation = "HOLD"
    st.metric("Recommendation", recommendation)
    st.metric("Confidence %", final_pct)
    st.subheader("Explain why (Groq)")
    if groq_client:
        prompt = (
            "Explain in 4 bullet points why the trend is chosen given the signals: "
            + json.dumps(ins["bert_scores"])
            + " and top keywords: "
            + json.dumps(ins["keywords"][:10])
        )
        try:
            resp = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": "You are practical."},
                          {"role": "user", "content": prompt}],
                max_tokens=250
            )
            st.write(resp.choices[0].message.content.strip())
        except Exception as e:
            st.write("Groq error:", e)
    else:
        st.write("Groq not set. Here is rule-based reasoning:")
        st.write("-", f"Avg sentiment {round(avg_sent,3)}")
        st.write("-", f"Top keyword {ins['keywords'][0][0] if ins['keywords'] else 'N/A'}")
        st.write("-", f"Risk {ins['risk']}")
        st.write("-", f"Volatility {ins['volatility']}")

    st.subheader("Actionable Recommendations")
    if groq_client:
        p2 = (
            "Give 5 short actionable recommendations for an investor or manager based on trend "
            + json.dumps(ins["trend"])
            + " and signals."
        )
        try:
            resp2 = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": "You are strategic."},
                          {"role": "user", "content": p2}],
                max_tokens=250
            )
            st.write(resp2.choices[0].message.content.strip())
        except Exception as e:
            st.write("Groq error:", e)
    else:
        st.write("1. Monitor sentiment trends.\n2. Hedge on risk signals.\n3. Diversify.\n4. Stay alert to story mismatch.\n5. Update data often.")

elif page == "Research Summary":
    st.title("Research / Paper Integration Summary")
    st.write("We integrated ideas from multimodal fake-news detection research (e.g. dataset challenges, modality fusion, real-time, explainability).")
    summary = (
        f"Integration done at {datetime.utcnow().isoformat()}\n"
        "- Text–image consistency check\n"
        "- Multimodal sentiment (text + image)\n"
        "- Article similarity heatmap\n"
        "- Topic modeling via LDA\n"
        "- Keyword radar chart\n"
        "- Risk, TSI, Volatility metrics\n"
        "- Explainability via Groq LLM\n"
        "- Multilingual detection (langdetect)"
    )
    st.download_button("Download summary", summary, file_name="research_integration.txt")
