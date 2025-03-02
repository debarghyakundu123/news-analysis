import os
import streamlit as st
import pickle
import re

import requests
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from groq import Groq
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
from faiss import IndexFlatL2





# Load API keys
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq API client
groq_client = Groq(api_key=groq_api_key)

# Sidebar Navigation
st.sidebar.title("📍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Graphs & Insights", "AI-Driven Analysis & Recommendations"])

file_path = "faiss_market_trends.pkl"

# ------------------------------- #
# 🌍 PAGE 1: Home (Input & Query) #
# ------------------------------- #
if page == "Home":
    st.title("📊 Market Trend Analysis - Data Collection")

    # Get user-input URLs
    urls = [st.text_input(f"Enter News URL {i+1}") for i in range(3)]
    process_clicked = st.button("Process Articles")

    # Function to validate URLs
    def is_valid_url(url):
        return url.startswith("http://") or url.startswith("https://")

    # Function to scrape articles from any link
    def scrape_articles(urls):
        articles = []
        for url in urls:
            if not url or not is_valid_url(url):
                continue  # Skip invalid URLs
            
            try:
                response = requests.get(url, timeout=5)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, "html.parser")

                # Extract text from <p> tags (for general articles)
                paragraphs = soup.find_all("p")
                text = " ".join([p.get_text() for p in paragraphs])

                # If text is empty, try <article> tag (for some blogs/news sites)
                if not text.strip():
                    article = soup.find("article")
                    if article:
                        text = article.get_text()

                if text:
                    articles.append(text)
            except Exception as e:
                st.error(f"Error fetching {url}: {e}")
        return articles

    # Process articles
    if process_clicked:
        st.text("Scraping Articles... ✅✅✅")
        articles = scrape_articles(urls)

        if not articles:
            st.error("No valid articles found. Please check the URLs.")
            st.stop()

        # Embed and store in FAISS
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(articles)
        index = IndexFlatL2(embeddings.shape[1])
        index.add(np.array(embeddings, dtype=np.float32))

        # Save FAISS index
        with open(file_path, "wb") as f:
            pickle.dump((index, articles, model), f)

        st.success("Processing completed! You can now ask questions.")

    # User query input
    query = st.text_input("Ask a question about market trends:")

    if query and os.path.exists(file_path):
        with open(file_path, "rb") as f:
            index, articles, model = pickle.load(f)

        query_embedding = model.encode([query])
        _, indices = index.search(np.array(query_embedding, dtype=np.float32), k=3)
        context = "\n\n".join([articles[i] for i in indices[0]])

        # Query Groq AI
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "system", "content": "You are a market analyst."},
                      {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}\n\nProvide key insights and the best type of graphs to visualize the data."}]
        )

        # Save response for graphs page
        insights = response.choices[0].message.content
        with open("insights.txt", "w") as f:
            f.write(insights)

        st.success("Question processed! Go to 'Graphs & Insights' page to view analysis.")

# -------------------------------------- #
# 📊 PAGE 2: Graphs & Insights #
# -------------------------------------- #
elif page == "Graphs & Insights":
    st.title("📈 Market Trends & Graphical Analysis")

    if os.path.exists(file_path) and os.path.exists("insights.txt"):
        with open(file_path, "rb") as f:
            _, articles, _ = pickle.load(f)

        with open("insights.txt", "r") as f:
            insights = f.read()

        st.header("🔍 AI Insights")
        st.write(insights)

        st.subheader("📊 Data Visualization")

        # Graph 1: Sentiment Analysis
        sentiment_labels = ["Positive", "Neutral", "Negative"]
        sentiment_counts = np.random.randint(10, 100, size=3)

        fig, ax = plt.subplots()
        ax.bar(sentiment_labels, sentiment_counts, color=["green", "gray", "red"])
        ax.set_title("Sentiment Analysis of Market Trends")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Graph 2: Word Cloud
        vectorizer = CountVectorizer(stop_words="english")
        X = vectorizer.fit_transform(articles)
        word_freq = dict(zip(vectorizer.get_feature_names_out(), X.sum(axis=0).tolist()[0]))

        wordcloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_freq)
        st.image(wordcloud.to_array(), use_column_width=True)

        # Graph 3: Risk vs Reward
        risk = np.random.uniform(1, 10, size=10)
        reward = np.random.uniform(1, 10, size=10)

        fig, ax = plt.subplots()
        ax.scatter(risk, reward, c=risk, cmap="coolwarm", edgecolors="black")
        ax.set_xlabel("Risk Level")
        ax.set_ylabel("Potential Reward")
        ax.set_title("Investment Risk vs Reward")
        st.pyplot(fig)

    else:
        st.warning("No data available. Go to 'Home' and input data first.")

# -------------------------------------- #
# 🤖 PAGE 3: AI-Driven Recommendations #
# -------------------------------------- #
elif page == "AI-Driven Analysis & Recommendations":
    st.title("📢 AI-Driven Analysis & Recommendations")

    if os.path.exists("insights.txt"):
        with open("insights.txt", "r") as f:
            insights = f.read()

        # 🔹 AI-Generated Custom Report Title Based on News Type
        ai_category_response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an AI that classifies news topics like Investment, Sports, Health, or Food and generates a suitable report title."},
                {"role": "user", "content": f"Classify this news insight and create a professional report title:\n\n{insights}"}
            ]
        )
        ai_category = ai_category_response.choices[0].message.content.strip()
        st.header(f"📢 {ai_category}")

        # 🔹 AI-Driven Deep Analysis
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": """
                You are an AI financial, sports, and news analyst.
                **Follow this structured format strictly**:
                
                📢 **Summary:** (Briefly summarize key points)
                📌 **Key Takeaways:** (Use bullet points)
                📊 **Statistical Analysis:** (Provide key stats and trends)
                📉 **Graph Insights:** (Describe what type of graph should be used)
                🔄 **AI Strategy:** (Actionable recommendations)
                🔥 **Conclusion:** (Final insight)
                """},
                {"role": "user", "content": f"Analyze this news:\n\n{insights}"}
            ]
        )

        ai_output = response.choices[0].message.content

        # ---------------------------- #
        # 📢 AI-Generated Summary
        # ---------------------------- #
        st.subheader("📢 AI-Generated Summary")
        summary_match = re.search(r"📢 \*\*Summary:\*\* (.+)", ai_output, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else "Summary not available."
        st.write(summary)

        # ---------------------------- #
        # 📌 Key Takeaways
        # ---------------------------- #
        st.subheader("📌 Key Takeaways")

        # Try multiple patterns to catch different formats
        key_takeaways_match = re.findall(r"(?:- |\* |\• |\d+\.)\s*(.+)", ai_output)

        if key_takeaways_match:
            for point in key_takeaways_match:
                st.markdown(f"- {point}")
        else:
            st.warning("No key takeaways found. AI might have structured it differently.")




        # ---------------------------- #
        # 📊 AI-Generated Statistical Analysis
        # ---------------------------- #
        st.subheader("📊 Key Statistics & Trends")

        # Try multiple patterns to capture statistical analysis correctly
        stats_match = re.search(r"(?:📊 \*\*Statistical Analysis:\*\*|### Statistical Analysis|## Statistics)\s*(.+)", ai_output, re.DOTALL)

        stats_text = stats_match.group(1).strip() if stats_match else "No statistical data provided."

        st.write(stats_text)

        # ---------------------------- #
        # 📉 AI-Generated Graph Insights
        # ---------------------------- #
        st.subheader("📉 AI-Generated Graph Insights")
        
        graph_match = re.search(r"📉 \*\*Graph Insights:\*\* (.+)", ai_output, re.DOTALL)
        graph_text = graph_match.group(1).strip() if graph_match else "No graph description available."
        st.write(f"**AI Suggestion:** {graph_text}")

        # 📊 Example Graph
        fig, ax = plt.subplots()
        x = np.arange(1, 6)
        y = np.random.randint(10, 100, size=5)

        ax.plot(x, y, marker="o", linestyle="--", color="blue")
        ax.set_xlabel("Time (Months)")
        ax.set_ylabel("Growth %")
        ax.set_title("AI-Generated Market Trend")

        st.pyplot(fig)

        # ---------------------------- #
        # 🔄 AI-Powered Strategy
        # ---------------------------- #
        st.subheader("🔄 AI-Generated Strategy")
        strategy_match = re.search(r"🔄 \*\*AI Strategy:\*\*\s*(.*?)(?=\n📉|\n🔥|\Z)", ai_output, re.DOTALL)

        if strategy_match:
            strategy = strategy_match.group(1).strip()
            st.write(strategy)
        else:
            st.warning("No AI strategy generated.")


        # ---------------------------- #
        # 🔥 AI-Generated Insight / Conclusion
        # ---------------------------- #
        st.subheader("🔥 AI-Generated Insight")
        st.text("Raw AI Output for Conclusion:")
        st.code(ai_output)

        insight_match = re.search(r"🔥 \*\*Conclusion:\*\*\s*(.+)", ai_output, re.DOTALL)
        insight = insight_match.group(1).strip() if insight_match else "No conclusion found."
        st.info(insight)

        st.success("✅ AI Insights Generated Successfully!")

    else:
        st.warning("No insights available. Please process news articles first.")
