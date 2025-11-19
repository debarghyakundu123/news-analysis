# app.py
import os,sys,json,re,pickle
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
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
try:
    import faiss
    HAVE_FAISS=True
except:
    HAVE_FAISS=False
from groq import Groq
from langdetect import detect
load_dotenv()
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
groq_client=Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None
st.set_page_config(layout="wide",page_title="MarketTrend+Research")
st.sidebar.title("Nav")
page=st.sidebar.radio("Go to",["Home","Graphs & Insights","AI-Driven Analysis & Recommendations","Research Paper"])
EMB_PATH="emb.npy";ART_PATH="arts.pkl";INS_PATH="insights.json"
@st.cache_resource
def load_embed(): return SentenceTransformer("all-MiniLM-L6-v2")
@st.cache_resource
def load_sentiment(): 
    try:
        return pipeline("sentiment-analysis",model="distilbert-base-uncased-finetuned-sst-2-english")
    except:
        return pipeline("sentiment-analysis")
def is_url(u): return u and (u.startswith("http://") or u.startswith("https://"))
def fetch(url,timeout=8):
    h={"User-Agent":"Mozilla/5.0"}
    r=requests.get(url,timeout=timeout,headers=h); r.raise_for_status(); return r
def scrape(urls):
    arts=[]; imgs=[]
    for u in urls:
        if not is_url(u): continue
        try:
            r=fetch(u)
            s=BeautifulSoup(r.content,"html.parser")
            ps=s.find_all("p"); txt=" ".join([p.get_text().strip() for p in ps if p.get_text().strip()])
            if not txt.strip():
                a=s.find("article")
                if a: txt=a.get_text().strip()
            # try to find main image
            im=None
            og=s.find("meta",property="og:image")
            if og and og.get("content"): im=og.get("content")
            if not im:
                img=s.find("img")
                if img and img.get("src"): im=img.get("src")
            arts.append(txt if txt else "")
            imgs.append(im)
        except Exception as e:
            arts.append("") ; imgs.append(None)
    return arts,imgs
def save_embeddings(e,arts):
    np.save(EMB_PATH,e)
    with open(ART_PATH,"wb") as f: pickle.dump({"arts":arts,"ts":datetime.utcnow().isoformat()},f)
def load_embeddings():
    if os.path.exists(EMB_PATH) and os.path.exists(ART_PATH):
        e=np.load(EMB_PATH); d=pickle.load(open(ART_PATH,"rb")); return e,d.get("arts",[])
    return None,[]
def build_faiss(e):
    if not HAVE_FAISS: return None
    d=e.shape[1]; idx=faiss.IndexFlatL2(d); idx.add(np.array(e,dtype=np.float32)); return idx
def keyword_freq(arts,top=25):
    v=CountVectorizer(stop_words="english",max_features=2000); X=v.fit_transform(arts); freqs=np.array(X.sum(axis=0)).flatten(); vocab=v.get_feature_names_out()
    idx=np.argsort(freqs)[::-1][:top]; return [(vocab[i],int(freqs[i])) for i in idx]
def topic_model(arts,n=3):
    v=TfidfVectorizer(stop_words="english",max_features=2000); X=v.fit_transform(arts)
    lda=LatentDirichletAllocation(n_components=n,random_state=42); lda.fit(X)
    topics=[]
    for comp in lda.components_:
        terms=[v.get_feature_names_out()[i] for i in comp.argsort()[-8:][::-1]]
        topics.append(" ".join(terms))
    return topics
def sentiment_bert(arts,sent_pipe):
    scores=[]; labels=[]
    for a in arts:
        t=a.strip()
        if not t:
            scores.append(0.0); labels.append("NEUTRAL"); continue
        try:
            out=sent_pipe(t[:512])
            if isinstance(out,list): out=out[0]
            lab=out.get("label","NEUTRAL").upper(); prob=float(out.get("score",0.0))
            val=prob if lab=="POSITIVE" else -prob if lab=="NEGATIVE" else 0.0
            scores.append(val); labels.append(lab)
        except:
            scores.append(0.0); labels.append("NEUTRAL")
    return scores,labels
def sentiment_image_with_groq(img_url,groq_client):
    if not groq_client or not img_url: return {"label":"UNKNOWN","score":0.0,"explain":"No groq or image"}
    prompt=f"Describe the mood/emotion of this image and give Positive/Neutral/Negative with confidence. Image URL: {img_url}. Return JSON with label,confidence,explanation."
    try:
        r=groq_client.chat.completions.create(model="llama-3.3-70b-versatile",messages=[{"role":"system","content":"You are concise."},{"role":"user","content":prompt}],max_tokens=200)
        txt=r.choices[0].message.content.strip()
        j=None
        try: j=json.loads(txt)
        except:
            m=re.search(r"\{.*\}",txt,re.DOTALL)
            if m:
                try: j=json.loads(m.group(0))
                except: j={"label":"UNKNOWN","confidence":0,"explanation":txt}
        return {"label":j.get("label","UNKNOWN") if isinstance(j,dict) else "UNKNOWN","score":float(j.get("confidence",0)) if isinstance(j,dict) else 0.0,"explain":j.get("explanation",txt) if isinstance(j,dict) else txt}
    except Exception as e:
        return {"label":"ERROR","score":0.0,"explain":str(e)}
def image_text_consistency(art,img,groq_client):
    if not groq_client: return {"match":"Unknown","confidence":0,"reason":"Groq missing"}
    prompt=f"Given this article text snippet: '''{art[:800].replace('\\n',' ')}''' and image URL: {img}, answer: Does the image content support the text claim? Return JSON {{'match':'Yes'/'No','confidence':int,'reason':'short'}}"
    try:
        r=groq_client.chat.completions.create(model="llama-3.3-70b-versatile",messages=[{"role":"system","content":"You are a factual analyst."},{"role":"user","content":prompt}],max_tokens=250)
        txt=r.choices[0].message.content.strip()
        j=None
        try: j=json.loads(txt)
        except:
            m=re.search(r"\{.*\}",txt,re.DOTALL)
            if m:
                try: j=json.loads(m.group(0))
                except: j={"match":"Unknown","confidence":0,"reason":txt}
        return {"match":j.get("match","Unknown"),"confidence":float(j.get("confidence",0)),"reason":j.get("reason",txt)}
    except Exception as e:
        return {"match":"Error","confidence":0,"reason":str(e)}
def compute_similarity_matrix(emb):
    if emb is None: return None
    sim=cosine_similarity(emb); return sim
def trend_predict_groq(sent_scores,keywords,arts,groq_client):
    if not groq_client: return {"trend":"Unknown","confidence":0,"explain":"No groq"}
    p="You're a market analyst. Given sentiment scores array, top keywords list, and article snippets decide Uptrend/Downtrend/Neutral and return JSON {'trend':..., 'confidence':int,'explanation':...}. Sentiment:"+json.dumps(sent_scores)+". Keywords:"+json.dumps(keywords[:10])
    for i,a in enumerate(arts): p+=f"\nSnippet{i+1}:{a[:300].replace('\\n',' ')}"
    try:
        r=groq_client.chat.completions.create(model="llama-3.3-70b-versatile",messages=[{"role":"system","content":"You are concise."},{"role":"user","content":p}],max_tokens=300)
        txt=r.choices[0].message.content.strip()
        j=None
        try: j=json.loads(txt)
        except:
            m=re.search(r"\{.*\}",txt,re.DOTALL)
            if m:
                try: j=json.loads(m.group(0))
                except: j={"trend":"Unknown","confidence":0,"explain":txt}
        return {"trend":j.get("trend","Unknown"),"confidence":float(j.get("confidence",0)),"explain":j.get("explanation",txt)}
    except Exception as e:
        return {"trend":"Error","confidence":0,"explain":str(e)}
def risk_score(sent_scores,keyword_counts,headline_emotion=0.5):
    s=np.array(sent_scores); if_neg = np.mean(np.minimum(0,s)); pos=np.mean(np.maximum(0,s))
    intensity=(abs(pos)+abs(if_neg))
    kw_signal= (sum([c for_,c in keyword_counts[:5]])/50.0) if keyword_counts else 0
    score=(0.5*(1-abs(np.mean(s))) + 0.3*kw_signal + 0.2*headline_emotion)*100
    return round(float(np.clip(score,0,100)),2)
def tsi(sent_scores,sim):
    a=np.mean(sent_scores) if sent_scores else 0
    s= np.mean(sim) if sim is not None else 0.5
    return round(float(np.clip(a*0.7 + s*0.3, -1,1)),3)
def volatility(sent_scores):
    return float(np.std(sent_scores)) if sent_scores else 0.0
def radar_data(keywords):
    # simple buckets
    cats={"economy":0,"risk":0,"growth":0,"politics":0,"health":0}
    for w,c in keywords:
        s=w.lower()
        if any(x in s for x in ["econom","inflation","market","stock"]): cats["economy"]+=c
        if any(x in s for x in ["risk","uncertainty","crash","decline"]): cats["risk"]+=c
        if any(x in s for x in ["growth","rise","gain","profit"]): cats["growth"]+=c
        if any(x in s for x in ["polit","election","policy","gov"]): cats["politics"]+=c
        if any(x in s for x in ["health","covid","vaccine","disease"]): cats["health"]+=c
    vals=list(cats.values()); maxv=max(vals) if max(vals)>0 else 1
    return {k:round(v/maxv,3) for k,v in cats.items()}
# UI
if page=="Home":
    st.title("MarketTrend+ (input)")
    urls=[st.text_input(f"URL {i+1}",key=f"url{i}") for i in range(3)]
    source_type=st.selectbox("Source Type",["News Website","Blog","Twitter","YouTube","Other"])
    proc=st.button("Process Articles")
    if proc:
        with st.spinner("Scraping..."):
            arts,imgs=scrape(urls)
        if not any(arts): st.error("No articles"); st.stop()
        lang=[]
        for a in arts:
            try: lang.append(detect(a[:200])) 
            except: lang.append("en")
        embed_model=load_embed()
        emb=embed_model.encode(arts,convert_to_numpy=True)
        save_embeddings(emb,arts)
        fa_idx=build_faiss(emb) if HAVE_FAISS else None
        sent_pipe=load_sentiment()
        bert_scores,bert_labels=sentiment_bert(arts,sent_pipe)
        keywords=keyword_freq(arts,top=30)
        topics=topic_model(arts,n=3)
        sim=compute_similarity_matrix(emb)
        groq_res=trend_predict_groq(bert_scores,keywords,arts,groq_client)
        img_sent=[]
        img_text_cons=[]
        for a,img in zip(arts,imgs):
            img_sent.append(sentiment_image_with_groq(img,groq_client) if img else {"label":"NONE","score":0,"explain":"No image"})
            img_text_cons.append(image_text_consistency(a,img,groq_client) if img else {"match":"NoImage","confidence":0,"reason":"No image"})
        risk=risk_score(bert_scores,keywords,headline_emotion=0.5)
        heat = round((np.mean([abs(x) for x in bert_scores])*100 + risk)/2,2)
        tsi_v=tsi(bert_scores, np.mean(sim) if sim is not None else 0.5)
        vol=volatility(bert_scores)
        insights={"ts":datetime.utcnow().isoformat(),"arts":arts,"imgs":imgs,"lang":lang,"emb_shape":emb.shape,"bert_scores":bert_scores,"bert_labels":bert_labels,"keywords":keywords,"topics":topics,"sim":sim.tolist() if sim is not None else None,"groq":groq_res,"img_sent":img_sent,"img_text_cons":img_text_cons,"risk":risk,"heat":heat,"tsi":tsi_v,"vol":vol,"source_type":source_type}
        with open(INS_PATH,"w",encoding="utf-8") as f: json.dump(insights,f,indent=2)
        st.success("Processed and saved insights")
if page=="Graphs & Insights":
    st.title("Graphs & Insights")
    if not os.path.exists(INS_PATH): st.warning("Process articles first"); st.stop()
    with open(INS_PATH,"r",encoding="utf-8") as f:ins=json.load(f)
    arts=ins["arts"]; imgs=ins["imgs"]; bert_scores=ins["bert_scores"]; labels=ins["bert_labels"]; keywords=ins["keywords"]; topics=ins["topics"]; sim=np.array(ins["sim"]) if ins.get("sim") else None
    st.subheader("Sentiment Timeline (BERT)")
    x=np.arange(1,len(bert_scores)+1)
    fig,ax=plt.subplots(figsize=(7,3)); ax.plot(x,bert_scores,marker="o"); ax.axhline(0,linestyle="--",color="gray"); ax.set_xticks(x); ax.set_xticklabels([f"A{i}" for i in x]); ax.set_ylim(-1.05,1.05); ax.set_ylabel("Score"); st.pyplot(fig)
    st.subheader("Article Similarity Heatmap")
    if sim is not None:
        fig,ax=plt.subplots(figsize=(4,4)); im=ax.imshow(sim,cmap="viridis",vmin=0,vmax=1); ax.set_xticks(x-1); ax.set_yticks(x-1); ax.set_xticklabels([f"A{i}" for i in x]); ax.set_yticklabels([f"A{i}" for i in x]); plt.colorbar(im,ax=ax); st.pyplot(fig)
    else:
        st.info("FAISS/Similarity not available")
    st.subheader("Top Keywords")
    words=[w for w,_ in keywords[:20]]; counts=[c for _,c in keywords[:20]]
    fig,ax=plt.subplots(figsize=(8,3)); ax.barh(words[::-1],counts[::-1]); st.pyplot(fig)
    st.subheader("Word Cloud")
    wc=WordCloud(width=900,height=400,background_color="white").generate(" ".join(arts)); st.image(wc.to_array(),use_column_width=True)
    st.subheader("Topics (LDA)")
    for i,t in enumerate(topics): st.write(f"Topic {i+1}: {t}")
    st.subheader("Keyword Radar")
    rd=radar_data(keywords)
    labels=list(rd.keys()); values=list(rd.values()); angles=np.linspace(0,2*np.pi,len(labels),endpoint=False).tolist(); values+=values[:1]; angles+=angles[:1]
    fig=plt.figure(figsize=(4,4)); ax=fig.add_subplot(111,polar=True); ax.plot(angles,values,marker='o'); ax.fill(angles,values,alpha=0.25); ax.set_thetagrids(np.degrees(angles[:-1]),labels); st.pyplot(fig)
    st.subheader("Image-Text Consistency")
    for i,(imgc,im) in enumerate(zip(ins["img_text_cons"],imgs)):
        st.write(f"Article {i+1} consistency: {imgc.get('match')} (conf {imgc.get('confidence')}) Reason: {imgc.get('reason')}")
        if im: st.image(im,width=300)
    st.subheader("Image Sentiment (Groq)")
    for i,it in enumerate(ins["img_sent"]):
        st.write(f"Img {i+1}: {it.get('label')} conf {it.get('score')} explanation:{it.get('explain')[:120]}")
    st.subheader("Metrics")
    st.metric("Market Heat",ins["heat"]); st.metric("Risk Score",ins["risk"]); st.metric("TSI",ins["tsi"]); st.metric("Volatility",ins["vol"])
if page=="AI-Driven Analysis & Recommendations":
    st.title("AI Driven Analysis & Recommendations")
    if not os.path.exists(INS_PATH): st.warning("Process first"); st.stop()
    with open(INS_PATH,"r",encoding="utf-8") as f:ins=json.load(f)
    st.write("Groq Trend Prediction:"); st.write(ins["groq"])
    st.subheader("Hybrid Recommendation (BERT + Groq + Rule)")
    avg_sent=np.mean(ins["bert_scores"]) if ins["bert_scores"] else 0
    groq_trend=ins["groq"].get("trend","Unknown")
    final_score=0.6*(avg_sent+1)/2 + 0.3*(ins["risk"]/100) + 0.1*(ins["tsi"]+1)/2
    final_pct=round(final_score*100,2)
    if final_pct>60: rec="BUY"
    elif final_pct<40: rec="SELL"
    else: rec="HOLD"
    st.metric("Recommendation",rec); st.metric("Confidence %",final_pct)
    st.subheader("Explainability (Groq)")
    if groq_client:
        prompt="Explain in 4 bullet points why the predicted trend is correct given these signals. Sentiments:"+json.dumps(ins["bert_scores"])+" Top keywords:"+json.dumps(ins["keywords"][:10])+" Provide numbered bullets."
        try:
            r=groq_client.chat.completions.create(model="llama-3.3-70b-versatile",messages=[{"role":"system","content":"You are concise."},{"role":"user","content":prompt}],max_tokens=250)
            st.write(r.choices[0].message.content.strip())
        except Exception as e:
            st.write("Groq error:",e)
    else:
        st.write("Groq not configured - showing rule-based reasons:")
        reasons=[]
        reasons.append(f"Average sentiment {round(avg_sent,3)}")
        reasons.append(f"Top keyword {ins['keywords'][0][0] if ins['keywords'] else 'N/A'}")
        reasons.append(f"Risk score {ins['risk']}")
        reasons.append(f"Volatility {ins['vol']}")
        for r in reasons: st.write("-",r)
    st.subheader("Actionable Recommendations")
    if groq_client:
        p2="Produce 5 short actionable recommendations for an investor/manager given trend:"+json.dumps(ins["groq"])+" and signals. Numbered list."
        try:
            r=groq_client.chat.completions.create(model="llama-3.3-70b-versatile",messages=[{"role":"system","content":"You are practical."},{"role":"user","content":p2}],max_tokens=250)
            st.write(r.choices[0].message.content.strip())
        except Exception as e:
            st.write("Groq error",e)
    else:
        st.write("1. Monitor headlines. 2. Hedge if negative sentiment. 3. Gather more data.")
if page=="Research Paper":
    st.title("Research Paper Integration")
    st.write("Selected paper: Multimodal fake news detection (Yakkundi et al. 2025). Unique: multimodal lifecycle, datasets, real-time challenges.")
    summary=f"""Integration summary:
- Added: text-image consistency, multimodal sentiment (text+image), similarity heatmap, topic modeling, radar chart, risk/TSI/volatility, explainability via LLM, multilingual detection.
Generated:{datetime.utcnow().isoformat()}"""
    st.download_button("Download summary txt",summary,file_name="research_integration.txt")
