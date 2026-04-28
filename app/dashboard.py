# ============================================================
# File: app/dashboard.py
# Purpose: Streamlit dashboard for Social Media Sentiment Analysis
# Run with:  streamlit run app/dashboard.py
# ============================================================

import os
import sys
import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

# Allow importing from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import full_preprocess

# ── Page config ───────────────────────────────────────────
st.set_page_config(
    page_title="Social Media Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Sora', sans-serif; }

    .main { background: #0f1117; }

    .metric-card {
        background: linear-gradient(135deg, #1a1d2e 0%, #16192a 100%);
        border: 1px solid #2d3148;
        border-radius: 16px;
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; margin: 4px 0; }
    .metric-label { font-size: 0.85rem; color: #8892b0; letter-spacing: 1px; text-transform: uppercase; }

    .pos-card  { border-top: 3px solid #4CAF50; }
    .neg-card  { border-top: 3px solid #F44336; }
    .neu-card  { border-top: 3px solid #2196F3; }
    .tot-card  { border-top: 3px solid #9C27B0; }

    .pos-val   { color: #4CAF50; }
    .neg-val   { color: #F44336; }
    .neu-val   { color: #2196F3; }
    .tot-val   { color: #BB86FC; }

    .sentiment-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        letter-spacing: 0.5px;
    }
    .badge-positive { background: #1b3a1b; color: #4CAF50; }
    .badge-negative { background: #3a1b1b; color: #F44336; }
    .badge-neutral  { background: #1b2a3a; color: #2196F3; }

    .section-title {
        font-size: 1.2rem; font-weight: 600;
        color: #e0e0ff; margin-bottom: 12px;
        padding-bottom: 6px;
        border-bottom: 1px solid #2d3148;
    }

    .stTextArea textarea {
        background: #1a1d2e !important;
        color: #e0e0ff !important;
        border: 1px solid #2d3148 !important;
        border-radius: 10px !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white; border: none; border-radius: 10px;
        padding: 0.6rem 2rem; font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover { opacity: 0.85; transform: translateY(-1px); }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model_path = os.path.join("models", "sentiment_model.pkl")
    vec_path   = os.path.join("models", "tfidf_vectorizer.pkl")
    enc_path   = os.path.join("models", "label_encoder.pkl")

    if not all(os.path.exists(p) for p in [model_path, vec_path, enc_path]):
        return None, None, None

    with open(model_path, "rb") as f: model = pickle.load(f)
    with open(vec_path,   "rb") as f: vec   = pickle.load(f)
    with open(enc_path,   "rb") as f: le    = pickle.load(f)
    return model, vec, le

@st.cache_data
def load_dataset():
    path = os.path.join("data", "processed_posts.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

# ── Prediction helper ─────────────────────────────────────
def predict(text, model, vec, le):
    cleaned  = full_preprocess(text)
    X        = vec.transform([cleaned])
    label_id = model.predict(X)[0]
    proba    = model.predict_proba(X)[0]
    sentiment = le.inverse_transform([label_id])[0]
    prob_dict = {le.classes_[i]: round(float(p)*100, 1) for i, p in enumerate(proba)}
    return sentiment, prob_dict

# ── Color helpers ─────────────────────────────────────────
SENT_COLOR = {"positive": "#4CAF50", "negative": "#F44336", "neutral": "#2196F3"}
SENT_EMOJI = {"positive": "😊", "negative": "😠", "neutral": "😐"}

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 Dashboard Navigation")
    page = st.radio(
        "Select View",
        ["🏠 Overview", "🔍 Live Predictor", "📈 Analytics", "🏷️ Brand Tracker", "💡 Insights"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ⚙️ Filters")
    model, vec, le = load_artifacts()
    df = load_dataset()

    platform_filter = ["All"]
    brand_filter    = ["All"]
    if df is not None:
        platform_filter += sorted(df["platform"].unique().tolist())
        brand_filter    += sorted(df["brand"].unique().tolist())

    sel_platform = st.selectbox("Platform", platform_filter)
    sel_brand    = st.selectbox("Brand",    brand_filter)
    st.markdown("---")
    st.markdown("**Model Status**")
    if model:
        st.success("✅ Model loaded")
    else:
        st.error("❌ Model not found\nRun `python main.py` first")

# Apply filters
if df is not None:
    fdf = df.copy()
    if sel_platform != "All": fdf = fdf[fdf["platform"] == sel_platform]
    if sel_brand    != "All": fdf = fdf[fdf["brand"]    == sel_brand]
else:
    fdf = None

# ═══════════════════════════════════════════════════════════
#  PAGE 1 — OVERVIEW
# ═══════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown("# 📊 Social Media Sentiment Dashboard")
    st.markdown("*Real-time brand sentiment monitoring powered by ML*")

    if fdf is None or model is None:
        st.warning("⚠️ Run `python main.py` to generate data and train the model first.")
        st.stop()

    counts = fdf["sentiment"].value_counts()
    total  = len(fdf)
    pos    = counts.get("positive", 0)
    neg    = counts.get("negative", 0)
    neu    = counts.get("neutral",  0)

    # ── KPI cards ─────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"""<div class="metric-card tot-card">
        <div class="metric-label">Total Posts</div>
        <div class="metric-value tot-val">{total:,}</div>
    </div>""", unsafe_allow_html=True)
    c2.markdown(f"""<div class="metric-card pos-card">
        <div class="metric-label">Positive</div>
        <div class="metric-value pos-val">{pos:,}</div>
        <div class="metric-label">{pos/total*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
    c3.markdown(f"""<div class="metric-card neg-card">
        <div class="metric-label">Negative</div>
        <div class="metric-value neg-val">{neg:,}</div>
        <div class="metric-label">{neg/total*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)
    c4.markdown(f"""<div class="metric-card neu-card">
        <div class="metric-label">Neutral</div>
        <div class="metric-value neu-val">{neu:,}</div>
        <div class="metric-label">{neu/total*100:.1f}%</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts row 1 ──────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-title">Sentiment Distribution</div>', unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["Positive", "Negative", "Neutral"],
            values=[pos, neg, neu],
            hole=0.45,
            marker_colors=["#4CAF50", "#F44336", "#2196F3"],
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} posts<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0ff"), showlegend=False, height=320,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-title">Sentiment by Platform</div>', unsafe_allow_html=True)
        plat_sent = fdf.groupby(["platform", "sentiment"]).size().reset_index(name="count")
        fig2 = px.bar(
            plat_sent, x="platform", y="count", color="sentiment",
            color_discrete_map=SENT_COLOR, barmode="group",
            template="plotly_dark",
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0ff"), height=320,
            margin=dict(l=10, r=10, t=10, b=10),
            legend_title_text="Sentiment",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Charts row 2 ──────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown('<div class="section-title">Avg Likes by Sentiment</div>', unsafe_allow_html=True)
        likes_avg = fdf.groupby("sentiment")["likes"].mean().reset_index()
        fig3 = px.bar(
            likes_avg, x="sentiment", y="likes",
            color="sentiment", color_discrete_map=SENT_COLOR,
            template="plotly_dark", text_auto=".0f",
        )
        fig3.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0ff"), height=300,
            showlegend=False, margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown('<div class="section-title">Recent Posts Sample</div>', unsafe_allow_html=True)
        sample = fdf[["text", "sentiment", "brand", "platform"]].tail(8).copy()
        sample["sentiment"] = sample["sentiment"].apply(
            lambda s: f'<span class="sentiment-badge badge-{s}">{SENT_EMOJI[s]} {s}</span>'
        )
        st.write(sample.to_html(escape=False, index=False), unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  PAGE 2 — LIVE PREDICTOR
# ═══════════════════════════════════════════════════════════
elif page == "🔍 Live Predictor":
    st.markdown("# 🔍 Live Sentiment Predictor")
    st.markdown("*Type or paste any social media post to classify its sentiment instantly.*")

    if model is None:
        st.warning("⚠️ Model not found. Run `python main.py` first.")
        st.stop()

    text_input = st.text_area(
        "Enter post / tweet / review / comment:",
        placeholder="e.g.  Amazing delivery! Got my order in 2 hours. Highly recommend!",
        height=130,
    )

    col1, col2, col3 = st.columns([1, 1, 3])
    predict_btn = col1.button("🚀 Analyse")
    clear_btn   = col2.button("🗑️ Clear")

    if predict_btn and text_input.strip():
        with st.spinner("Analysing …"):
            sentiment, probs = predict(text_input, model, vec, le)

        color = SENT_COLOR[sentiment]
        emoji = SENT_EMOJI[sentiment]

        st.markdown(f"""
        <div style="background:linear-gradient(135deg,#1a1d2e,#16192a);
                    border:1px solid {color};border-radius:16px;
                    padding:24px 30px;margin:16px 0;">
            <div style="font-size:2rem;font-weight:700;color:{color};">
                {emoji} {sentiment.upper()}
            </div>
            <div style="color:#8892b0;font-size:0.9rem;margin-top:4px;">
                Confidence: <strong style="color:{color};">{probs[sentiment]}%</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Probability gauge
        st.markdown("#### Probability Breakdown")
        fig = go.Figure()
        for sent_label, prob_val in sorted(probs.items(), key=lambda x: -x[1]):
            fig.add_trace(go.Bar(
                x=[prob_val], y=[sent_label], orientation="h",
                marker_color=SENT_COLOR[sent_label],
                text=f"{prob_val}%", textposition="inside",
                name=sent_label,
            ))
        fig.update_layout(
            xaxis=dict(range=[0, 100], title="Probability (%)"),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#e0e0ff"), height=200,
            showlegend=False, margin=dict(l=10, r=10, t=10, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show cleaned text
        with st.expander("🔬 See how the text was processed"):
            cleaned = full_preprocess(text_input)
            st.markdown(f"**Original:** `{text_input}`")
            st.markdown(f"**Cleaned:** `{cleaned}`")

    # ── Batch prediction ───────────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Batch Prediction")
    st.markdown("Enter one post per line and classify all at once.")
    batch_input = st.text_area("Batch input:", height=150,
        placeholder="Post 1\nPost 2\nPost 3")

    if st.button("🚀 Analyse All") and batch_input.strip():
        lines = [l.strip() for l in batch_input.split("\n") if l.strip()]
        results = []
        for line in lines:
            s, p = predict(line, model, vec, le)
            results.append({"Text": line[:80], "Sentiment": s,
                             "Confidence": f"{p[s]}%"})
        rdf = pd.DataFrame(results)
        st.dataframe(rdf, use_container_width=True)

        counts = rdf["Sentiment"].value_counts()
        fig = px.pie(values=counts.values, names=counts.index,
                     color=counts.index, color_discrete_map=SENT_COLOR,
                     template="plotly_dark")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                          font=dict(color="#e0e0ff"), height=280)
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════
#  PAGE 3 — ANALYTICS
# ═══════════════════════════════════════════════════════════
elif page == "📈 Analytics":
    st.markdown("# 📈 Deep Analytics")

    if fdf is None:
        st.warning("⚠️ Run `python main.py` first.")
        st.stop()

    # ── Sentiment trend (simulated time series) ────────────
    st.markdown("### Sentiment Trend Over Time (Simulated)")
    fdf2 = fdf.copy()
    fdf2["day"] = pd.to_datetime("2024-01-01") + pd.to_timedelta(
        np.random.randint(0, 90, size=len(fdf2)), unit="D"
    )
    trend = fdf2.groupby(["day", "sentiment"]).size().reset_index(name="count")

    fig = px.line(trend, x="day", y="count", color="sentiment",
                  color_discrete_map=SENT_COLOR, template="plotly_dark",
                  markers=True)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e0e0ff"), height=360,
                      xaxis_title="Date", yaxis_title="Post Count")
    st.plotly_chart(fig, use_container_width=True)

    # ── Engagement analysis ────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Likes Distribution by Sentiment")
        fig2 = px.box(fdf, x="sentiment", y="likes", color="sentiment",
                      color_discrete_map=SENT_COLOR, template="plotly_dark")
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#e0e0ff"), height=320,
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### Retweets Distribution by Sentiment")
        fig3 = px.violin(fdf, x="sentiment", y="retweets", color="sentiment",
                         color_discrete_map=SENT_COLOR, template="plotly_dark",
                         box=True)
        fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                           font=dict(color="#e0e0ff"), height=320,
                           showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # ── Word cloud ─────────────────────────────────────────
    st.markdown("### Word Clouds by Sentiment")
    wc_cols = st.columns(3)
    for idx, sent in enumerate(["positive", "negative", "neutral"]):
        texts = " ".join(fdf[fdf["sentiment"] == sent]["processed_text"].dropna().tolist())
        if texts.strip():
            wc = WordCloud(
                width=600, height=300,
                background_color="black",
                colormap="Greens" if sent=="positive"
                         else "Reds" if sent=="negative" else "Blues",
                max_words=80,
            ).generate(texts)
            fig_wc, ax = plt.subplots(figsize=(5, 2.5))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            ax.set_title(sent.capitalize(), color="white", fontsize=12)
            fig_wc.patch.set_facecolor("#0f1117")
            wc_cols[idx].pyplot(fig_wc)
            plt.close(fig_wc)

# ═══════════════════════════════════════════════════════════
#  PAGE 4 — BRAND TRACKER
# ═══════════════════════════════════════════════════════════
elif page == "🏷️ Brand Tracker":
    st.markdown("# 🏷️ Brand-wise Sentiment Tracker")

    if fdf is None:
        st.warning("⚠️ Run `python main.py` first.")
        st.stop()

    brand_sent = fdf.groupby(["brand", "sentiment"]).size().reset_index(name="count")
    brand_total = fdf.groupby("brand").size().reset_index(name="total")
    brand_pos   = fdf[fdf["sentiment"]=="positive"].groupby("brand").size().reset_index(name="pos")
    brand_summary = brand_total.merge(brand_pos, on="brand", how="left").fillna(0)
    brand_summary["pos_pct"] = (brand_summary["pos"] / brand_summary["total"] * 100).round(1)
    brand_summary = brand_summary.sort_values("pos_pct", ascending=False)

    # ── Brand sentiment heatmap ────────────────────────────
    st.markdown("### Brand Positivity Score")
    fig = px.bar(
        brand_summary, x="brand", y="pos_pct",
        color="pos_pct",
        color_continuous_scale=["#F44336", "#FF9800", "#4CAF50"],
        template="plotly_dark", text_auto=".1f",
    )
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#e0e0ff"), height=380,
                      xaxis_title="Brand", yaxis_title="Positive %",
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Stacked bar ────────────────────────────────────────
    st.markdown("### Post Volume by Brand & Sentiment")
    fig2 = px.bar(brand_sent, x="brand", y="count", color="sentiment",
                  color_discrete_map=SENT_COLOR, barmode="stack",
                  template="plotly_dark")
    fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                       font=dict(color="#e0e0ff"), height=380)
    st.plotly_chart(fig2, use_container_width=True)

    # ── Table ──────────────────────────────────────────────
    st.markdown("### Brand Rankings")
    st.dataframe(
        brand_summary[["brand","total","pos","pos_pct"]]
            .rename(columns={"brand":"Brand","total":"Total Posts",
                              "pos":"Positive Posts","pos_pct":"Positivity %"}),
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════
#  PAGE 5 — INSIGHTS
# ═══════════════════════════════════════════════════════════
elif page == "💡 Insights":
    st.markdown("# 💡 Business Insights & Recommendations")

    if fdf is None:
        st.warning("⚠️ Run `python main.py` first.")
        st.stop()

    total = len(fdf)
    pos_pct = len(fdf[fdf["sentiment"]=="positive"]) / total * 100
    neg_pct = len(fdf[fdf["sentiment"]=="negative"]) / total * 100
    neu_pct = len(fdf[fdf["sentiment"]=="neutral"])  / total * 100

    top_platform = fdf[fdf["sentiment"]=="negative"]["platform"].value_counts().idxmax()
    top_brand    = fdf[fdf["sentiment"]=="positive"]["brand"].value_counts().idxmax()
    worst_brand  = fdf[fdf["sentiment"]=="negative"]["brand"].value_counts().idxmax()

    insights = [
        ("🟢", "Brand Health Score", f"Overall positivity is {pos_pct:.1f}%. "
         f"{'Healthy brand perception.' if pos_pct>40 else 'Needs improvement.'}"),
        ("🔴", "Crisis Alert", f"Negative sentiment is {neg_pct:.1f}%. "
         f"Platform most affected: **{top_platform}**. Immediate response recommended."),
        ("🏆", "Top Performer", f"**{top_brand}** has the highest positive mention volume. "
         "Leverage this brand for cross-promotional campaigns."),
        ("⚠️", "Brand Needing Attention", f"**{worst_brand}** receives the most negative mentions. "
         "Investigate root causes and improve service touchpoints."),
        ("💡", "Content Strategy", f"Neutral posts represent {neu_pct:.1f}% of mentions. "
         "Engage this audience with targeted campaigns to shift them positive."),
        ("📊", "Engagement Insight", "Positive posts receive significantly more likes and retweets. "
         "Focus on positive storytelling and user success stories."),
    ]

    for icon, title, desc in insights:
        st.markdown(f"""
        <div style="background:#1a1d2e;border-radius:12px;
                    padding:18px 22px;margin-bottom:14px;
                    border-left:4px solid #667eea;">
            <div style="font-size:1rem;font-weight:600;color:#e0e0ff;margin-bottom:6px;">
                {icon} {title}
            </div>
            <div style="color:#8892b0;font-size:0.92rem;line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Quick stats ────────────────────────────────────────
    st.markdown("### 📌 Quick Stats")
    q1, q2, q3 = st.columns(3)
    q1.metric("Avg Likes (Positive)", f"{fdf[fdf['sentiment']=='positive']['likes'].mean():.0f}")
    q2.metric("Avg Likes (Negative)", f"{fdf[fdf['sentiment']=='negative']['likes'].mean():.0f}")
    q3.metric("Platforms Tracked", fdf["platform"].nunique())

# ── Footer ────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center;color:#555;font-size:0.8rem;'>"
    "Social Media Sentiment Dashboard | Built with Python · Scikit-learn · Streamlit · Plotly"
    "</div>",
    unsafe_allow_html=True,
)