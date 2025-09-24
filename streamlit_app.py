# app/streamlit_app.py — FINAL
# Single‑post classifier with side‑by‑side: Data‑centric (TF‑IDF+LR) vs Model‑centric (DistilBERT)
# Shows probabilities, key‑phrase highlights, and LLM summary. Logs predictions.
#
# Run locally:
#   pip install streamlit scikit-learn pandas numpy transformers torch python-dotenv requests
#   streamlit run app/streamlit_app.py

import os
import time
import json
import pandas as pd
import streamlit as st
import requests
from dotenv import load_dotenv

from ml.predict import Predictors, render_html_highlight

load_dotenv()

st.set_page_config(page_title="Breakup Sentiment Explorer", page_icon="💔", layout="wide")
st.title("💔 Domain-Specific Sentiment — Side‑by‑Side Models")

# --- Sidebar controls ---
st.sidebar.header("Settings")
show_probs = st.sidebar.checkbox("Show probabilities", value=True)
show_highlights = st.sidebar.checkbox("Highlight key phrases", value=True)
use_llm_summary = st.sidebar.checkbox("LLM contextual summary", value=True)

llm_base = os.getenv("LLM_API_BASE", "").rstrip("/")
llm_model = os.getenv("LLM_MODEL", "local-llm")
llm_key = os.getenv("LLM_API_KEY", "")
log_path = os.getenv("LOG_PATH", "logs/predictions.csv")

@st.cache_resource
def get_predictors():
    return Predictors()

pred = get_predictors()

# --- LLM summary helper ---
def llm_summarize(text: str, label: str) -> str:
    if not (llm_base and llm_key):
        return "(LLM not configured — set LLM_API_BASE and LLM_API_KEY)"
    url = f"{llm_base}/chat/completions"
    headers = {"Authorization": f"Bearer {llm_key}", "Content-Type": "application/json"}
    prompt = (
        "You summarize Reddit break-up posts.\n"
        f"Predicted sentiment: {label}\n"
        "Explain briefly (2-3 sentences) why the text reflects this sentiment, pointing out key phrases.\n\n"
        f"Text:\n{text}"
    )
    body = {
        "model": llm_model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 180,
    }
    try:
        r = requests.post(url, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"(LLM error) {e}"

# --- UI Tabs ---
tab_single, tab_about = st.tabs(["Single Post", "About & Tips"])

with tab_single:
    sample = st.text_area("Paste a Reddit post:", height=180, placeholder="Type or paste the post here…")

    if st.button("Classify", type="primary"):
        if not sample.strip():
            st.warning("Please paste some text.")
        else:
            cols = st.columns(2)

            # Left: Data‑centric (TF‑IDF + LR)
            with cols[0]:
                st.subheader("Data‑centric — TF‑IDF + Logistic Regression")
                res_lr = pred.predict_lr(sample)
                if res_lr is None:
                    st.info("Model not available (provide ml/data_centric/tfidf_vectorizer.joblib and logreg_model.joblib)")
                else:
                    st.markdown(f"**Prediction:** {res_lr['label']}")
                    if show_probs:
                        st.bar_chart(pd.DataFrame.from_dict(res_lr["probs"], orient="index", columns=["prob"]))
                    if show_highlights:
                        spans = pred.highlight_lr(sample, top_k=8)
                        html = render_html_highlight(sample, spans)
                        st.markdown(html, unsafe_allow_html=True)

            # Right: Model‑centric (DistilBERT)
            with cols[1]:
                st.subheader("Model‑centric — DistilBERT (fine‑tuned)")
                res_hf = pred.predict_hf(sample)
                if res_hf is None:
                    st.info("Model not available (place your fine‑tuned model files in ml/model_centric/)")
                else:
                    st.markdown(f"**Prediction:** {res_hf['label']}")
                    if show_probs:
                        st.bar_chart(pd.DataFrame.from_dict(res_hf["probs"], orient="index", columns=["prob"]))
                    if show_highlights:
                        spans = pred.highlight_hf(sample, top_k=8)
                        html = render_html_highlight(sample, spans)
                        st.markdown(html, unsafe_allow_html=True)

            # Choose label for LLM summary (prefer model-centric if available)
            label_for_summary = None
            if res_hf and res_hf.get("label"):
                label_for_summary = res_hf["label"]
            elif res_lr and res_lr.get("label"):
                label_for_summary = res_lr["label"]

            if use_llm_summary and label_for_summary:
                with st.spinner("Generating contextual summary…"):
                    summary = llm_summarize(sample, label_for_summary)
                st.subheader("LLM Contextual Summary")
                st.write(summary)

            # Logging (for your report/demo)
            try:
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                import csv
                row = {
                    "ts": int(time.time()),
                    "text": sample,
                    "label_lr": res_lr.get("label") if res_lr else "",
                    "label_hf": res_hf.get("label") if res_hf else "",
                    "probs_lr": json.dumps(res_lr.get("probs")) if res_lr else "",
                    "probs_hf": json.dumps(res_hf.get("probs")) if res_hf else "",
                }
                exists = os.path.isfile(log_path)
                with open(log_path, "a", newline='', encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if not exists:
                        w.writeheader()
                    w.writerow(row)
                st.caption(f"Logged to {log_path}")
            except Exception as e:
                st.caption(f"(Log error) {e}")

with tab_about:
    st.markdown(
        """
        **How it works**
        - Left panel (**Data‑centric**) uses a classical TF‑IDF + Logistic Regression pipeline (your uploaded vectorizer + classifier).
        - Right panel (**Model‑centric**) uses your fine‑tuned DistilBERT (loaded from an HF‑style folder).
        - **Key‑phrase highlights**:
          - For TF‑IDF+LR, we show n‑grams with highest contribution toward the predicted class.
          - For DistilBERT, we use a simple token‑occlusion method (mask a token and see the probability drop).
        - **LLM summary** explains *why* the sentiment fits, referencing phrases/themes.
        """
    )
