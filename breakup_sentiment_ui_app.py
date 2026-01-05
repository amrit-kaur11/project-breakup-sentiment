# app.py
# Streamlit UI for domain-specific sentiment analysis with two strategies:
# - Data-centric: TF-IDF + Logistic Regression (from your tfidf_output.zip)
# - Model-centric: DistilBERT fine-tune (from your saved HF artifacts)
# Also includes basic highlights and an optional LLM "why" explanation via Ollama or OpenAI-compatible APIs.

import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

import requests

# -----------------------------
# ======== CONFIG AREA ========
# Edit these paths to where you saved the artifacts locally.
# Example layout:
#   complete_project/
#     app.py
#     artifacts/
#       tfidf/
#         tfidf_vectorizer.joblib
#         logreg_model.joblib
#         metrics.json (optional)
#       distilbert/
#         config.json
#         model.safetensors
#         tokenizer.json
#         tokenizer_config.json
#         special_tokens_map.json
#         vocab.txt
#         training_args.bin (optional)
# -----------------------------

LABELS = ["anger", "confusion", "relief", "sadness"]
# Default locations (can be overridden in the sidebar or via env vars)
# Hard-coded defaults to your Google Drive paths (you can still change them in the sidebar)
DEFAULT_TFIDF_DIR = os.getenv("TFIDF_DIR", r"G:\My Drive\Domain-Specific Sentiment Analysis with Low-Resource NLP Techniques -- Amrit Kaur\complete_project\artifacts\tfidf_output")
DEFAULT_BERT_DIR  = os.getenv("BERT_DIR",  r"G:\My Drive\Domain-Specific Sentiment Analysis with Low-Resource NLP Techniques -- Amrit Kaur\complete_project\artifacts\distilbert_out_chunked")
DEFAULT_DEMO_CSV  = os.getenv("AUGMENTED_CSV",  r"G:\My Drive\Domain-Specific Sentiment Analysis with Low-Resource NLP Techniques -- Amrit Kaur\complete_project\data\augmented.csv")


# LLM providers (choose one):
# 1) Ollama (local). Install: https://ollama.com (then pull a model, e.g., `ollama pull llama3.1`)
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
# 2) OpenAI-compatible endpoint (optional): set OPENAI_API_KEY and optionally OPENAI_BASE_URL
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Maximum tokens per BERT window. You can raise to 384 or 512 if GPU memory allows.
BERT_MAX_LENGTH = int(os.getenv("BERT_MAX_LENGTH", "256"))  # try 384 or 512 if you can
BERT_STRIDE = int(os.getenv("BERT_STRIDE", "64"))

# -----------------------------
# ======== UTILITIES ==========
# -----------------------------

def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


def colorize_tokens(tokens_with_scores: List[Tuple[str, float]], max_tokens: int = 25) -> str:
    """Return HTML string with tokens highlighted from light to strong based on score.
    tokens_with_scores: list of (token_str, importance_score)
    """
    # Normalize scores to [0,1]
    scores = np.array([s for _, s in tokens_with_scores])
    if len(scores) == 0:
        return ""
    if np.ptp(scores) == 0:
        norm = np.zeros_like(scores)
    else:
        norm = (scores - scores.min()) / (scores.max() - scores.min())

    # Cap number of tokens displayed
    tokens_with_scores = tokens_with_scores[:max_tokens]
    html_parts = []
    for (tok, _), alpha in zip(tokens_with_scores, norm[:max_tokens]):
        # pastel red for emphasis; adjust alpha (0.1..1)
        alpha = 0.15 + 0.85 * float(alpha)
        span = f'<span style="background-color: rgba(255, 0, 0, {alpha:.2f}); padding: 2px; margin: 1px; border-radius: 4px; display: inline-block;">{tok}</span>'
        html_parts.append(span)
    return " ".join(html_parts)


# -----------------------------
# ====== LOAD ARTIFACTS =======
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_tfidf_pipeline(vec_path: Path, clf_path: Path):
    if not vec_path.exists() or not clf_path.exists():
        return None, None
    vectorizer = joblib.load(vec_path)
    clf = joblib.load(clf_path)
    return vectorizer, clf


@st.cache_resource(show_spinner=False)
def load_bert(bert_dir: Path):
    # Handle common layout:
    #  - Parent folder contains two subfolders: 'best' (model) and 'best_tok' (tokenizer)
    #  - Or everything is in a single folder
    if bert_dir is None or not bert_dir.exists():
        return None, None

    model_dir = bert_dir
    tok_dir = bert_dir

    # If exported as separate subfolders, prefer them
    if (bert_dir / "best").exists():
        model_dir = bert_dir / "best"
    if (bert_dir / "best_tok").exists():
        tok_dir = bert_dir / "best_tok"

    # Sanity checks for required files
    has_model_files = (model_dir / "config.json").exists() and (
        (model_dir / "pytorch_model.bin").exists() or (model_dir / "model.safetensors").exists()
    )
    has_tokenizer_fast = (tok_dir / "tokenizer.json").exists()
    has_tokenizer_slow = (tok_dir / "vocab.txt").exists()

    if not has_model_files:
        st.error(f"DistilBERT model files not found in: {model_dir}")
        return None, None
    if not (has_tokenizer_fast or has_tokenizer_slow):
        st.error(
            f"Tokenizer files not found in: {tok_dir}. Need 'tokenizer.json' (fast) or 'vocab.txt' (slow)."
        )
        return None, None

    try:
        tok = AutoTokenizer.from_pretrained(str(tok_dir), use_fast=has_tokenizer_fast)
        model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))
        model.eval()
        return tok, model
    except Exception as e:
        st.error(f"Failed to load DistilBERT (model from {model_dir}, tokenizer from {tok_dir}): {e}")
        return None, None


# -----------------------------
# ===== PREDICT: TF-IDF =======
# -----------------------------

def predict_tfidf(text: str, vectorizer, clf) -> Tuple[str, Dict[str, float], List[Tuple[str, float]]]:
    X = vectorizer.transform([text])
    probs = clf.predict_proba(X)[0]
    pred_idx = int(np.argmax(probs))
    pred_label = LABELS[pred_idx]
    prob_map = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

    # Simple explanation: token contributions for predicted class
    # contribution = feature_value * coef[class, feature]
    if hasattr(clf, "coef_"):
        feats = vectorizer.get_feature_names_out()
        coef = clf.coef_[pred_idx]
        x = X.toarray()[0]
        # Only non-zero features contribute
        nz_idx = np.where(x != 0)[0]
        pairs = []
        for j in nz_idx:
            contribution = x[j] * coef[j]
            if contribution > 0:
                token = feats[j]
                pairs.append((token, float(contribution)))
        # sort by contribution descending
        pairs.sort(key=lambda t: t[1], reverse=True)
    else:
        pairs = []

    return pred_label, prob_map, pairs


# -----------------------------
# ===== PREDICT: BERT  =========
# -----------------------------

def _aggregate_logits(logits_list: List[torch.Tensor]) -> torch.Tensor:
    # mean aggregation across overflow windows
    stacked = torch.stack(logits_list, dim=0)
    return torch.mean(stacked, dim=0)


def predict_bert(text: str, tokenizer, model) -> Tuple[str, Dict[str, float], List[Tuple[str, float]]]:
    # Tokenize with overflow, then aggregate predictions across windows
    encoded = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=BERT_MAX_LENGTH,
        stride=BERT_STRIDE,
        return_overflowing_tokens=True,
    )

    logits_list = []
    token_attn_pairs: List[Tuple[str, float]] = []

    with torch.no_grad():
        for i in range(encoded["input_ids"].shape[0]):
            inputs = {k: v[i].unsqueeze(0) for k, v in encoded.items() if k in ["input_ids", "attention_mask"]}
            outputs = model(**inputs, output_attentions=True)
            logits = outputs.logits.squeeze(0)
            logits_list.append(logits)

            # Basic attention-based token importance (last layer, averaged heads)
            if outputs.attentions is not None:
                last = outputs.attentions[-1]  # [batch, heads, seq, seq]
                attn = last.mean(dim=1).squeeze(0)  # [seq, seq]
                # importance of each token = attention paid to [CLS] (token 0) by others, or sum across rows
                cls_importance = attn[0]  # how much other tokens attend to CLS
                # Map back to tokens (skip special tokens for display)
                tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze(0))
                for tok, score in zip(tokens, cls_importance.tolist()):
                    if tok not in ("[CLS]", "[SEP]", "[PAD]"):
                        token_attn_pairs.append((tok.replace("##", ""), float(score)))

    agg_logits = _aggregate_logits(logits_list)
    probs = F.softmax(agg_logits, dim=-1).detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    pred_label = LABELS[pred_idx]
    prob_map = {LABELS[i]: float(probs[i]) for i in range(len(LABELS))}

    # Sort attention tokens by score desc
    token_attn_pairs.sort(key=lambda t: t[1], reverse=True)

    return pred_label, prob_map, token_attn_pairs


# -----------------------------
# ======== LLM EXPLAIN =========
# -----------------------------
LLM_SYSTEM = (
    "You are an assistant that explains why a Reddit breakup post expresses a certain emotion. "
    "Keep it 3-5 sentences. Reference specific phrases. Don’t repeat the post."
)

LLM_USER_TEMPLATE = (
    "Post (shortened):\n{snippet}\n\n"
    "Model prediction: {label}\n\n"
    "Explain in simple language why this text likely expresses **{label}**. "
    "Mention up to 3 phrases that hint at this emotion."
)


def llm_explain(snippet: str, label: str) -> str:
    # Prefer Ollama if running locally; otherwise try OpenAI-compatible
    if OPENAI_API_KEY:
        try:
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY:}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": OPENAI_MODEL,
                "messages": [
                    {"role": "system", "content": LLM_SYSTEM},
                    {"role": "user", "content": LLM_USER_TEMPLATE.format(snippet=snippet, label=label)},
                ],
                "temperature": 0.2,
            }
            r = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=60)
            r.raise_for_status()
            data = r.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"(LLM error via OpenAI-compatible API: {e})"

    # Ollama fallback
    try:
        payload = {
            "model": OLLAMA_MODEL,
            "prompt": LLM_SYSTEM + "\n\n" + LLM_USER_TEMPLATE.format(snippet=snippet, label=label),
            "stream": False,
            "options": {"temperature": 0.2},
        }
        r = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=60)
        r.raise_for_status()
        data = r.json()
        return data.get("response", "(No LLM response)").strip()
    except Exception as e:
        return f"(LLM error via Ollama: {e})"


# -----------------------------
# ========== UI ===============
# -----------------------------
st.set_page_config(page_title="Breakup Sentiment Classifier", layout="wide")
st.title("💔 Breakup Sentiment Classifier — Two Strategies + LLM Explanations")

with st.sidebar:
    st.header("Settings")
    st.markdown("**Artifacts folders**")
    tfidf_path_str = st.text_input("TF-IDF folder", value=DEFAULT_TFIDF_DIR)
    bert_path_str = st.text_input("DistilBERT folder", value=DEFAULT_BERT_DIR)
    demo_csv_path_str = st.text_input("demo.csv path (optional)", value=DEFAULT_DEMO_CSV)

    # Resolve user-provided paths
    TFIDF_DIR = Path(tfidf_path_str)
    BERT_DIR = Path(bert_path_str)
    DEMO_CSV = Path(demo_csv_path_str)

    model_choice = st.radio(
        "Choose a model:",
        ["Data-centric: TF-IDF + Logistic Regression", "Model-centric: DistilBERT"],
        index=0,
    )

    st.markdown("**BERT window size** (advanced)")
    st.caption("Increase if you have enough memory. Then click Predict again.")
    st.write(f"max_length = {BERT_MAX_LENGTH}, stride = {BERT_STRIDE}")

    st.markdown("**LLM provider**")
    st.caption("Set OPENAI_API_KEY (and optional OPENAI_BASE_URL) for OpenAI-compatible; otherwise uses Ollama.")

# Load models
vec_fp = TFIDF_DIR / "tfidf_vectorizer.joblib"
clf_fp = TFIDF_DIR / "logreg_model.joblib"
vectorizer, logreg = load_tfidf_pipeline(vec_fp, clf_fp)

tokenizer, bert_model = load_bert(BERT_DIR)

col1, col2 = st.columns([3, 2], gap="large")

with col1:
    st.subheader("Try it out ✍️")
    text = st.text_area(
        "Paste a breakup-related Reddit post:",
        height=200,
        placeholder="We were together for three years...",
    )
    run_demo = st.checkbox("Use demo.csv (batch)")

    if st.button("Predict", type="primary"):
        if run_demo and DEMO_CSV.exists():
            df = pd.read_csv(DEMO_CSV)
            if "text" not in df.columns:
                st.error("demo.csv must have a 'text' column")
            else:
                rows = []
                for row in df.to_dict(orient="records"):
                    t = str(row["text"])[:4000]
                    if model_choice.startswith("Data-centric"):
                        if vectorizer is None or logreg is None:
                            st.error("TF-IDF artifacts not found.")
                            break
                        label, probs, pairs = predict_tfidf(t, vectorizer, logreg)
                    else:
                        if tokenizer is None or bert_model is None:
                            st.error("DistilBERT artifacts not found.")
                            break
                        label, probs, pairs = predict_bert(t, tokenizer, bert_model)
                    rows.append({
                        "text": t,
                        "prediction": label,
                        **{f"prob_{k}": v for k, v in probs.items()}
                    })
                if rows:
                    out = pd.DataFrame(rows)
                    st.dataframe(out)
                    # Offer CSV download
                    csv = out.to_csv(index=False).encode("utf-8")
                    st.download_button("Download predictions CSV", csv, "predictions.csv", "text/csv")
        else:
            if not text.strip():
                st.warning("Please paste some text first.")
            else:
                t = text.strip()[:8000]
                if model_choice.startswith("Data-centric"):
                    if vectorizer is None or logreg is None:
                        st.error("TF-IDF artifacts not found.")
                    else:
                        label, probs, pairs = predict_tfidf(t, vectorizer, logreg)
                else:
                    if tokenizer is None or bert_model is None:
                        st.error("DistilBERT artifacts not found.")
                    else:
                        label, probs, pairs = predict_bert(t, tokenizer, bert_model)

                if 'label' in locals():
                    st.success(f"Prediction: **{label}**")
                    # Probabilities bar
                    chart_df = pd.DataFrame({"label": list(probs.keys()), "probability": list(probs.values())})
                    st.bar_chart(chart_df.set_index("label"))

                    # Token highlights
                    if pairs:
                        st.markdown("**Highlighted clues (top tokens):**")
                        html = colorize_tokens(pairs, max_tokens=40)
                        st.markdown(html, unsafe_allow_html=True)

                    # LLM explanation
                    st.markdown("---")
                    st.markdown("**LLM Summary:**")
                    # Make a short snippet to avoid sending too-long text
                    snippet = (t[:800] + ("..." if len(t) > 800 else "")).replace("\n", " ")
                    explanation = llm_explain(snippet, label)
                    st.write(explanation)

with col2:
    st.subheader("How it works 🧠")
    st.markdown(
        """
        - **Two strategies**:
          - **Data-centric** (TF-IDF + Logistic Regression): counts important words/phrases and learns weights for each emotion.
          - **Model-centric** (DistilBERT): a transformer reads the text and predicts the emotion using context.
        - **Bars** show confidence for each emotion.
        - **Highlights** show words that most likely pushed the model to its decision.
        - **LLM** explains the decision in simple words.
        """
    )

    st.subheader("Tips")
    st.markdown(
        """
        - If your posts are long, try a bigger BERT window (set `BERT_MAX_LENGTH=384` or `512`).
        - For the **data-centric** model, export your own `tfidf_vectorizer.joblib` and `logreg_model.joblib` into `artifacts/tfidf/`.
        - For the **model-centric** model, keep all DistilBERT files in `artifacts/distilbert/` and load via `from_pretrained`.
        - To use the **LLM**:
          1. **OpenAI-compatible**: set `OPENAI_API_KEY` (and optional `OPENAI_BASE_URL`).
          2. **Ollama**: install Ollama, run locally, and (optionally) `export OLLAMA_MODEL=llama3.1`.
        """
    )

# End of app.py
