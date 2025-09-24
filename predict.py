# ml/predict.py — FINAL
# Loads your data‑centric (TF‑IDF + LR) and model‑centric (DistilBERT) models
# with smart fallbacks to your uploaded /mnt/data artifacts.
# Also provides key‑phrase highlighting and HTML marker rendering.

from __future__ import annotations
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

# Optional imports (only used if LR/HF models are available)
try:
    import joblib
    from sklearn.pipeline import Pipeline
except Exception:
    joblib = None
    Pipeline = None

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except Exception:
    torch = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None

# ----------------------
# Path preferences
# ----------------------
PREFERRED = {
    "labels": "ml/labels.json",
    "vec": "ml/data_centric/tfidf_vectorizer.joblib",
    "lr": "ml/data_centric/logreg_model.joblib",
    "bert": "ml/model_centric",  # folder with config.json, tokenizer*, vocab.txt, model weights
}
UPLOADED = {
    "labels": "/mnt/data/labels.json",
    "vec": "/mnt/data/tfidf_vectorizer.joblib",
    "lr": "/mnt/data/logreg_model.joblib",
    "bert": "/mnt/data",  # you uploaded config/tokenizer/vocab/weights here
}

DEFAULT_LABELS = ["anger", "confusion", "relief", "sadness"]


def _first_existing(*paths: str) -> str:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return ""


def _load_labels() -> List[str]:
    path = _first_existing(PREFERRED["labels"], UPLOADED["labels"])
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return DEFAULT_LABELS


@dataclass
class Predictors:
    labels_path: str = PREFERRED["labels"]
    hf_dir: str = PREFERRED["bert"]
    lr_vec_path: str = PREFERRED["vec"]
    lr_path: str = PREFERRED["lr"]

    def __post_init__(self):
        # Labels
        self.labels: List[str] = _load_labels()

        # Data‑centric (TF‑IDF + LR)
        self.lr_pipe = None
        self.lr_classes: List[str] | None = None
        self.has_lr = False
        if joblib and Pipeline:
            try:
                vec_file = _first_existing(self.lr_vec_path, UPLOADED["vec"])
                lr_file = _first_existing(self.lr_path, UPLOADED["lr"])
                if vec_file and lr_file:
                    vec = joblib.load(vec_file)
                    clf = joblib.load(lr_file)
                    self.lr_pipe = Pipeline([("tfidf", vec), ("lr", clf)])
                    self.lr_classes = list(getattr(clf, "classes_", self.labels))
                    self.has_lr = True
            except Exception:
                self.lr_pipe = None
                self.has_lr = False

        # Model‑centric (DistilBERT)
        self.tokenizer = None
        self.hf_model = None
        self.has_hf = False
        if AutoTokenizer and AutoModelForSequenceClassification:
            try:
                bert_dir = _first_existing(self.hf_dir, UPLOADED["bert"])
                if bert_dir:
                    self.tokenizer = AutoTokenizer.from_pretrained(bert_dir)
                    self.hf_model = AutoModelForSequenceClassification.from_pretrained(bert_dir)
                    self.hf_model.eval()
                    self.has_hf = True
            except Exception:
                self.tokenizer = None
                self.hf_model = None
                self.has_hf = False

    # ----------------------
    # Predictions
    # ----------------------
    def predict_lr(self, text: str):
        """Predict with TF‑IDF+LR and return {label, probs: {label: p}}.
        Ensures the probability dict covers self.labels order, filling 0 for any missing class.
        """
        if not self.has_lr:
            return None
        probs = self.lr_pipe.predict_proba([text])[0]
        clf = self.lr_pipe.named_steps.get("lr")
        classes = list(getattr(clf, "classes_", self.labels))
        # Winner
        idx = int(np.argmax(probs))
        pred_label = classes[idx]
        # Normalize to fixed label set order
        prob_map = {classes[i]: float(probs[i]) for i in range(len(classes))}
        full = {lbl: float(prob_map.get(lbl, 0.0)) for lbl in self.labels}
        return {"label": pred_label, "probs": full}

    def predict_hf(self, text: str):
        if not self.has_hf or torch is None:
            return None
        with torch.no_grad():
            toks = self.tokenizer(text, truncation=True, padding=True, return_tensors="pt")
            logits = self.hf_model(**toks).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        # Map via id2label if present, otherwise our default order
        id2label = getattr(self.hf_model.config, "id2label", {i: l for i, l in enumerate(self.labels)})
        idx = int(np.argmax(probs))
        pred_label = id2label.get(idx, self.labels[idx] if idx < len(self.labels) else str(idx))
        # Build full prob map aligned to self.labels when possible
        # If id2label is consistent with our labels, this will align.
        hf_probs: Dict[str, float] = {}
        for i, p in enumerate(probs):
            lbl = id2label.get(i, str(i))
            hf_probs[lbl] = float(p)
        # Ensure output in self.labels order
        full = {lbl: float(hf_probs.get(lbl, 0.0)) for lbl in self.labels}
        return {"label": pred_label, "probs": full}

    def predict_both(self, text: str):
        return {
            "data_centric": self.predict_lr(text),
            "model_centric": self.predict_hf(text),
        }

    # ----------------------
    # Key‑phrase highlighting
    # ----------------------
    def highlight_lr(self, text: str, top_k: int = 8) -> List[Tuple[str, float]]:
        """For Logistic Regression: contribution = tfidf_value * coef_for_predicted_class"""
        if not self.has_lr:
            return []
        vec = self.lr_pipe.named_steps.get("tfidf")
        clf = self.lr_pipe.named_steps.get("lr")
        if vec is None or clf is None:
            return []
        X = vec.transform([text])
        feature_names = np.array(vec.get_feature_names_out())
        # Predicted class (string) and index into coef_
        probs = clf.predict_proba(X)[0]
        pred_idx = int(np.argmax(probs))
        pred_class = clf.classes_[pred_idx]
        # coef_ is [n_classes, n_features]
        # Find the row index of the predicted class in clf.classes_
        class_row = int(np.where(clf.classes_ == pred_class)[0][0])
        coefs = clf.coef_[class_row]
        X_csr = X.tocsr()
        idxs = X_csr.indices
        vals = X_csr.data
        contrib: List[Tuple[str, float]] = []
        for i, v in zip(idxs, vals):
            contrib.append((feature_names[i], float(v * coefs[i])))
        contrib.sort(key=lambda t: abs(t[1]), reverse=True)
        return contrib[:top_k]

    # B) For DistilBERT: simple token occlusion importance (single text)
    # For each token, mask it and see drop in predicted class probability.
    def highlight_hf(self, text: str, top_k: int = 8) -> List[Tuple[str, float]]:
        if not self.has_hf or torch is None:
            return []
        with torch.no_grad():
            toks = self.tokenizer(text, return_tensors="pt", truncation=True)
            base_logits = self.hf_model(**toks).logits
            base_probs = torch.softmax(base_logits, dim=-1)[0]
            cls_idx = int(torch.argmax(base_probs).item())
            base_p = float(base_probs[cls_idx].item())
            words = text.split()
            importances: List[Tuple[str, float]] = []
            for wi, w in enumerate(words):
                masked_text = " ".join(words[:wi] + ["[MASK]"] + words[wi+1:])
                mtoks = self.tokenizer(masked_text, return_tensors="pt", truncation=True)
                mlogits = self.hf_model(**mtoks).logits
                mprobs = torch.softmax(mlogits, dim=-1)[0]
                drop = base_p - float(mprobs[cls_idx].item())
                importances.append((w, max(0.0, drop)))
            importances.sort(key=lambda t: t[1], reverse=True)
            return importances[:top_k]


# ---------- Utilities ----------
def render_html_highlight(text: str, spans: List[Tuple[str, float]], max_abs=None) -> str:
    """Return HTML with <mark> spans scaled by weight.
    spans: list of (token_or_ngram, weight) tuples. We highlight exact token matches.
    """
    if not spans:
        return text
    tokens = text.split()
    # Normalize weights
    weights = [w for _, w in spans]
    max_w = max(weights) if (weights and max(weights) > 0) else 1.0
    highlighted: List[str] = []
    # Simple token-level match
    score_map = {s.lower(): w / max_w for s, w in spans}
    for tok in tokens:
        key = tok.lower().strip(",.!?;:\"'()[]{}")
        if key in score_map and score_map[key] > 0:
            alpha = 0.2 + 0.6 * score_map[key]  # 0.2 to 0.8 opacity
            style = (
                f"background-color: rgba(255, 215, 0, {alpha}); "
                "border-radius:4px; padding:0 2px;"
            )
            highlighted.append(f"<mark style=\"{style}\">{tok}</mark>")
        else:
            highlighted.append(tok)
    return " ".join(highlighted)
