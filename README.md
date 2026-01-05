# Domain-Specific Sentiment Analysis (Reddit Break-Ups)
Multi-class sentiment/emotion classification for breakup posts, comparing:
- **Model-centric**: DistilBERT fine-tuning
- **Data-centric**: TF-IDF + Logistic Regression (+ simple augmentation)
Includes a **Streamlit UI** with **LLM explanations** (“why” the text fits a class).

**Classes:** `anger`, `confusion`, `relief`, `sadness`

---

## 1) Quick Start (UI)

### Requirements
```bash
pip install --upgrade pip
pip install streamlit transformers tokenizers safetensors scikit-learn joblib numpy pandas requests
# CPU Torch (Windows/CPU):
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

### Run
```bash
streamlit run breakup_sentiment_ui_app.py
```

In the **sidebar**, set paths (edit if different):

- **TF-IDF folder**  
  `G:\My Drive\Sentiment Analysis\artifacts\tfidf_output`
- **DistilBERT folder** (parent containing `best/` and `best_tok/`)  
  `G:\My Drive\Sentiment Analysis\artifacts\distilbert_out_chunked`
- **demo.csv path (optional)**  
  `G:\My Drive\Sentiment Analysis\artifacts\demo.csv`

The app shows prediction, probability bars, token highlights, and an LLM “explain like I’m 10”.

---

## 2) Change LLM Provider (for explanations)

The app prefers **OpenAI-compatible** APIs if keys are set; otherwise it uses **Ollama** locally.

### OpenAI (ChatGPT) example
```bash
# PowerShell (set before running streamlit)
$env:OPENAI_API_KEY = "sk-...your_openai_key..."
$env:OPENAI_BASE_URL = "https://api.openai.com/v1"
$env:OPENAI_MODEL = "gpt-4o-mini"
```

### Other OpenAI-compatible providers
Replace base URL + model, e.g. OpenRouter/Together/Groq.

### Ollama (local, fallback)
Install Ollama and (optionally):
```bash
$env:OLLAMA_MODEL = "llama3.1"
$env:OLLAMA_URL = "http://localhost:11434"
```

---

## 3) Reproduce Training

> Scripts live in `notebooks_or_scripts/` (names below).

### A) EDA & Baselines
```bash
python "dataset_review,_eda_&_baseline_models.py"
```
Outputs: data summary, simple baselines.

### B) Model-centric (DistilBERT fine-tune)
```bash
python "model_centric_—_distilbert_fine_tuning.py"
```
Outputs saved under `artifacts/distilbert_out_chunked/`:
- `best/` → final model (`config.json`, `model.safetensors`, tokenizer files if exported there)
- `best_tok/` → tokenizer files (app can load tokenizer from here)
- `best_summary.json`, `grid_results.json`

### C) Data-centric (TF-IDF + LR + augmentation)
```bash
python "data_centric_all_in_one.py"
```
Outputs saved under `artifacts/tfidf_output/`:
- `tfidf_vectorizer.joblib`
- `logreg_model.joblib`
- `metrics.json` (optional)

---

## 4) Expected Artifacts Layout

```
artifacts/
  tfidf_output/
    tfidf_vectorizer.joblib
    logreg_model.joblib
    metrics.json
  distilbert_out_chunked/
    best/            # model
      config.json
      model.safetensors
      tokenizer.json
      tokenizer_config.json
      special_tokens_map.json
      vocab.txt
    best_tok/        # tokenizer (kept for completeness)
    best_summary.json
    grid_results.json
```

> The UI auto-detects `best/` for model and `best_tok/` for tokenizer.

---

## 5) Notes & Tips
- For long posts, you can set `BERT_MAX_LENGTH=384` (env var) before running to use more context.
- A small **rules re-ranker** in the UI nudges very clear cues (emoji/ALL CAPS/“idk”)—toggle in the sidebar.
- If you see “artifacts not found”, check the paths and filenames exactly as above.
