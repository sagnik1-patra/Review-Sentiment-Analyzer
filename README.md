🛒 ReviewSentiment Analyzer — README

Actionable customer‑feedback insights with fast NLP: clean and label reviews (rule‑based or ML), visualize patterns, export artifacts (HDF5/PKL/YAML/JSON), and ship instant predictions via Gradio UI or simple scripts.

✨ What this repo gives you

End‑to‑end pipeline (review_sentiment_analyzer.py):

Ingest CSV/JSON (map your text/rating/id/category columns)

Clean text (HTML/URLs/emoji/punct), lemmatize tokens

Sentiment engines: VADER/TextBlob (no training) or TF‑IDF + Logistic Regression

Visuals: sentiment distribution, Rating × Sentiment heatmap, positive/negative wordclouds

Insights: products with high rating but negative text (hidden issues), top keywords

Gradio UI for instant predictions (single & batch tabs)

Artifacts builder (Windows) (build_artifacts.py):

Reads your SQLite + CSV + hashes.txt

Dedupes via hashes.txt and updates it

Trains ML model & exports: HDF5, PKL, YAML, JSON, eval CSV

Prediction packs:

Notebook utilities (cells): single/batch/file prediction without argparse

CLI script (predict_simple.py): single text or whole file → predictions CSV

All three files are provided in canvas:

Review Sentiment Analyzer — Full Pipeline (python + Gradio + Visuals) → review_sentiment_analyzer.py

Review Sentiment — Build Artifacts From Sqlite+csv+hashes (windows) → build_artifacts.py

Review Sentiment — Simple Prediction Script (CLI + Importable) → predict_simple.py

Review Sentiment Notebook (Jupyter) — Build Artifacts & UI (optional, no argparse)

ReviewSentiment — Prediction & Results (Notebook Cells + Script) (optional, no argparse)

📦 Requirements

Python 3.11 (tested on Windows 10/11)

Packages (install via pip):

pandas numpy scikit-learn nltk textblob vaderSentiment matplotlib seaborn wordcloud gradio tqdm emoji beautifulsoup4 html5lib pyyaml joblib

(Optional): for BERT fine‑tuning: torch transformers datasets accelerate

Create a virtual environment (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\activate
pip install -U pip
pip install pandas numpy scikit-learn nltk textblob vaderSentiment matplotlib seaborn wordcloud gradio tqdm emoji beautifulsoup4 html5lib pyyaml joblib

Jupyter users: If you see TqdmWarning: IProgress not found, install widgets:

pip install ipywidgets
🗂️ Data & Paths

Default Windows inputs for the artifacts builder:

SQLite : C:\Users\NXTWAVE\Downloads\Review Sentiment Analyzer\archive\database.sqlite
CSV    : C:\Users\NXTWAVE\Downloads\Review Sentiment Analyzer\archive\Reviews.csv
Hashes : C:\Users\NXTWAVE\Downloads\Review Sentiment Analyzer\archive\hashes.txt
Output : C:\Users\NXTWAVE\Downloads\Review Sentiment Analyzer

Typical column names per dataset:

Amazon Fine Food Reviews: Text, Score, ProductId, Summary

Generic e‑commerce: reviewText, rating, productID, productCategory

🚀 Quickstart A — Full Pipeline (CSV/JSON → model + visuals + UI)

Script: review_sentiment_analyzer.py

# 1) One‑time NLTK download
python review_sentiment_analyzer.py --init-nltk


# 2) Train + analyze on a CSV (map your columns)
python review_sentiment_analyzer.py `
  --data "C:/path/Reviews.csv" `
  --text-col Text `
  --rating-col Score `
  --id-col ProductId `
  --category-col Category `
  --engine ml --do-eda `
  --outdir "C:/path/out"


# 3) Launch Gradio app (uses model if present, else rule-based)
python review_sentiment_analyzer.py --app --outdir "C:/path/out"


# 4) Quick rule-based scoring without training
python review_sentiment_analyzer.py `
  --data "C:/path/Reviews.csv" --text-col Text --rating-col Score `
  --engine rule --predict-only --outdir "C:/path/out"

Key outputs (in --outdir)

model_tfidf_lr.joblib — trained TF‑IDF + Logistic Regression model

eval_classification_report.csv, eval_confusion_matrix.csv, eval_predictions_valid.csv

viz_sentiment_distribution.png, viz_rating_x_sentiment_heatmap.png, viz_wordcloud_positive.png, viz_wordcloud_negative.png

top_keywords_positive.csv, top_keywords_negative.csv, products_hidden_issues.csv

Common flags

Flag	Description
--data	Path to CSV or JSON reviews file
--text-col	Column with raw review text
--rating-col	Numeric rating (e.g., Score 1–5)
--id-col	Product identifier (e.g., ProductId, asin)
--category-col	Product category name/id
--label-col	If you already have labeled sentiment (positive/negative/neutral)
--engine	ml (default) or rule
--do-eda	Save visuals/insights
--app	Launch Gradio UI
--outdir	Output folder (defaults to ./sentiment_out)
⚙️ Quickstart B — Build artifacts from SQLite + CSV + hashes (Windows)

Script: build_artifacts.py

# Runs with the default paths listed above
python build_artifacts.py

Exports to C:\Users\NXTWAVE\Downloads\Review Sentiment Analyzer:

processed_reviews.h5 — cleaned & normalized dataset

model_sentiment.pkl — trained TF‑IDF + Logistic Regression pipeline

build_metadata.yaml — inputs, schema, counts, metrics

insights.json — sentiment distribution + hidden‑issues products

eval_predictions_valid.csv — validation predictions

updates archive\hashes.txt with new dedupe hashes

🔮 Quickstart C — Predictions (single, batch, or file)
Option 1: Jupyter Notebook

Use the “ReviewSentiment — Prediction & Results (Notebook Cells + Script)” file.

# Single / batch in notebook
from pathlib import Path
TEST = [
  "The camera is great but the battery dies too fast",
  "Terrible build quality",
]
res = predict_many(TEST)
res.head()


# File prediction → CSV
file_out = predict_from_file(
    input_path=Path(r"C:\Users\NXTWAVE\Downloads\Review Sentiment Analyzer\archive\Reviews.csv"),
    text_col='Text',
    out_csv=Path(r"C:\Users\NXTWAVE\Downloads\Review Sentiment Analyzer\predictions_from_reviews.csv")
)
Option 2: CLI Script

Script: predict_simple.py

# Single text
python predict_simple.py --model "C:/path/out/model_tfidf_lr.joblib" `
  --text "Battery life is poor but display is gorgeous"


# Batch file → predictions CSV
python predict_simple.py --model "C:/path/out/model_tfidf_lr.joblib" `
  --file  "C:/Users/NXTWAVE/Downloads/Review Sentiment Analyzer/archive/Reviews.csv" `
  --text-col Text `
  --out   "C:/Users/NXTWAVE/Downloads/Review Sentiment Analyzer/predictions_from_reviews.csv"

If --model is omitted, the script will try model_tfidf_lr.joblib beside the script. If no model is found, it falls back to VADER (rule‑based) predictions.

Option 3: Gradio UI (from Full Pipeline)
python review_sentiment_analyzer.py --app --outdir "C:/path/out"

Open the printed http://127.0.0.1:7860 URL and use the Single or Batch tab.

🧠 How labels are created

Supervised path: If you pass --label-col, the model trains on your labels.

Rating‑derived: If --rating-col is numeric: >=4 → positive, <=2 → negative, else neutral.

Rule‑based fallback: VADER/TextBlob polarity mapped to positive/neutral/negative.

If neutral is very scarce (< ~50 samples), the pipeline automatically trains a binary model for stability.

📊 Visuals & Insights

Distribution: class balance for quick sanity checks

Rating × Sentiment heatmap: where stars disagree with text sentiment

Wordclouds: separate clouds for positive and negative reviews

Hidden‑issues products: high average rating but ≥15% negative text (≥10 reviews)
![Confusion Matrix Heatmap](viz_wordcloud_positive.png)
🧩 Project layout (suggested)
.
├─ review_sentiment_analyzer.py         # full CLI + Gradio
├─ build_artifacts.py                   # Windows builder (SQLite+CSV+hashes)
├─ predict_simple.py                    # lightweight CLI predictor
├─ notebooks/
│  ├─ ReviewSentiment_Notebook.ipynb    # optional: build & UI without argparse
│  └─ Prediction_Results.ipynb          # optional: notebook prediction pack
└─ outputs/
   ├─ model_tfidf_lr.joblib
   ├─ eval_*.csv
   ├─ viz_*.png
   ├─ processed_reviews.h5
   ├─ model_sentiment.pkl
   ├─ build_metadata.yaml
   └─ insights.json
🛡️ Troubleshooting

Argparse error in Jupyter:

ipykernel_launcher.py ... error: unrecognized arguments: -f ...kernel.json

Use the Notebook versions (no CLI flags) provided in canvas.

tqdm IProgress warning in Jupyter:

TqdmWarning: IProgress not found

Install widgets:

pip install ipywidgets

PowerShell can’t activate venv (execution policy):

# Current user scope is usually enough
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

Then re‑run: .\.venv\Scripts\Activate.ps1

Unicode/HTML in reviews: The cleaner strips emojis/HTML/URLs by default. Tweak clean_text() if you want to keep emojis (for downstream features).

Imbalanced classes: Use --do-eda to inspect class balance. Consider class weights or downsampling if needed.

⚡ Performance tips

Start with TF‑IDF + Logistic Regression (fast, strong baseline)

Tune --tfidf max_features and ngram_range inside the pipeline

For large corpora, consider sub‑sampling for faster iteration

Optional: switch to DistilBERT fine‑tuning if GPU is available (not included by default)

🔐 Privacy note

Reviews may contain PII. Mask or drop columns you don’t need before exporting artifacts. Avoid committing raw datasets to public repos.

🙌 Credits

VADER (Hutto & Gilbert), TextBlob, scikit‑learn, NLTK, Matplotlib, Seaborn, WordCloud, Gradio

Thanks to open review datasets (e.g., Amazon Reviews, IMDb) for benchmarking
AUTHOR
SAGNIK PATRA
