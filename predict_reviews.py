
#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

OUT_DIR = Path(r"C:\Users\NXTWAVE\Downloads\Review Sentiment Analyzer")
MODEL_PATH = OUT_DIR / 'model_sentiment.pkl'

def rule_score(text: str) -> float:
    vs = SentimentIntensityAnalyzer()
    return float(vs.polarity_scores(text).get('compound', 0.0))

def score_to_label(score: float, pos=0.2, neg=-0.2) -> str:
    return 'positive' if score>=pos else ('negative' if score<=neg else 'neutral')

def predict_one(text: str):
    text = (text or "").strip()
    if not text:
        return "", 0.0
    try:
        pipe = joblib.load(MODEL_PATH)
        lab = pipe.predict([text])[0]
        try:
            proba = pipe.predict_proba([text])
            conf = float(np.max(proba))
        except Exception:
            conf = 0.0
        return lab, conf
    except Exception:
        comp = rule_score(text)
        return score_to_label(comp), float(abs(comp))

def predict_file(file_path: Path, text_col: str, out_path: Path):
    if file_path.suffix.lower() == '.json':
        df_in = pd.read_json(file_path, lines=True)
    else:
        df_in = pd.read_csv(file_path)
    if text_col not in df_in.columns:
        raise SystemExit(f"Column '{text_col}' not in file. Available: {df_in.columns.tolist()}")
    rows = []
    for t in df_in[text_col].astype(str).fillna(""):
        lab, conf = predict_one(t)
        rows.append((lab, conf))
    df_in['sentiment'] = [r[0] for r in rows]
    df_in['confidence'] = [r[1] for r in rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_in.to_csv(out_path, index=False)
    print(f"[OK] Wrote predictions → {out_path}")

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--text', type=str, help='Single review text to predict')
    ap.add_argument('--file', type=str, help='CSV/JSON path for batch prediction')
    ap.add_argument('--text-col', type=str, default='reviewText', help='Text column name in the file')
    ap.add_argument('--out', type=str, default=str(OUT_DIR / 'predictions.csv'))
    args = ap.parse_args()

    if args.text:
        lab, conf = predict_one(args.text)
        print(json.dumps({'text': args.text, 'sentiment': lab, 'confidence': conf}, ensure_ascii=False))
    elif args.file:
        predict_file(Path(args.file), args.text_col, Path(args.out))
    else:
        ap.print_help()
