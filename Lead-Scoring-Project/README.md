# X Education Lead Scoring

ML pipeline + Streamlit app that scores leads 0-100 on likelihood to convert, so the sales team can prioritize calls instead of treating every lead the same.

## Structure

```
Lead-Scoring-Project/
├── data/
│   └── Lead_Scoring.csv
├── model/
│   ├── pipeline.joblib
│   ├── metadata.json
│   └── feature_importance.json
├── src/
│   ├── clean_data.py       # leakage removal + missing-data handling
│   ├── features.py         # binary mapping + ColumnTransformer
│   ├── train.py            # trains/compares models, saves artifacts
│   ├── scoring.py          # load_pipeline(), score_one(), score_batch()
│   └── agent.py            # Tab 1: tier lookup + Gemini outreach draft
├── app.py                  # Streamlit entrypoint, 3 tabs
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

For the Tab 1 outreach draft to actually call Gemini, add a `.env` file:

```
GEMINI_API_KEY=your-key-here
```

Without it, Tab 1 still shows the score/tier/recommendation, just skips the generated draft.

## Train

```bash
PYTHONPATH=src python3 src/train.py
```

Separate manual step, not run from the app. Splits the data 80/20 (stratified, fixed random_state), cleans it, cross-validates Logistic Regression / Random Forest / XGBoost on ROC-AUC, picks the best, checks calibration, converts probability to a 0-100 score, and finds the lowest threshold that hits ≥80% precision on the test set. Writes `pipeline.joblib`, `metadata.json`, `feature_importance.json`.

Current run: RandomForest, test ROC-AUC 0.876, Hot threshold 66 (80.3% precision), Warm threshold 46.

## Run

```bash
streamlit run app.py
```

- **Tab 1 - Single Lead Predictor:** form entry → score, tier badge, next action, AI-drafted outreach.
- **Tab 2 - Batch Scoring:** CSV upload → scored/ranked CSV download.
- **Tab 3 - Analytics Dashboard:** KPIs, score distribution, feature importance.

## Notes

- Leakage columns (`Tags`, `Lead Quality`, `Lead Profile`, the 4 Asymmetrique columns, `Last Notable Activity`, ID columns) are dropped before anything else — they're only known after a rep has already worked the lead.
- Missing categorical data (>~1%) becomes an explicit `"Unknown"` category instead of being dropped. Numeric missing → median. `"Select"` placeholder treated as missing. Rare categories (<1%) collapse to `"Other"`.
- Hot/Warm/Cold thresholds come from the precision-recall curve on the held-out test set, not a guess.
- Thresholds and recommendation text both live in `metadata.json`, nowhere else, so retuning doesn't need a code change.
- `clean_data.py` / `features.py` are shared by `train.py` and `scoring.py`, and `score_one()` / `score_batch()` route through the same transform — single-lead and batch scoring can't drift apart.
- The Gemini agent runs only from Tab 1, once per lookup, and only gets occupation/specialization/source/score/tier — no IDs, no full row.
- `OneHotEncoder(handle_unknown="ignore")` so an unseen category in a batch upload doesn't crash the app.
