"""Load the trained pipeline and score leads. Used by both app tabs."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_pipeline(model_dir: str = "model"):
    model_dir = Path(model_dir)
    pipeline = joblib.load(model_dir / "pipeline.joblib")
    with open(model_dir / "metadata.json") as f:
        metadata = json.load(f)
    return pipeline, metadata


# Fallback only -- used if an older metadata.json predates the
# "recommendations" key. The authoritative copy is always metadata.json,
# written once by train.py, per Rules.md #5 (single source of truth).
_DEFAULT_RECOMMENDATIONS = {
    "Hot": "Call within 2 hours",
    "Warm": "Add to nurture sequence",
    "Cold": "Newsletter only",
}


def tier_and_recommendation(score: int, metadata: dict) -> tuple[str, str]:
    hot = metadata["hot_threshold"]
    warm = metadata["warm_threshold"]
    recommendations = metadata.get("recommendations", _DEFAULT_RECOMMENDATIONS)

    if score >= hot:
        tier = "Hot"
    elif score >= warm:
        tier = "Warm"
    else:
        tier = "Cold"

    return tier, recommendations.get(tier, _DEFAULT_RECOMMENDATIONS[tier])


def score_one(lead_dict: dict, pipeline, metadata: dict) -> tuple[int, str, str]:
    df = pd.DataFrame([lead_dict])
    prob = pipeline.predict_proba(df)[0, 1]
    score = int(round(prob * 100))
    tier, recommendation = tier_and_recommendation(score, metadata)
    return score, tier, recommendation


def score_batch(df: pd.DataFrame, pipeline, metadata: dict) -> pd.DataFrame:
    required = set(metadata["feature_columns"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Upload is missing required columns: {sorted(missing)}")

    probs = pipeline.predict_proba(df)[:, 1]
    scores = np.round(probs * 100).astype(int)

    tiers, recommendations = [], []
    for s in scores:
        tier, rec = tier_and_recommendation(int(s), metadata)
        tiers.append(tier)
        recommendations.append(rec)

    result = df.copy()
    result["Lead Score"] = scores
    result["Tier"] = tiers
    result["Recommended Action"] = recommendations
    return result


if __name__ == "__main__":
    pipeline, metadata = load_pipeline()

    # a handful of hand-built example leads, deliberately covering a range
    # of profiles and some missing/unseen values
    example_leads = [
        {
            "Lead Origin": "API",
            "Lead Source": "Olark Chat",
            "Do Not Email": "No",
            "Do Not Call": "No",
            "TotalVisits": 0,
            "Total Time Spent on Website": 0,
            "Page Views Per Visit": 0,
            "Last Activity": "Page Visited on Website",
            "Country": "India",
            "Specialization": "Select",
            "How did you hear about X Education": "Select",
            "What is your current occupation": "Unemployed",
            "City": "Mumbai",
            "A free copy of Mastering The Interview": "No",
        },
        {
            "Lead Origin": "Lead Add Form",
            "Lead Source": "Reference",
            "Do Not Email": "No",
            "Do Not Call": "No",
            "TotalVisits": 8,
            "Total Time Spent on Website": 1200,
            "Page Views Per Visit": 4.5,
            "Last Activity": "SMS Sent",
            "Country": "India",
            "Specialization": "Business Administration",
            "How did you hear about X Education": "Word Of Mouth",
            "What is your current occupation": "Working Professional",
            "City": "Thane & Outskirts",
            "A free copy of Mastering The Interview": "Yes",
        },
        {
            # sparse input: only a few fields provided, rest should be
            # imputed/defaulted gracefully rather than crashing
            "Lead Origin": "Landing Page Submission",
            "Lead Source": "Direct Traffic",
            "TotalVisits": 3,
            "Total Time Spent on Website": 250,
        },
        {
            # an unseen category the pipeline never saw during training
            "Lead Origin": "API",
            "Lead Source": "SomeBrandNewChannel",
            "Do Not Email": "Yes",
            "TotalVisits": 1,
            "Total Time Spent on Website": 30,
            "Page Views Per Visit": 1.0,
            "Last Activity": "Unsubscribed",
            "Country": "France",
            "Specialization": "Select",
            "What is your current occupation": "Student",
        },
    ]

    print("score_one() checks:")
    for i, lead in enumerate(example_leads, start=1):
        score, tier, recommendation = score_one(lead, pipeline, metadata)
        print(f"  Lead {i}: score={score:3d}  tier={tier:5s}  action='{recommendation}'")

    print("\nscore_batch() check on the full training CSV:")
    full_df = pd.read_csv("data/Lead_Scoring.csv")
    scored = score_batch(full_df, pipeline, metadata)
    print(scored[["Lead Score", "Tier", "Recommended Action"]].describe(include="all"))
    print(f"\nTier distribution:\n{scored['Tier'].value_counts()}")
