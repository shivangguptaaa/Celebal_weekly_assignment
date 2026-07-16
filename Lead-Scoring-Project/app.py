"""Streamlit app entrypoint. Tabs 1, 2, and 3 are all wired up."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from scoring import load_pipeline, score_batch, score_one  # noqa: E402
from agent import draft_outreach_tool  # noqa: E402

DATA_PATH = Path(__file__).parent / "data" / "Lead_Scoring.csv"
MODEL_DIR = Path(__file__).parent / "model"

CATEGORICAL_DROPDOWN_COLUMNS = [
    "Lead Origin",
    "Lead Source",
    "Last Activity",
    "Country",
    "Specialization",
    "How did you hear about X Education",
    "What is your current occupation",
    "City",
]

BINARY_FLAG_COLUMNS = [
    "Do Not Email",
    "Do Not Call",
    "A free copy of Mastering The Interview",
]

TIER_COLORS = {"Hot": "#e74c3c", "Warm": "#f39c12", "Cold": "#3498db"}


@st.cache_resource
def get_pipeline_and_metadata():
    return load_pipeline(str(MODEL_DIR))


@st.cache_data
def get_dropdown_options() -> dict[str, list[str]]:
    raw = pd.read_csv(DATA_PATH)
    options = {}
    for col in CATEGORICAL_DROPDOWN_COLUMNS:
        if col not in raw.columns:
            continue
        values = [v for v in raw[col].dropna().unique().tolist() if v != "Select"]
        options[col] = sorted(values)
    return options


@st.cache_data
def get_feature_importance() -> list[dict]:
    with open(MODEL_DIR / "feature_importance.json") as f:
        return json.load(f)


@st.cache_data
def get_default_scored_batch(_pipeline, metadata: dict) -> pd.DataFrame:
    """Fallback dashboard data source when no batch has been uploaded yet."""
    raw = pd.read_csv(DATA_PATH)
    return score_batch(raw, _pipeline, metadata)


def render_result(score: int, tier: str, recommendation: str) -> None:
    color = TIER_COLORS[tier]
    st.markdown(
        f"""
        <div style="padding: 1.25rem; border-radius: 0.5rem; background-color: {color}22;
                    border: 2px solid {color}; text-align: center;">
            <div style="font-size: 0.9rem; color: {color}; font-weight: 600; letter-spacing: 0.05em;">
                {tier.upper()} LEAD
            </div>
            <div style="font-size: 3rem; font-weight: 700; color: {color};">
                {score}
            </div>
            <div style="font-size: 1rem; color: #333;">
                {recommendation}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tab1(pipeline, metadata, dropdown_options, feature_columns) -> None:
    st.subheader("Single Lead Predictor")
    st.caption("Enter a lead's details to get an instant score and recommended next action.")

    with st.form("single_lead_form"):
        st.markdown("**Behavioral**")
        col1, col2, col3 = st.columns(3)
        total_visits = col1.number_input("Total Visits", min_value=0, value=0, step=1)
        time_on_site = col2.number_input(
            "Total Time Spent on Website (seconds)", min_value=0, value=0, step=10
        )
        page_views = col3.number_input("Page Views Per Visit", min_value=0.0, value=0.0, step=0.5)

        last_activity = None
        if "Last Activity" in feature_columns:
            last_activity = st.selectbox("Last Activity", dropdown_options.get("Last Activity", []))

        st.markdown("**Source**")
        col4, col5, col6 = st.columns(3)
        lead_origin = col4.selectbox("Lead Origin", dropdown_options.get("Lead Origin", []))
        lead_source = col5.selectbox("Lead Source", dropdown_options.get("Lead Source", []))
        how_heard = None
        if "How did you hear about X Education" in feature_columns:
            how_heard = col6.selectbox(
                "How did you hear about X Education",
                dropdown_options.get("How did you hear about X Education", []),
            )

        st.markdown("**Profile**")
        col7, col8 = st.columns(2)
        occupation = col7.selectbox(
            "Current Occupation", dropdown_options.get("What is your current occupation", [])
        )
        specialization = col8.selectbox("Specialization", dropdown_options.get("Specialization", []))
        col9, col10 = st.columns(2)
        city = col9.selectbox("City", dropdown_options.get("City", []))
        country = col10.selectbox("Country", dropdown_options.get("Country", []))

        flag_values = {}
        flag_cols_present = [c for c in BINARY_FLAG_COLUMNS if c in feature_columns]
        if flag_cols_present:
            st.markdown("**Preferences**")
            cols = st.columns(len(flag_cols_present))
            for flag_col, col in zip(flag_cols_present, cols):
                flag_values[flag_col] = col.selectbox(flag_col, ["No", "Yes"])

        submitted = st.form_submit_button("Score This Lead")

    if submitted:
        lead_dict = {
            "TotalVisits": total_visits,
            "Total Time Spent on Website": time_on_site,
            "Page Views Per Visit": page_views,
            "Lead Origin": lead_origin,
            "Lead Source": lead_source,
            "What is your current occupation": occupation,
            "Specialization": specialization,
            "City": city,
            "Country": country,
        }
        if last_activity is not None:
            lead_dict["Last Activity"] = last_activity
        if how_heard is not None:
            lead_dict["How did you hear about X Education"] = how_heard
        lead_dict.update(flag_values)

        score, tier, recommendation = score_one(lead_dict, pipeline, metadata)
        render_result(score, tier, recommendation)

        st.markdown("**Suggested Outreach**")
        try:
            draft = draft_outreach_tool(
                occupation=occupation,
                specialization=specialization,
                lead_source=lead_source,
                score=score,
                tier=tier,
            )
            st.text_area("Draft (edit before sending)", value=draft, height=220)
        except RuntimeError as e:
            st.info(str(e))
        except Exception as e:
            st.warning(f"Couldn't generate an outreach draft right now: {e}")


def render_tab2(pipeline, metadata) -> None:
    st.subheader("Batch Scoring")
    st.caption("Upload a CSV of new leads to score them all at once.")

    uploaded_file = st.file_uploader("Upload leads CSV", type=["csv"])

    if uploaded_file is None:
        return

    try:
        upload_df = pd.read_csv(uploaded_file)
    except Exception:
        st.error("Couldn't read that file as a CSV. Please check the format and try again.")
        return

    try:
        scored_df = score_batch(upload_df, pipeline, metadata)
    except ValueError as e:
        st.error(str(e))
        return

    st.session_state["last_scored_df"] = scored_df
    st.session_state["last_scored_label"] = f"uploaded batch ({uploaded_file.name})"

    st.success(f"Scored {len(scored_df)} leads.")
    st.dataframe(scored_df, width='stretch')

    csv_bytes = scored_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download scored leads as CSV",
        data=csv_bytes,
        file_name="scored_leads.csv",
        mime="text/csv",
    )


def render_tab3(pipeline, metadata) -> None:
    st.subheader("Analytics Dashboard")

    scored_df = st.session_state.get("last_scored_df")
    if scored_df is not None:
        source_label = st.session_state.get("last_scored_label", "your uploaded batch")
    else:
        scored_df = get_default_scored_batch(pipeline, metadata)
        source_label = "the training dataset (upload a batch in Tab 2 to see your own data here)"

    st.caption(f"Showing: {source_label}")

    total_leads = len(scored_df)
    avg_score = scored_df["Lead Score"].mean()
    pct_hot = (scored_df["Tier"] == "Hot").mean() * 100
    precision_at_hot = metadata["precision_at_hot_threshold"] * 100
    target_met = metadata["precision_target_met"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Leads", f"{total_leads:,}")
    col2.metric("Average Score", f"{avg_score:.1f}")
    col3.metric("% Flagged Hot", f"{pct_hot:.1f}%")
    col4.metric(
        "Precision @ Hot (validation)",
        f"{precision_at_hot:.1f}%",
        delta="meets 80% target" if target_met else "below 80% target",
        delta_color="normal" if target_met else "inverse",
    )

    st.markdown("**Score Distribution**")
    hist_fig = px.histogram(
        scored_df,
        x="Lead Score",
        nbins=25,
        color="Tier",
        color_discrete_map={"Hot": "#e74c3c", "Warm": "#f39c12", "Cold": "#3498db"},
    )
    hist_fig.update_layout(bargap=0.05, height=350)
    st.plotly_chart(hist_fig, width='stretch')

    st.markdown("**What Drives the Score (Feature Importance)**")
    importance_data = get_feature_importance()
    top_n = 15
    importance_df = pd.DataFrame(importance_data[:top_n])
    importance_fig = px.bar(
        importance_df.sort_values("importance"),
        x="importance",
        y="feature",
        orientation="h",
    )
    importance_fig.update_layout(height=450, yaxis_title="", xaxis_title="Relative importance")
    st.plotly_chart(importance_fig, width='stretch')


def main() -> None:
    st.set_page_config(page_title="X Education Lead Scoring", layout="centered")
    st.title("X Education Lead Scoring")

    pipeline, metadata = get_pipeline_and_metadata()
    dropdown_options = get_dropdown_options()
    feature_columns = metadata["feature_columns"]

    tab1, tab2, tab3 = st.tabs(["Single Lead Predictor", "Batch Scoring", "Analytics Dashboard"])

    with tab1:
        render_tab1(pipeline, metadata, dropdown_options, feature_columns)

    with tab2:
        render_tab2(pipeline, metadata)

    with tab3:
        render_tab3(pipeline, metadata)


if __name__ == "__main__":
    main()
