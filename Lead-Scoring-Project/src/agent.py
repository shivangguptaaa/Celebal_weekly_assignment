"""Tab 1 agent module: deterministic next-action lookup + a Gemini-drafted
outreach message. Called only from Tab 1, once per rep-initiated lookup --
never invoked in a loop over Tab 2 batch rows.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

from scoring import tier_and_recommendation

load_dotenv()  # reads GEMINI_API_KEY from a .env file in the project root, if present

GEMINI_MODEL = "gemini-2.5-flash-lite"


def next_action_tool(score: int, metadata: dict) -> dict:
    """Pure deterministic lookup against metadata.json's thresholds.

    No LLM call here -- this is the one piece of the recommendation that
    can never be a hallucination. Reuses the exact same function scoring.py
    already uses, so there is only one place the Hot/Warm/Cold logic lives.
    """
    tier, recommendation = tier_and_recommendation(score, metadata)
    return {"tier": tier, "recommendation": recommendation}


def _get_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to a .env file in the project "
            "root (see .env.example) -- never hardcode it in source."
        )
    from google import genai

    return genai.Client(api_key=api_key)


def draft_outreach_tool(occupation: str, specialization: str, lead_source: str, score: int, tier: str) -> str:
    """Calls Gemini to draft a short, personalized outreach message.

    Field-minimized by design: only occupation, specialization, lead
    source, score, and tier are ever sent. No identifiers, no name, email,
    or phone, and never the full lead row -- this function's signature is
    the enforcement mechanism, since there's simply nowhere to pass
    anything else in.
    """
    client = _get_client()

    prompt = (
        "You are helping a sales rep at an ed-tech company (X Education) write a short, "
        "warm outreach message to a prospective student lead. Choose whichever fits "
        "better: a brief email (with subject line) or a 3-4 sentence call script. "
        "Use [Name] as a placeholder for the lead's name, since it isn't provided. "
        "Keep it concise and specific to this lead's profile. Do not invent any facts "
        "about the lead beyond what's given below.\n\n"
        "Lead profile:\n"
        f"- Occupation: {occupation or 'Unknown'}\n"
        f"- Area of interest: {specialization or 'Unknown'}\n"
        f"- How they found us: {lead_source or 'Unknown'}\n"
        f"- Lead score: {score}/100\n"
        f"- Priority tier: {tier}\n"
    )

    response = client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
    return response.text


def run_agent(lead_dict: dict, score: int, metadata: dict) -> dict:
    """Tab 1 entry point: next_action_tool first, then draft_outreach_tool.

    `lead_dict` is the full form input, but only the minimized fields
    below are ever pulled out of it and forwarded to the LLM call.
    """
    action = next_action_tool(score, metadata)

    draft = draft_outreach_tool(
        occupation=lead_dict.get("What is your current occupation", ""),
        specialization=lead_dict.get("Specialization", ""),
        lead_source=lead_dict.get("Lead Source", ""),
        score=score,
        tier=action["tier"],
    )

    return {"tier": action["tier"], "recommendation": action["recommendation"], "draft": draft}


if __name__ == "__main__":
    # next_action_tool needs no network and no API key -- verify it directly
    fake_metadata = {"hot_threshold": 66, "warm_threshold": 46}
    for test_score in [90, 55, 20]:
        print(f"score={test_score}: {next_action_tool(test_score, fake_metadata)}")

    print()
    if os.environ.get("GEMINI_API_KEY"):
        result = run_agent(
            lead_dict={
                "What is your current occupation": "Working Professional",
                "Specialization": "Business Administration",
                "Lead Source": "Reference",
                "Prospect ID": "should-never-be-sent",  # proves field minimization
            },
            score=94,
            metadata=fake_metadata,
        )
        print(result)
    else:
        print("GEMINI_API_KEY not set -- skipping the live draft_outreach_tool call.")
        print("(next_action_tool above needs no API key and already ran successfully.)")
