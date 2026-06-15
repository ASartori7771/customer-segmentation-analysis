"""
LLM Layer
----------
Two features powered by the OpenAI API:
  1. explain_anomalies  — reads the latest pipeline summary JSON and returns
                          a plain-English data quality report.
  2. generate_marketing_strategies — takes the cluster_info DataFrame and
                                     returns AI-generated persona names and
                                     campaign ideas per cluster.

Usage (from the notebook):
    from llm import explain_anomalies, generate_marketing_strategies
"""

import os
import glob
import json
import pandas as pd
from openai import OpenAI


# ──────────────────────────────────────────────
# HELPER
# ──────────────────────────────────────────────

def ask_gpt(system_prompt: str, user_prompt: str, max_tokens: int = 500) -> str:
    """
    Single call to GPT-4o-mini.
    Returns the response text as a string.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=max_tokens,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
    )
    return response.choices[0].message.content.strip()


# ──────────────────────────────────────────────
# FEATURE 1 — ANOMALY EXPLAINER
# ──────────────────────────────────────────────

SYSTEM_ANOMALY = """
You are a data quality analyst writing for a non-technical business audience.
You will receive a summary of a data pipeline run, including rows that were
dropped or modified and the reasons why.
Write a clear, concise paragraph (max 100 words) explaining what data quality
issues were found, what was done about them, and whether the analyst should
be concerned. Use plain English — no technical jargon.
""".strip()


def explain_anomalies(log_dir: str = "./logs") -> str:
    """
    Finds the most recent pipeline summary JSON, extracts anomaly info,
    and returns a plain-English GPT explanation.
    Run the pipeline at least once before calling this.
    """
    summaries = sorted(glob.glob(f"{log_dir}/summary_*.json"))
    if not summaries:
        return "No pipeline summary found. Run pipeline.py first."

    with open(summaries[-1]) as f:
        summary = json.load(f)

    lines = [
        f"Pipeline run ID: {summary['run_id']}",
        f"Total rows ingested: {summary['total_rows_in']}",
        f"Total rows after cleaning: {summary['total_rows_out']}",
        f"Total rows dropped: {summary['rows_dropped']}",
        "",
    ]

    for source in summary["sources"]:
        lines.append(f"File: {source['source']}")
        lines.append(f"  - Duplicate rows removed: {source['duplicates_dropped']}")
        lines.append(f"  - Outlier rows removed: {source['outliers_dropped']}")
        for col, count in source["rows_imputed"].items():
            lines.append(f"  - Missing values filled in '{col}': {count} row(s)")
        for w in source["warnings"]:
            lines.append(f"  - Warning: {w}")

    return ask_gpt(SYSTEM_ANOMALY, "\n".join(lines), max_tokens=200)


# ──────────────────────────────────────────────
# FEATURE 2 — MARKETING STRATEGY GENERATOR
# ──────────────────────────────────────────────

SYSTEM_MARKETING = """
You are a senior marketing strategist specialising in customer segmentation.
You will receive statistics for a customer segment: average age, average annual
income, average spending score (1-100), and their top product categories.

Your response must be valid JSON with exactly this structure:
{
  "persona_name": "A memorable 2-4 word name for this segment",
  "persona_description": "One sentence describing who these people are",
  "campaigns": [
    {
      "title": "Campaign name",
      "channel": "Best channel (e.g. Instagram, Email, In-store)",
      "idea": "One sentence describing the campaign concept"
    }
  ]
}

Return 3 campaigns. Return ONLY the JSON object — no explanation, no markdown fences.
""".strip()


def _build_cluster_prompt(cluster_id: int, stats: dict) -> str:
    top_cats = ", ".join(
        f"{cat} ({count} customers)"
        for cat, count in stats["top_categories"]
    )
    return (
        f"Cluster ID: {cluster_id}\n"
        f"Average age: {stats['age']:.1f} years\n"
        f"Average annual income: ${stats['income']:.0f}k\n"
        f"Average spending score: {stats['spending']:.1f} / 100\n"
        f"Top product categories: {top_cats}"
    )


def generate_marketing_strategies(cluster_info: pd.DataFrame) -> dict:
    """
    Loops over each cluster in cluster_info, calls GPT for each one,
    and returns a dict keyed by cluster ID with persona + campaign ideas.

    cluster_info must have columns: Cluster, Age, Income, Spent, Favorite_Category
    (exactly as built in the notebook's interpretation cell).
    """
    results = {}

    for cluster_id in sorted(cluster_info["Cluster"].unique()):
        data = cluster_info[cluster_info["Cluster"] == cluster_id]

        stats = {
            "age":            data["Age"].mean(),
            "income":         data["Income"].mean(),
            "spending":       data["Spent"].mean(),
            "top_categories": list(data["Favorite_Category"].value_counts().head(3).items()),
        }

        print(f"Generating strategy for Cluster {cluster_id}...")
        raw = ask_gpt(SYSTEM_MARKETING, _build_cluster_prompt(cluster_id, stats), max_tokens=400)

        # Strip markdown fences if the model adds them despite instructions
        clean = raw.replace("```json", "").replace("```", "").strip()

        try:
            parsed = json.loads(clean)
        except json.JSONDecodeError:
            parsed = {"error": "GPT returned non-JSON output", "raw": raw}

        results[cluster_id] = {"stats": stats, "strategy": parsed}

    return results
