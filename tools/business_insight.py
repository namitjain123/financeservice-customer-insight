"""Step 4 - turn the labelled, exploded data into executive-readable charts.

No LLM calls here - pure pandas + matplotlib over the output of Step 3.

Produces four plots:
  1) Top N Customer Pain Points       - which themes come up most often
  2) Pain Points by Product           - which products are driving which themes
  3) Neutral Share by Theme           - the one sentiment signal this data has
  4) Theme Trends Over Time           - are the top themes growing or shrinking

Differences from the original survey pipeline this replaces:
  * column names match the CFPB schema (`date`, lowercase) instead of the
    original survey export's (`Date`)
  * chart 3 was originally a stacked Negative/Neutral/Positive bar, on the
    assumption sentiment would vary enough to show relative severity. Real
    output showed otherwise: Positive is 0.0% for every theme (there's no
    such thing as a positive complaint here) and Negative sits at 88-100%
    for all of them - stacked on a 0-1 axis, every bar looked identical.
    Rebuilt to plot the one dimension that actually varies (Neutral share),
    see tools/business_insight.py's _theme_severity() docstring.
  * returns real per-chart row counts instead of a fixed "for example" claim
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import matplotlib.pyplot as plt

from config import settings

SENTIMENT_COLORS = {"Negative": "#E45756", "Neutral": "#F2CF5B", "Positive": "#72B7B2"}


def _top_pain_points(df: pd.DataFrame, top_n: int) -> tuple[pd.Series, str]:
    # One row per complaint (Step 2 clusters complaints, not topic phrases),
    # so this counts complaints per theme, not topic mentions - a complaint
    # contributes to exactly one theme, never inflating more than one bar.
    counts = df.groupby("general_topic_l1").size().sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    counts.iloc[::-1].plot(kind="barh", ax=ax, color="#4C78A8")
    ax.set_title(f"Top {top_n} Customer Pain Points")
    ax.set_xlabel("Complaints")
    ax.set_ylabel("Theme")
    fig.tight_layout()

    path = str(settings.ARTIFACTS / "plot_top_pain_points.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return counts, path


def _pain_points_by_product(df: pd.DataFrame) -> str:
    cross = df.pivot_table(
        index="general_topic_l1", columns=settings.PRODUCT_COLUMN,
        values="topic_discussed", aggfunc="count", fill_value=0,
    )

    fig, ax = plt.subplots(
        figsize=(max(8, 0.6 * (cross.shape[1] + 4)), max(6, 0.4 * (cross.shape[0] + 4)))
    )
    im = ax.imshow(cross.values, aspect="auto", cmap="Blues")
    ax.set_yticks(range(cross.shape[0]))
    ax.set_yticklabels(cross.index.tolist())
    ax.set_xticks(range(cross.shape[1]))
    ax.set_xticklabels(cross.columns.tolist(), rotation=45, ha="right")
    ax.set_title("Pain Points by Product (Counts)")
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()

    path = str(settings.ARTIFACTS / "heatmap_pain_points_by_product.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _theme_severity(df: pd.DataFrame, top_themes: list[str]) -> str:
    """Negative and Neutral share per theme, each on its own labeled panel.

    Originally a single 0-1 stacked Negative/Neutral/Positive bar. Measured
    against real output: Positive is 0.0% for every theme (every row here IS
    a complaint - there's no such thing as a "positive complaint" in this
    dataset, not just a rare one), so it's dropped from the plot rather than
    drawn as a permanently-empty slice. Negative (88-100%) and Neutral
    (0-11.5%) sit on wildly different scales - one shared 0-100% axis would
    flatten Neutral back into invisibility, the exact problem that motivated
    this rewrite - so each gets its own panel with its own y-range, and every
    bar is labeled with its exact percentage rather than relying on the
    reader to compare bar heights precisely.
    """
    counts = (
        df.groupby(["general_topic_l1", "customer_sentiment"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["Negative", "Neutral", "Positive"], fill_value=0)
    )
    shares = (counts.div(counts.sum(axis=1), axis=0).fillna(0) * 100)
    shares_top = shares.loc[[t for t in shares.index if t in top_themes]]

    fig, (ax_neg, ax_neu) = plt.subplots(1, 2, figsize=(14, 7))

    for ax, sentiment in ((ax_neg, "Negative"), (ax_neu, "Neutral")):
        series = shares_top[sentiment].sort_values(ascending=False)
        ax.bar(series.index, series.values, color=SENTIMENT_COLORS[sentiment])
        for i, v in enumerate(series.values):
            ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
        # Headroom for the label text above each bar - not capped at 100 even
        # though the data itself can't exceed it, or a bar reaching exactly
        # 100% has nowhere to put its own label and collides with the title.
        ax.set_ylim(0, max(series.max() * 1.15, 5))
        ax.set_title(f"{sentiment} Share by Theme")
        ax.set_ylabel(f"Share of Mentions That Are {sentiment} (%)")
        ax.set_xlabel("Theme (Top)")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    fig.suptitle("Theme Severity by Sentiment")
    # Reserve room below the rotated tick labels before placing the footnote
    # in figure coordinates - anchoring to the axes (ax.transAxes) put this
    # text at a fixed offset from the plot area, which the rotated labels'
    # own height then ran straight into.
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    fig.text(
        0.5, 0.02,
        "Positive sentiment is 0% for every theme in this data and is omitted above - "
        "every row is a complaint, so there is no positive class to compare against.",
        ha="center", fontsize=8, color="dimgray",
    )

    path = str(settings.ARTIFACTS / "theme_severity_stacked.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _theme_trends(df: pd.DataFrame, top_themes: list[str]) -> str:
    dft = df.copy()
    dft[settings.DATE_COLUMN] = pd.to_datetime(dft[settings.DATE_COLUMN], errors="coerce")
    dft = dft.dropna(subset=[settings.DATE_COLUMN])
    dft["month"] = dft[settings.DATE_COLUMN].dt.to_period("M").dt.to_timestamp()

    trend = (
        dft[dft["general_topic_l1"].isin(top_themes)]
        .groupby(["month", "general_topic_l1"]).size()
        .reset_index(name="count")
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for theme in top_themes:
        sub = trend[trend["general_topic_l1"] == theme]
        ax.plot(sub["month"], sub["count"], marker="o", label=theme)

    ax.set_title("Theme Trends Over Time (Top Themes)")
    ax.set_ylabel("Complaints per Month")
    ax.set_xlabel("Month")
    ax.legend(title="Theme", bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()

    path = str(settings.ARTIFACTS / "theme_trends_over_time.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def generate_insights(
    input_csv=settings.OUTPUT_CSV,
    top_n: int = settings.TOP_N_THEMES,
) -> dict[str, Any]:
    """Read the labelled, exploded CSV and produce all four charts."""
    df = pd.read_csv(input_csv)
    required = {"general_topic_l1", "topic_discussed", settings.PRODUCT_COLUMN,
                settings.DATE_COLUMN, "customer_sentiment"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"input CSV is missing required columns: {missing}")

    top_counts, p1 = _top_pain_points(df, top_n)
    top_themes = top_counts.index.tolist()

    p2 = _pain_points_by_product(df)
    p3 = _theme_severity(df, top_themes)
    p4 = _theme_trends(df, top_themes)

    return {
        "status": "success",
        "input_rows": len(df),
        "distinct_themes": int(df["general_topic_l1"].nunique()),
        "top_n": top_n,
        "top_themes": top_counts.to_dict(),
        "plots": [p1, p2, p3, p4],
        "input_path": str(input_csv),
    }


if __name__ == "__main__":
    import sys

    n = int(sys.argv[1]) if len(sys.argv) > 1 else settings.TOP_N_THEMES
    print(json.dumps(generate_insights(top_n=n), indent=2))
