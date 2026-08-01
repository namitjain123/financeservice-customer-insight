"""Web UI for the pipeline. Same shape as the original project's Streamlit app,
adapted for this schema and rewritten around three things that changed since:

  * Progress is shown from each step's REAL return value, not streamed agent
    narration. The original streamed `team.run_stream()` text - exactly the
    mechanism that, in an earlier version of this project, let an agent
    announce "1,000 responses processed (for example)" against a 100-row
    file. main.py doesn't trust narration; neither does this UI.
  * Results and evaluation are read from whatever's already on disk in
    artifacts/, not gated behind "did you click Run in this exact browser
    session." A CLI run (python main.py / python -m tools.topic_clustering)
    populates the same files this UI reads - reopening the page should show
    them immediately, not force a costly re-run just to look at what's
    already there.
  * Evaluation lets you pick which real CFPB label to score clusters
    against - the fine-grained `Issue` field or the coarser `product` field
    - since they measure different things and neither alone is the full
    picture. Comparing the pipeline's clusters against real labels at all is
    this project's actual differentiator - the original had no equivalent,
    because its dataset had no ground truth to check against.

Run with:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import asyncio

import pandas as pd
import streamlit as st

from config import settings
from tools.business_insight import generate_insights
from tools.cluster_labelling import label_clusters
from tools.topic_clustering import cluster_topics
from tools.topic_extraction import extract_topics

st.set_page_config(page_title="Complaint Insight Pipeline", layout="wide")
st.title("Financial Services Customer Insights")
st.markdown(
    "Turns CFPB consumer complaints into a labelled dataset and four executive "
    "charts, then checks the clustering against the complaints' real category "
    "labels - a step most pipelines like this skip."
)

# --- data source --------------------------------------------------------
st.markdown("---")
st.subheader("1. Data source")

source = st.radio(
    "Run on:",
    ["Built-in CFPB dataset (has ground truth for evaluation)", "Upload a CSV"],
    index=0,
)

input_csv = settings.INPUT_CSV

if source.startswith("Built-in"):
    if not input_csv.exists():
        st.error(f"{input_csv} not found. See data.md for how to generate it.")
        st.stop()

    full = pd.read_csv(input_csv)
    st.success(f"{len(full):,} complaints available, {full['product'].nunique()} products")
    with st.expander("Preview"):
        st.dataframe(full.head(10))

    limit = st.number_input(
        "Rows to process",
        min_value=5,
        max_value=len(full),
        value=min(300, len(full)),
        step=5,
        help=(
            "Steps 1 and 3 call the Gemini API per row/cluster; the free tier is "
            "rate-limited to roughly 15 requests/minute, so 300 rows will take "
            "noticeably longer than a quick test - budget several minutes, more "
            "if the model chain falls back under rate limiting. 30 rows is the "
            "smallest size verified end-to-end if you want a fast check first."
        ),
    )

else:
    uploaded = st.file_uploader("CSV with a 'narrative' column", type=["csv"])
    limit = None
    if uploaded is None:
        st.info("Upload a CSV to continue.")
        st.stop()

    df = pd.read_csv(uploaded)
    if settings.TEXT_COLUMN not in df.columns:
        st.error(
            f"No '{settings.TEXT_COLUMN}' column found. Columns present: "
            f"{', '.join(df.columns)}"
        )
        st.stop()

    st.success(f"{len(df):,} rows, column check passed")
    with st.expander("Preview"):
        st.dataframe(df.head(10))

    # A separate working file, not data/input.csv - overwriting the curated
    # 3,000-row dataset (with its matching ground_truth.csv) to run an
    # arbitrary upload is exactly the kind of silent, hard-to-undo action
    # this project's tools otherwise go out of their way to avoid.
    upload_dir = settings.ARTIFACTS / "streamlit_upload"
    upload_dir.mkdir(exist_ok=True)
    input_csv = upload_dir / "input.csv"
    if "complaint_id" not in df.columns:
        df.insert(0, "complaint_id", range(len(df)))
    df.to_csv(input_csv, index=False)

# --- run -----------------------------------------------------------------
st.markdown("---")
st.subheader("2. Run")

if st.button("Run pipeline", type="primary"):

    async def run_all() -> None:
        with st.status("Step 1/4 - extracting topics...", expanded=True) as s:
            r = await extract_topics(input_csv=input_csv, limit=limit)
            st.json(r)
            s.update(label="Step 1/4 - extract_topics done", state="complete")

        with st.status("Step 2/4 - clustering topics...", expanded=True) as s:
            r = cluster_topics()
            st.json(r)
            s.update(label="Step 2/4 - cluster_topics done", state="complete")

        with st.status("Step 3/4 - labelling clusters...", expanded=True) as s:
            r = await label_clusters()
            st.json(r)
            s.update(label="Step 3/4 - label_clusters done", state="complete")

        with st.status("Step 4/4 - generating charts...", expanded=True) as s:
            r = generate_insights()
            st.json(r)
            s.update(label="Step 4/4 - generate_insights done", state="complete")

    asyncio.run(run_all())
    st.rerun()  # re-read the freshly written files from disk, same path as an existing run

# --- results ---------------------------------------------------------------
# Reads whatever's on disk in artifacts/ right now - populated either by
# clicking "Run pipeline" above, or by a prior CLI run (python main.py,
# python -m tools.topic_clustering, ...). Not gated behind session state, so
# reopening this page shows existing results immediately, at zero cost.
st.markdown("---")
st.subheader("3. Results")

if settings.OUTPUT_CSV.exists():
    out = pd.read_csv(settings.OUTPUT_CSV)
    st.caption(f"From {settings.OUTPUT_CSV}, last written "
               f"{pd.Timestamp.fromtimestamp(settings.OUTPUT_CSV.stat().st_mtime):%Y-%m-%d %H:%M}")
    st.markdown(f"**{len(out):,} complaints, {out['general_topic_l1'].nunique()} themes**")

    plots = [
        ("Top pain points", settings.ARTIFACTS / "plot_top_pain_points.png"),
        ("Pain points by product", settings.ARTIFACTS / "heatmap_pain_points_by_product.png"),
        ("Theme severity by sentiment", settings.ARTIFACTS / "theme_severity_stacked.png"),
        ("Theme trends over time", settings.ARTIFACTS / "theme_trends_over_time.png"),
    ]
    cols = st.columns(2)
    for i, (title, path) in enumerate(plots):
        with cols[i % 2]:
            st.markdown(f"**{title}**")
            if path.exists():
                st.image(str(path), use_container_width=True)
            else:
                st.info("Not generated yet - run Step 4 (generate_insights).")

    st.markdown("---")
    st.subheader("Enriched output")
    st.dataframe(out.head(50))
    st.download_button(
        "Download output.csv",
        data=out.to_csv(index=False),
        file_name="output.csv",
        mime="text/csv",
    )
else:
    st.info("No output.csv yet - run the pipeline above, or via the CLI (`python main.py`), to generate it.")

# --- evaluation --------------------------------------------------------------
# Reads artifacts/df_with_clusters.csv, same "already on disk" logic as
# above - independent of the pipeline run button, and free to re-run (no API
# calls, just re-scoring an existing clustering against a chosen ground
# truth), so switching between Issue/product doesn't cost anything.
st.markdown("---")
st.subheader("4. Evaluation against real CFPB categories")
st.caption(
    "The clusters above were produced with no knowledge of the complaints' real "
    "category labels (clustering reads only `narrative` - see tools/topic_clustering.py). "
    "This scores that already-finished clustering against a real label, chosen below."
)

if not settings.CLUSTERS_CSV.exists():
    st.info("No clustered data yet - run at least through Step 2 above.")
else:
    from eval.choose_k import TRUTH_SOURCES

    truth_choice = st.radio(
        "Score against:",
        list(TRUTH_SOURCES.keys()),
        format_func=lambda t: {
            "issue": "Issue (fine-grained - 60-80 specific problem types)",
            "product": "product (coarse - 11 financial product categories)",
        }.get(t, t),
        horizontal=True,
    )

    if st.button("Run evaluation"):
        try:
            from eval.cluster_eval import evaluate_clusters

            ev = evaluate_clusters(truth=truth_choice)
            st.session_state["eval_result"] = ev
        except Exception as exc:  # noqa: BLE001
            st.error(f"Evaluation failed: {exc}")

    ev = st.session_state.get("eval_result")
    if ev and ev.get("truth") == truth_choice:
        n = ev["n_complaints_evaluated"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Adjusted Rand Index", ev["adjusted_rand_index"])
        c2.metric("Normalized Mutual Info", ev["normalized_mutual_info"])
        c3.metric("Mean cluster purity", ev["mean_cluster_purity"])
        st.caption(
            f"Scored on {n:,}/{ev['n_complaints_total']:,} complaints "
            f"({ev['n_categories_scored']}/{ev['n_categories_total']} categories - "
            f"{ev['n_categories_excluded_as_sparse']} excluded for having fewer than "
            f"{ev['min_category_support']} examples, which no unsupervised method could learn)"
        )

        if n < 200:
            st.warning(
                f"Only {n} complaints were evaluated. ARI/NMI are unreliable "
                "below a few hundred examples spread across dozens of true "
                "categories - treat these numbers as a mechanism check, not "
                "a result. Re-run with more rows for a trustworthy score."
            )
        if ev["junk_drawer_clusters"]:
            st.caption(
                f"Junk-drawer clusters (unusually diffuse - see breakdown below): "
                f"{ev['junk_drawer_clusters']}"
            )
        with st.expander("Per-cluster breakdown"):
            st.dataframe(pd.DataFrame(ev["cluster_breakdown"]).sort_values("purity"))
