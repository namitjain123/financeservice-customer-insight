"""MCP server exposing the four pipeline tools over stdio.

This is NOT how main.py runs the pipeline - main.py calls the tool functions
directly, in-process, which is strictly faster (no subprocess, no JSON-RPC
serialization) for a fixed sequence with one caller. This server exists for
the other reason to use MCP: any MCP-compatible client - Claude Desktop, VS
Code Copilot, another agent - can attach to it and call these tools without
writing custom integration code, without knowing this project's internals,
and without needing to import Python at all.

Run standalone:
    python MCP/server.py

Or drive it via agent_pipeline.py using MCPStdioTool instead of direct
function references - see orchestration/agent_pipeline.py for that wiring.
"""

from __future__ import annotations

import sys
from pathlib import Path

# So `tools.*` and `config.*` resolve when this file is launched directly
# (python MCP/server.py) rather than as a package via `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mcp.server.fastmcp import FastMCP

from tools.business_insight import generate_insights
from tools.cluster_labelling import label_clusters
from tools.topic_clustering import cluster_topics
from tools.topic_extraction import extract_topics

mcp = FastMCP(
    name="cfpb-complaint-insights",
    instructions=(
        "Four tools that turn raw CFPB consumer complaints into a labelled, "
        "analytics-ready dataset and four executive charts. Each tool reads "
        "its input from the previous tool's output file and takes no "
        "arguments - call them in order: extract_topics, cluster_topics, "
        "label_clusters, generate_insights."
    ),
)

# FastMCP builds each tool's schema from the function's own signature and
# docstring - the same docstrings tools/*.py already had, not rewritten here.
mcp.tool()(extract_topics)
mcp.tool()(cluster_topics)
mcp.tool()(label_clusters)
mcp.tool()(generate_insights)


if __name__ == "__main__":
    mcp.run(transport="stdio")
