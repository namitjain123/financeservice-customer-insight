"""Alternative orchestration: one Agent, a system prompt, and the four tools -
the same shape as the original project's agents/SurveyInsightAgent.py
(AutoGen's AssistantAgent), including its choice to reach the tools over MCP
rather than importing them directly.

main.py's default path (four plain sequential function calls) is the one to
trust for anything programmatic. This file exists for a side-by-side
comparison with the original design, at real costs worth knowing:

  * MCP here is a straight subprocess-overhead cost, not a benefit. The
    tools are in this same codebase, this same process family - the original
    project's own commented-out line
    (`# tools=[TopicExtraction, TopicClustering, ClusterLabelling, businessInsight]`)
    shows a direct-import path was sitting right there and went unused. Kept
    this way here specifically to demonstrate/compare against that pattern,
    not because it's the recommended way to wire this agent - see
    MCP/server.py's docstring for when MCP actually earns its overhead
    (an external, non-Python, or cross-codebase caller - none of which this
    file is).
  * Step order is enforced by the LLM correctly following SYSTEM_PROMPT, not
    by code structure. Nothing stops it skipping a step, calling one twice,
    or running them out of order.
  * The agent's own narration is not the pipeline's status. Read the tool
    RESULTS in the response, not the prose around them - this is the exact
    shape of bug we found in the original project, where the agent announced
    "1,000 responses processed (for example)" against a 100-row file.
  * An extra LLM call happens just to decide "call tool 2 next," even though
    there is no actual decision being made - the order is always 1,2,3,4.
"""

from __future__ import annotations

import sys
from pathlib import Path

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools

from config import settings

MCP_SERVER_SCRIPT = Path(__file__).resolve().parent.parent / "MCP" / "server.py"

SYSTEM_PROMPT = """You run a 4-step pipeline that turns raw CFPB consumer complaints
into a labelled, analytics-ready dataset and four executive charts.

Call the four tools in this exact order, once each, waiting for each result
before calling the next:

1. extract_topics    - LLM extracts 2-5 topic phrases + sentiment per complaint
2. cluster_topics     - embeds topic phrases, groups them with KMeans
3. label_clusters     - LLM names each cluster, explodes to one row per topic
4. generate_insights  - draws the four charts from the labelled data

Each tool takes no arguments - it reads its input from the previous step's
output file and writes its own. After all four have run, report the exact
counts each tool returned (rows processed, clusters found, charts written).
Never state a number you did not receive from a tool result. Reply with
TERMINATE once you have reported those counts."""


async def _load_mcp_tools():
    """Spawn MCP/server.py as a subprocess and load its four tools over stdio.

    `command=sys.executable` (this venv's own interpreter), not a bare
    "python3" - the original project hardcoded "python3", which silently
    resolves to whatever's first on PATH rather than the venv the rest of
    this process is running in.
    """
    params = StdioServerParams(
        command=sys.executable,
        args=[str(MCP_SERVER_SCRIPT)],
        read_timeout_seconds=400,
    )
    return await mcp_server_tools(params)


async def build_agent() -> AssistantAgent:
    # Gemini via its OpenAI-compatible endpoint isn't one of AutoGen's known
    # model families, so model_info must be supplied explicitly or client
    # construction raises - "unknown" is honest about that rather than
    # claiming one of the literal gemini-* families this exact model isn't.
    model_client = OpenAIChatCompletionClient(
        model=settings.CHAT_MODEL,
        api_key=settings.require_api_key(),
        base_url=settings.BASE_URL,
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "structured_output": False,
            "family": "unknown",
        },
    )
    tools = await _load_mcp_tools()
    return AssistantAgent(
        name="SurveyInsightAgent",
        model_client=model_client,
        system_message=SYSTEM_PROMPT,
        tools=tools,
        max_tool_iterations=4,  
        reflect_on_tool_use=True,
    )


async def run_agent_pipeline() -> str:
    """Run the pipeline via agent tool-calling. Returns the agent's final text.

    Prefer main.py's plain sequential calls for anything you need to trust
    programmatically - this function's return value is narration, and
    narration is exactly what the original project got wrong.
    """
    agent = await build_agent()
    result = await agent.run(task="Run the pipeline.")
    final = result.messages[-1]
    return getattr(final, "content", str(final))
