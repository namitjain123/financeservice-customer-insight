"""Alternative orchestration: one Agent, a system prompt, and the four tools -
the same shape as the original project's agents/SurveyInsightAgent.py
(AutoGen's AssistantAgent), pointed at this project's own tools instead of
the MCP tool wrappers the original used.

main.py's default path (four plain sequential function calls) is the one to
trust for anything programmatic. This file exists for a side-by-side
comparison with the original design, at a real cost worth knowing:

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

from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

from config import settings
from tools.business_insight import generate_insights
from tools.cluster_labelling import label_clusters
from tools.topic_clustering import cluster_topics
from tools.topic_extraction import extract_topics

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


async def _extract_topics() -> dict:
    """Step 1: LLM extracts 2-5 topic phrases + sentiment per complaint."""
    return await extract_topics()


def _cluster_topics() -> dict:
    """Step 2: embeds topic phrases, groups them with KMeans."""
    return cluster_topics()


async def _label_clusters() -> dict:
    """Step 3: LLM names each cluster, explodes to one row per topic."""
    return await label_clusters()


def _generate_insights() -> dict:
    """Step 4: draws the four charts from the labelled data."""
    return generate_insights()


def build_agent() -> AssistantAgent:
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
    return AssistantAgent(
        name="SurveyInsightAgent",
        model_client=model_client,
        system_message=SYSTEM_PROMPT,
        # Wrapped zero-arg: the real tool functions accept optional path
        # overrides with untyped defaults (fine for main.py's direct calls),
        # but AutoGen's FunctionTool needs full annotations on every param to
        # build a schema, and these tools take no arguments in this pipeline
        # anyway - each reads the previous step's output file by convention.
        tools=[_extract_topics, _cluster_topics, _label_clusters, _generate_insights],
        max_tool_iterations=4,  # one call per step; nothing here needs a 5th
        reflect_on_tool_use=True,
    )


async def run_agent_pipeline() -> str:
    """Run the pipeline via agent tool-calling. Returns the agent's final text.

    Prefer main.py's plain sequential calls for anything you need to trust
    programmatically - this function's return value is narration, and
    narration is exactly what the original project got wrong.
    """
    agent = build_agent()
    result = await agent.run(task="Run the pipeline.")
    final = result.messages[-1]
    return getattr(final, "content", str(final))
