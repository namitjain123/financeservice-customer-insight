"""Alternative orchestration: one Agent, a system prompt, and the four tools -
the same shape as the original project's AutoGen AssistantAgent, ported to
Microsoft Agent Framework.

This is simpler to read than orchestration/pipeline.py's WorkflowBuilder graph,
at a real cost worth knowing before you point main.py at this instead:

  * Step order is enforced by the LLM correctly following SYSTEM_PROMPT, not
    by the graph structure. Nothing stops it skipping a step, calling one
    twice, or running them out of order.
  * The agent's own narration is not the pipeline's status. Read the tool
    RESULTS in the response, not the prose around them - this is the exact
    shape of bug we found in the original project, where the agent announced
    "1,000 responses processed (for example)" against a 100-row file.
  * An extra LLM call happens just to decide "call tool 2 next," even though
    there is no actual decision being made - the order is always 1,2,3,4.

Use this if you want the agent to read as closer to the original for a
side-by-side comparison. Use orchestration/pipeline.py if you want the
order structurally guaranteed.
"""

from __future__ import annotations

from agent_framework import Agent
from agent_framework.openai import OpenAIChatClient

from config import settings
from models.llm_client import get_async_client
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
Never state a number you did not receive from a tool result."""


def build_agent() -> Agent:
    client = OpenAIChatClient(model=settings.CHAT_MODEL, async_client=get_async_client())
    return Agent(
        client=client,
        name="SurveyInsightAgent",
        instructions=SYSTEM_PROMPT,
        tools=[extract_topics, cluster_topics, label_clusters, generate_insights],
    )


async def run_agent_pipeline() -> str:
    """Run the pipeline via agent tool-calling. Returns the agent's final text.

    Prefer orchestration.pipeline.run_pipeline() for anything you need to
    trust programmatically - this function's return value is narration, and
    narration is exactly what the original project got wrong.
    """
    async with build_agent() as agent:
        result = await agent.run("Run the pipeline.")
        return result.text
