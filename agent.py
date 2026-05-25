"""
agent.py  ·  LangGraph ReAct agent for intelligent document comparison.

The agent autonomously:
  1. Decides which tools to call and in what order
  2. Refines its queries based on intermediate observations
  3. Calls multiple tools to build a complete picture
  4. Synthesises a final answer with citations

Streaming support: yields typed step dicts for the UI to render live.
"""

from __future__ import annotations
import os
from typing import Iterator, Dict, Any
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from agent_tools import ALL_TOOLS, configure_tools
from vector_store import VectorStoreManager


SYSTEM_PROMPT = """You are an expert document analyst with access to two PDF documents.
Your job is to answer the user's comparative questions with precision, citations, and depth.

## Your tools
- search_document_a / search_document_b: targeted semantic search in one document
- compare_topic: efficient side-by-side search in both documents simultaneously
- find_conflicts: multi-angle search to surface contradictions
- find_common_ground: search for agreements and shared positions
- get_document_overview: broad sweep to understand a document's scope

## Reasoning strategy
1. For complex questions, START with get_document_overview on both docs to understand scope
2. Use compare_topic for direct comparisons — it's more efficient than searching each doc separately
3. If you find interesting content, use search_document_a / search_document_b with MORE SPECIFIC queries to dig deeper
4. Use find_conflicts ONLY when looking for disagreements/contradictions
5. Use find_common_ground ONLY when looking for agreements/similarities
6. Call tools MULTIPLE TIMES with different queries if needed — thoroughness matters

## Answer format
Structure your final answer as:

### 🔍 Key Differences
[bullet points with [Doc Name] citations]

### 🤝 Similarities / Common Ground
[bullet points]

### ⚠️ Conflicts / Contradictions
[if any — with page references]

### 📋 Summary
[2-3 sentence synthesis]

Always cite which document a finding comes from and the page number when possible.
If one document doesn't address a topic, explicitly state that."""


class DocumentComparisonAgent:
    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        self._llm = ChatMistralAI(
            model=model,
            mistral_api_key=api_key,
            temperature=0.1,
        )
        self._agent = create_react_agent(
            model=self._llm,
            tools=ALL_TOOLS,
            prompt=SYSTEM_PROMPT,
        )

    def configure(
        self,
        store: VectorStoreManager,
        name_a: str,
        name_b: str,
        top_k: int = 5,
    ):
        """Wire agent tools to the live vector store."""
        configure_tools(store, name_a, name_b, top_k)

    def stream_steps(self, query: str) -> Iterator[Dict[str, Any]]:
        """
        Stream agent execution steps as typed dicts:
          {"type": "thought",      "content": "..."}
          {"type": "tool_call",    "tool": "...", "input": "..."}
          {"type": "observation",  "tool": "...", "content": "..."}
          {"type": "final",        "content": "..."}
          {"type": "error",        "content": "..."}
        """
        try:
            for chunk in self._agent.stream(
                {"messages": [HumanMessage(content=query)]},
                stream_mode="values",
            ):
                messages = chunk.get("messages", [])
                if not messages:
                    continue

                last = messages[-1]

                # ── AI message (thought + tool calls) ────────────────────── #
                if isinstance(last, AIMessage):
                    # Emit reasoning text if present
                    if last.content and isinstance(last.content, str) and last.content.strip():
                        yield {"type": "thought", "content": last.content.strip()}

                    # Emit each tool call
                    for tc in (last.tool_calls or []):
                        tool_name = tc.get("name", "unknown_tool")
                        tool_input = tc.get("args", {})
                        # Get the first string arg as the display input
                        display_input = next(iter(tool_input.values()), str(tool_input))
                        yield {
                            "type": "tool_call",
                            "tool": tool_name,
                            "input": display_input,
                        }

                # ── Tool observation ──────────────────────────────────────── #
                elif isinstance(last, ToolMessage):
                    yield {
                        "type": "observation",
                        "tool": last.name or "tool",
                        "content": last.content[:800] + ("..." if len(last.content) > 800 else ""),
                    }

            # ── Final answer: last AI message with no tool calls ─────────── #
            final_state = self._agent.invoke(
                {"messages": [HumanMessage(content=query)]}
            )
            final_messages = final_state.get("messages", [])
            for msg in reversed(final_messages):
                if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                    yield {"type": "final", "content": msg.content}
                    break

        except Exception as e:
            yield {"type": "error", "content": str(e)}

    def invoke(self, query: str) -> str:
        """Non-streaming invoke — returns final answer string."""
        result = self._agent.invoke(
            {"messages": [HumanMessage(content=query)]}
        )
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage) and not msg.tool_calls and msg.content:
                return msg.content
        return "Agent did not produce a final answer."
