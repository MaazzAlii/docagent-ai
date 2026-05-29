"""
agent.py  ·  LangGraph ReAct agent for intelligent document comparison.

FIXES applied:
  1. Removed double invoke() bug — final answer now captured during stream
  2. Master system prompt upgraded — adaptive format, no forced sections
  3. System prompt dynamically injects actual document names
"""

from __future__ import annotations
import os
from typing import Iterator, Dict, Any
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from agent_tools import ALL_TOOLS, configure_tools
from vector_store import VectorStoreManager


# ── Master System Prompt (upgraded) ──────────────────────────────────────────
SYSTEM_PROMPT_TEMPLATE = """You are an expert document analyst with access to two documents:
  - Document A: [{name_a}]
  - Document B: [{name_b}]

## OBJECTIVE
Answer the user's question by reasoning across both documents. Always cite which document
supports each claim using [{name_a}] or [{name_b}] with page numbers where available.

## TOOL STRATEGY
- ALWAYS start with get_document_overview on BOTH documents (doc_a then doc_b) before anything else
- Use compare_topic for any direct comparison question — it is your primary workhorse tool
- Use search_document_a / search_document_b to dig deeper into specific claims or sections
- Use find_conflicts ONLY when explicitly looking for contradictions or disagreements
- Use find_common_ground ONLY when looking for agreements or shared positions
- Call tools multiple times with refined queries if first results are insufficient
- Minimum 3 tool calls per question for thorough analysis

## CRITICAL RULES
- NEVER assume what documents contain — always use tools to verify first
- If a document does not address a topic, explicitly state: "[Doc X] does not mention this topic"
- Cite page numbers whenever available: e.g. [{name_a}, p.3]
- Adapt your analysis to the document type:
    * Resumes/CVs → compare skills, experience, education, projects
    * Research papers → compare methodology, findings, conclusions
    * Contracts/Legal → compare clauses, obligations, terms
    * Reports → compare recommendations, data, scope
- Do NOT force sections that are irrelevant to the document type

## ANSWER FORMAT
Use ONLY the sections that are genuinely relevant to what you found.

### 🔍 Key Differences
(only include if real differences exist between the documents)

### 🤝 Common Ground / Similarities
(only include if genuine agreements or overlaps exist)

### ⚠️ Conflicts / Contradictions
(only include if actual contradictions were found — do NOT fabricate conflicts)

### 📋 Summary
Always end with a 2-3 sentence synthesis of your findings.

IMPORTANT: If no meaningful differences or conflicts exist (e.g. two versions of the same
person's resume), say so directly and clearly explain what the documents actually contain
and how they relate. Never pad your answer with invented analysis."""


def build_system_prompt(name_a: str, name_b: str) -> str:
    """Inject actual document names into the system prompt."""
    return SYSTEM_PROMPT_TEMPLATE.format(
        name_a=name_a or "Document A",
        name_b=name_b or "Document B",
    )


class DocumentComparisonAgent:
    def __init__(self, api_key: str, model: str = "mistral-large-latest"):
        self._api_key = api_key
        self._model = model
        self._llm = ChatMistralAI(
            model=model,
            mistral_api_key=api_key,
            temperature=0.1,
        )
        self._name_a = "Document A"
        self._name_b = "Document B"
        self._agent = self._build_agent()

    def _build_agent(self):
        prompt = build_system_prompt(self._name_a, self._name_b)
        return create_react_agent(
            model=self._llm,
            tools=ALL_TOOLS,
            prompt=prompt,
        )

    def configure(
        self,
        store: VectorStoreManager,
        name_a: str,
        name_b: str,
        top_k: int = 5,
    ):
        """Wire agent tools to the live vector store and rebuild with correct doc names."""
        self._name_a = name_a or "Document A"
        self._name_b = name_b or "Document B"
        configure_tools(store, self._name_a, self._name_b, top_k)
        # Rebuild agent so system prompt reflects actual document names
        self._agent = self._build_agent()

    def stream_steps(self, query: str) -> Iterator[Dict[str, Any]]:
        """
        Stream agent execution steps as typed dicts.
        Types: thought | tool_call | observation | final | error

        FIX: Final answer is now captured DURING the stream.
              The old double invoke() has been removed.
        """
        final_answer = None

        try:
            for chunk in self._agent.stream(
                {"messages": [HumanMessage(content=query)]},
                stream_mode="values",
            ):
                messages = chunk.get("messages", [])
                if not messages:
                    continue

                last = messages[-1]

                # ── AI message: thought + tool calls ─────────────────────── #
                if isinstance(last, AIMessage):
                    # Emit reasoning text if present
                    if last.content and isinstance(last.content, str) and last.content.strip():
                        # If no tool calls → this is the final answer
                        if not last.tool_calls:
                            final_answer = last.content.strip()
                        else:
                            yield {"type": "thought", "content": last.content.strip()}

                    # Emit each tool call
                    for tc in (last.tool_calls or []):
                        tool_name = tc.get("name", "unknown_tool")
                        tool_input = tc.get("args", {})
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

            # Emit final answer captured from the stream (NO second invoke)
            if final_answer:
                yield {"type": "final", "content": final_answer}
            else:
                yield {"type": "error", "content": "Agent did not produce a final answer."}

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
