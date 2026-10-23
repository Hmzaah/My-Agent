"""Wires the agent steps into a LangGraph state machine.

plan -> retrieve -> grade --sufficient--------------------------> generate
                      |                                               ^
                      +--insufficient, rewrites left--> rewrite -> retrieve
                      +--insufficient, web allowed----> web_search -> grade
                      +--otherwise-------------------------------------+
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph

from jolt.agent.nodes import AgentNodes
from jolt.agent.state import AgentState


def build_graph(nodes: AgentNodes) -> Any:
    max_rewrites = nodes.settings.max_rewrites

    def route_after_grade(state: AgentState) -> str:
        if state.get("sufficient"):
            return "generate"
        used_web = state.get("used_web", False)
        index_empty = len(nodes.retriever.store) == 0
        if not used_web and not index_empty and state.get("rewrites", 0) < max_rewrites:
            return "rewrite"
        if state.get("web_allowed", False) and not used_web:
            return "web_search"
        return "generate"

    graph = StateGraph(AgentState)
    graph.add_node("plan", nodes.plan)
    graph.add_node("retrieve", nodes.retrieve)
    graph.add_node("grade", nodes.grade)
    graph.add_node("rewrite", nodes.rewrite)
    graph.add_node("web_search", nodes.web_search)
    graph.add_node("generate", nodes.generate)

    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges(
        "grade",
        route_after_grade,
        {"generate": "generate", "rewrite": "rewrite", "web_search": "web_search"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("web_search", "grade")
    graph.add_edge("generate", END)
    return graph.compile()
