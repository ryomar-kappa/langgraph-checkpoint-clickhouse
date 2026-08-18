from __future__ import annotations

import operator
from typing import Annotated
from uuid import uuid4

import pytest
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from langgraph.checkpoint.clickhouse import AsyncClickHouseSaver, ClickHouseSaver


class AccumulatingState(TypedDict):
    events: Annotated[list[str], operator.add]


def append_tick(state: AccumulatingState) -> dict[str, list[str]]:
    return {"events": ["tick"]}


def build_accumulating_graph(checkpointer):
    return (
        StateGraph(AccumulatingState)
        .add_node("tick", append_tick)
        .add_edge(START, "tick")
        .add_edge("tick", END)
        .compile(checkpointer=checkpointer)
    )


@pytest.mark.integration
def test_real_graph_sync_accumulates_and_time_travels(sync_saver: ClickHouseSaver) -> None:
    graph = build_accumulating_graph(sync_saver)
    config = {"configurable": {"thread_id": str(uuid4())}}

    assert graph.invoke({"events": ["first"]}, config)["events"] == ["first", "tick"]
    assert graph.invoke({"events": ["second"]}, config)["events"] == [
        "first",
        "tick",
        "second",
        "tick",
    ]
    history = list(graph.get_state_history(config))
    assert len(history) >= 4
    assert history[0].values["events"] == ["first", "tick", "second", "tick"]
    assert history[-1].values.get("events", []) == []


@pytest.mark.integration
async def test_real_graph_async_accumulates(async_saver: AsyncClickHouseSaver) -> None:
    graph = build_accumulating_graph(async_saver)
    config = {"configurable": {"thread_id": str(uuid4())}}

    first = await graph.ainvoke({"events": ["first"]}, config)
    second = await graph.ainvoke({"events": ["second"]}, config)
    assert first["events"] == ["first", "tick"]
    assert second["events"] == ["first", "tick", "second", "tick"]
    history = [state async for state in graph.aget_state_history(config)]
    assert history[0].values["events"] == second["events"]


class ApprovalState(TypedDict):
    events: Annotated[list[str], operator.add]


def ask_for_approval(state: ApprovalState) -> dict[str, list[str]]:
    answer = interrupt({"question": "approve?"})
    return {"events": [f"approved:{answer}"]}


@pytest.mark.integration
def test_interrupt_and_resume_is_persisted(sync_saver: ClickHouseSaver) -> None:
    graph = (
        StateGraph(ApprovalState)
        .add_node("approval", ask_for_approval)
        .add_edge(START, "approval")
        .add_edge("approval", END)
        .compile(checkpointer=sync_saver)
    )
    config = {"configurable": {"thread_id": str(uuid4())}}

    paused = graph.invoke({"events": ["started"]}, config)
    assert paused["__interrupt__"][0].value == {"question": "approve?"}

    resumed = graph.invoke(Command(resume="yes"), config)
    assert resumed["events"] == ["started", "approved:yes"]
