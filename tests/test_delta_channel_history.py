from __future__ import annotations

from typing import Any
from uuid import uuid4

import clickhouse_connect
import pytest
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import Checkpoint, get_checkpoint_id
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.types import _DeltaSnapshot

from langgraph.checkpoint.clickhouse import AsyncClickHouseSaver

from .conftest import unique_prefix


async def _put_step(
    saver: AsyncClickHouseSaver,
    *,
    thread_id: str,
    parent: RunnableConfig | None,
    step: int,
    values: dict[str, Any] | None = None,
    writes: list[tuple[str, Any]] | None = None,
) -> RunnableConfig:
    configurable: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": ""}
    if parent is not None:
        parent_id = get_checkpoint_id(parent)
        assert parent_id is not None
        configurable["checkpoint_id"] = parent_id
    config: RunnableConfig = {"configurable": configurable}
    channel_values = values or {}
    checkpoint = Checkpoint(
        v=1,
        id=str(uuid6(clock_seq=-1)),
        ts="",
        channel_values=channel_values,
        channel_versions={channel: step + 1 for channel in channel_values},
        versions_seen={},
        updated_channels=None,
    )
    stored = await saver.aput(
        config,
        checkpoint,
        {"source": "loop", "step": step},
        checkpoint["channel_versions"],
    )
    if writes:
        await saver.aput_writes(stored, writes, f"task-{step:03}")
    return stored


@pytest.mark.integration
async def test_delta_history_uses_nearest_snapshot_and_excludes_head(
    async_saver: AsyncClickHouseSaver,
) -> None:
    thread_id = str(uuid4())
    parent: RunnableConfig | None = None
    configs: list[RunnableConfig] = []
    for step in range(6):
        values = {"ch": _DeltaSnapshot(step)} if step in {0, 3} else {}
        writes = [] if step in {0, 3} else [("ch", step)]
        parent = await _put_step(
            async_saver,
            thread_id=thread_id,
            parent=parent,
            step=step,
            values=values,
            writes=writes,
        )
        configs.append(parent)

    history = await async_saver.aget_delta_channel_history(config=configs[-1], channels=["ch"])
    channel_history = history["ch"]
    assert "seed" in channel_history
    assert channel_history["seed"].value == 3
    assert [write[2] for write in history["ch"]["writes"]] == [4]
    assert 5 not in [write[2] for write in history["ch"]["writes"]]


@pytest.mark.integration
async def test_delta_history_channels_stop_independently(
    async_saver: AsyncClickHouseSaver,
) -> None:
    thread_id = str(uuid4())
    parent: RunnableConfig | None = None
    configs: list[RunnableConfig] = []
    for step in range(5):
        values: dict[str, Any] = {}
        if step == 1:
            values["a"] = _DeltaSnapshot("seed-a")
        if step == 3:
            values["b"] = _DeltaSnapshot("seed-b")
        parent = await _put_step(
            async_saver,
            thread_id=thread_id,
            parent=parent,
            step=step,
            values=values,
            writes=[("a", step), ("b", step)],
        )
        configs.append(parent)

    history = await async_saver.aget_delta_channel_history(config=configs[-1], channels=["a", "b"])
    assert "seed" in history["a"]
    assert "seed" in history["b"]
    assert history["a"]["seed"].value == "seed-a"
    assert history["b"]["seed"].value == "seed-b"
    assert [write[2] for write in history["a"]["writes"]] == [1, 2, 3]
    assert [write[2] for write in history["b"]["writes"]] == [3]


@pytest.mark.integration
async def test_delta_history_walks_to_root_without_seed(
    async_saver: AsyncClickHouseSaver,
) -> None:
    thread_id = str(uuid4())
    parent: RunnableConfig | None = None
    configs: list[RunnableConfig] = []
    for step in range(4):
        parent = await _put_step(
            async_saver,
            thread_id=thread_id,
            parent=parent,
            step=step,
            writes=[("ch", step)],
        )
        configs.append(parent)

    history = await async_saver.aget_delta_channel_history(config=configs[-1], channels=["ch"])
    assert "seed" not in history["ch"]
    assert [write[2] for write in history["ch"]["writes"]] == [0, 1, 2]


@pytest.mark.integration
async def test_delta_history_plain_migration_seed_includes_its_own_write(
    async_saver: AsyncClickHouseSaver,
) -> None:
    thread_id = str(uuid4())
    parent = await _put_step(
        async_saver,
        thread_id=thread_id,
        parent=None,
        step=0,
        writes=[("ch", "older-than-seed")],
    )
    parent = await _put_step(
        async_saver,
        thread_id=thread_id,
        parent=parent,
        step=1,
        values={"ch": [10, 20]},
        writes=[("ch", "at-seed")],
    )
    parent = await _put_step(
        async_saver,
        thread_id=thread_id,
        parent=parent,
        step=2,
        writes=[("ch", "after-seed")],
    )
    head = await _put_step(
        async_saver,
        thread_id=thread_id,
        parent=parent,
        step=3,
        writes=[("ch", "pending-at-head")],
    )

    history = await async_saver.aget_delta_channel_history(config=head, channels=["ch"])
    assert "seed" in history["ch"]
    assert history["ch"]["seed"] == [10, 20]
    assert [write[2] for write in history["ch"]["writes"]] == [
        "at-seed",
        "after-seed",
    ]


@pytest.mark.integration
async def test_legacy_delta_hint_preserves_plain_seed_during_prune(
    clickhouse_kwargs: dict[str, Any],
) -> None:
    client = await clickhouse_connect.get_async_client(**clickhouse_kwargs)
    saver = AsyncClickHouseSaver(
        client,
        table_prefix=unique_prefix("legacy_delta"),
        legacy_delta_channels=["ch"],
    )
    await saver.setup()
    try:
        thread_id = str(uuid4())
        parent = await _put_step(
            saver,
            thread_id=thread_id,
            parent=None,
            step=0,
            writes=[("ch", "older-than-seed")],
        )
        parent = await _put_step(
            saver,
            thread_id=thread_id,
            parent=parent,
            step=1,
            values={"ch": [10, 20]},
            writes=[("ch", "at-seed")],
        )
        parent = await _put_step(
            saver,
            thread_id=thread_id,
            parent=parent,
            step=2,
            writes=[("ch", "after-seed")],
        )
        head = await _put_step(
            saver,
            thread_id=thread_id,
            parent=parent,
            step=3,
            writes=[("ch", "pending-at-head")],
        )

        before = await saver.aget_delta_channel_history(config=head, channels=["ch"])
        await saver.aprune([thread_id], strategy="keep_latest")
        after = await saver.aget_delta_channel_history(config=head, channels=["ch"])

        assert after == before
        assert (
            len([row async for row in saver.alist({"configurable": {"thread_id": thread_id}})]) == 3
        )
    finally:
        await saver.drop_tables()
        await client.close()


@pytest.mark.integration
async def test_delta_history_empty_channel_request(
    async_saver: AsyncClickHouseSaver,
) -> None:
    head = await _put_step(
        async_saver,
        thread_id=str(uuid4()),
        parent=None,
        step=0,
        values={"ch": _DeltaSnapshot(1)},
    )
    assert await async_saver.aget_delta_channel_history(config=head, channels=[]) == {}
