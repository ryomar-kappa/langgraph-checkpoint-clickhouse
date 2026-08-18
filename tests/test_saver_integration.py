from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import clickhouse_connect
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import Checkpoint, empty_checkpoint
from langgraph.checkpoint.base.id import uuid6
from langgraph.checkpoint.serde.types import ERROR, _DeltaSnapshot

from langgraph.checkpoint.clickhouse import AsyncClickHouseSaver, ClickHouseSaver
from langgraph.checkpoint.clickhouse._internal import WRITES_DDL, quote_identifier

from .conftest import unique_prefix


def _config(
    thread_id: str | None = None,
    *,
    checkpoint_ns: str = "",
    checkpoint_id: str | None = None,
) -> dict[str, Any]:
    configurable: dict[str, Any] = {
        "thread_id": thread_id or str(uuid4()),
        "checkpoint_ns": checkpoint_ns,
    }
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def _checkpoint(values: dict[str, Any] | None = None) -> Checkpoint:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = values or {}
    checkpoint["channel_versions"] = {key: 1 for key in checkpoint["channel_values"]}
    return checkpoint


@pytest.mark.integration
def test_sync_roundtrip_list_none_and_typed_values(sync_saver: ClickHouseSaver) -> None:
    config = _config()
    value = {
        "bytes": b"\x00\xff\x10",
        "when": datetime(2026, 8, 18, tzinfo=timezone.utc),
        "message": HumanMessage(content="ClickHouse round-trip"),
    }
    checkpoint = _checkpoint(value)
    stored = sync_saver.put(
        {**config, "metadata": {"tenant": "alpha"}},
        checkpoint,
        {"source": "input", "step": -1, "nested": {"enabled": True}},
        checkpoint["channel_versions"],
    )

    loaded = sync_saver.get_tuple(stored)
    assert loaded is not None
    assert loaded.checkpoint["channel_values"] == value
    assert loaded.metadata["tenant"] == "alpha"

    all_rows = list(sync_saver.list(None, filter={"nested": {"enabled": True}}))
    assert [row.checkpoint["id"] for row in all_rows] == [checkpoint["id"]]


@pytest.mark.integration
def test_pending_write_retry_semantics(sync_saver: ClickHouseSaver) -> None:
    stored = sync_saver.put(_config(), _checkpoint(), {"source": "loop", "step": 0}, {})
    task_id = str(uuid4())

    sync_saver.put_writes(stored, [("regular", "first")], task_id)
    sync_saver.put_writes(stored, [("regular", "second")], task_id)
    loaded = sync_saver.get_tuple(stored)
    assert loaded is not None
    assert loaded.pending_writes == [(task_id, "regular", "first")]

    sync_saver.put_writes(stored, [(ERROR, "first error")], task_id)
    sync_saver.put_writes(stored, [(ERROR, "latest error")], task_id)
    loaded = sync_saver.get_tuple(stored)
    assert loaded is not None
    errors = [write for write in loaded.pending_writes or [] if write[1] == ERROR]
    assert errors == [(task_id, ERROR, "latest error")]

    sync_saver.put_writes(stored, [(ERROR, "mixed error"), ("second", 1)], task_id)
    loaded = sync_saver.get_tuple(stored)
    assert loaded is not None
    assert (task_id, ERROR, "mixed error") in (loaded.pending_writes or [])
    assert (task_id, "second", 1) in (loaded.pending_writes or [])


@pytest.mark.integration
def test_copy_from_missing_source_preserves_target(sync_saver: ClickHouseSaver) -> None:
    target_thread = str(uuid4())
    stored = sync_saver.put(
        _config(target_thread),
        _checkpoint({"value": "valuable-target"}),
        {"source": "input", "step": -1},
        {"value": 1},
    )

    sync_saver.copy_thread("missing-source", target_thread)

    loaded = sync_saver.get_tuple(stored)
    assert loaded is not None
    assert loaded.checkpoint["channel_values"] == {"value": "valuable-target"}


@pytest.mark.integration
async def test_concurrent_async_pending_writes(
    async_saver: AsyncClickHouseSaver,
) -> None:
    stored = await async_saver.aput(_config(), _checkpoint(), {"source": "loop", "step": 0}, {})
    await asyncio.gather(
        *(
            async_saver.aput_writes(stored, [("results", index)], f"task-{index:03}")
            for index in range(32)
        )
    )
    loaded = await async_saver.aget_tuple(stored)
    assert loaded is not None
    assert sorted(write[2] for write in loaded.pending_writes or []) == list(range(32))


@pytest.mark.integration
async def test_retry_order_is_shared_across_saver_instances(
    clickhouse_kwargs: dict[str, Any],
) -> None:
    prefix = unique_prefix("multi_instance")
    first_client = await clickhouse_connect.get_async_client(**clickhouse_kwargs)
    second_client = await clickhouse_connect.get_async_client(**clickhouse_kwargs)
    first = AsyncClickHouseSaver(first_client, table_prefix=prefix)
    second = AsyncClickHouseSaver(second_client, table_prefix=prefix)
    await first.setup()
    try:
        await second.setup()
        stored = await first.aput(_config(), _checkpoint(), {"source": "loop", "step": 0}, {})
        task_id = str(uuid4())
        await first.aput_writes(stored, [("regular", "first"), (ERROR, "old-error")], task_id)
        await second.aput_writes(stored, [("regular", "second"), (ERROR, "new-error")], task_id)

        loaded = await second.aget_tuple(stored)
        assert loaded is not None
        assert loaded.pending_writes == [
            (task_id, ERROR, "new-error"),
            (task_id, "regular", "first"),
        ]
    finally:
        await first.drop_tables()
        await first_client.close()
        await second_client.close()


@pytest.mark.integration
async def test_same_thread_delete_and_put_are_serialized(
    async_saver: AsyncClickHouseSaver,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    thread_id = str(uuid4())
    await async_saver.aput(
        _config(thread_id), _checkpoint({"value": "old"}), {"source": "input", "step": -1}, {}
    )
    original_command = async_saver.client.command
    checkpoint_deleted = asyncio.Event()
    release_delete = asyncio.Event()

    async def observed_command(command: str, *args: Any, **kwargs: Any) -> Any:
        result = await original_command(command, *args, **kwargs)
        if command.startswith("DELETE FROM") and async_saver.checkpoints_table in command:
            checkpoint_deleted.set()
            await release_delete.wait()
        return result

    monkeypatch.setattr(async_saver.client, "command", observed_command)
    delete_task = asyncio.create_task(async_saver.adelete_thread(thread_id))
    await checkpoint_deleted.wait()
    put_task = asyncio.create_task(
        async_saver.aput(
            _config(thread_id),
            _checkpoint({"value": "new"}),
            {"source": "input", "step": -1},
            {},
        )
    )
    await asyncio.sleep(0.02)
    was_serialized = not put_task.done()
    release_delete.set()
    stored = (await asyncio.gather(delete_task, put_task))[1]

    assert was_serialized
    loaded = await async_saver.aget_tuple(stored)
    assert loaded is not None
    assert loaded.checkpoint["channel_values"] == {"value": "new"}


@pytest.mark.integration
@pytest.mark.parametrize("invalid_schema", ["engine", "sorting_key", "partition_key"])
def test_setup_rejects_incompatible_existing_schema(
    clickhouse_kwargs: dict[str, Any], invalid_schema: str
) -> None:
    prefix = unique_prefix("bad_schema")
    client = clickhouse_connect.get_client(**clickhouse_kwargs)
    saver = ClickHouseSaver(client, table_prefix=prefix)
    table = quote_identifier(f"{prefix}_writes")
    ddl = WRITES_DDL.format(table=table)
    if invalid_schema == "engine":
        ddl = ddl.replace("ENGINE = ReplacingMergeTree(revision)", "ENGINE = MergeTree")
    else:
        if invalid_schema == "sorting_key":
            ddl = ddl.replace(
                "ORDER BY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)",
                "ORDER BY (thread_id, checkpoint_ns, checkpoint_id)",
            )
        else:
            ddl = ddl.replace(
                "PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)",
                "PARTITION BY checkpoint_ns\nPRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)",
            )
    try:
        client.command(ddl)
        with pytest.raises(RuntimeError, match="Incompatible ClickHouse table"):
            saver.setup()
    finally:
        saver.drop_tables()
        client.close()


def test_async_insert_without_acknowledgement_is_rejected() -> None:
    with pytest.raises(ValueError, match="wait_for_async_insert=0"):
        ClickHouseSaver(
            object(),  # type: ignore[arg-type]
            insert_settings={"async_insert": 1, "wait_for_async_insert": 0},
        )


@pytest.mark.integration
async def test_async_client_session_id_is_rejected(
    clickhouse_kwargs: dict[str, Any],
) -> None:
    for options in (
        {"session_id": f"lgcp-test-{uuid4()}"},
        {"settings": {"session_id": f"lgcp-test-{uuid4()}"}},
    ):
        client = await clickhouse_connect.get_async_client(**clickhouse_kwargs, **options)
        try:
            with pytest.raises(ValueError, match="session_id"):
                AsyncClickHouseSaver(client)
        finally:
            await client.close()


@pytest.mark.integration
async def test_delta_channel_history_uses_parent_chain(
    async_saver: AsyncClickHouseSaver,
) -> None:
    thread_id = str(uuid4())
    parent: dict[str, Any] | None = None
    configs: list[dict[str, Any]] = []
    for step in range(4):
        config = _config(thread_id)
        if parent is not None:
            config["configurable"]["checkpoint_id"] = parent["configurable"]["checkpoint_id"]
        channel_values = {"delta": _DeltaSnapshot(10)} if step == 0 else {}
        checkpoint = Checkpoint(
            v=1,
            id=str(uuid6(clock_seq=-1)),
            ts="",
            channel_values=channel_values,
            channel_versions={"delta": step + 1},
            versions_seen={},
            updated_channels=None,
        )
        parent = await async_saver.aput(
            config,
            checkpoint,
            {"source": "loop", "step": step},
            checkpoint["channel_versions"],
        )
        configs.append(parent)
        await async_saver.aput_writes(parent, [("delta", step + 1)], f"task-{step}")

    history = await async_saver.aget_delta_channel_history(config=configs[-1], channels=["delta"])
    assert history["delta"]["seed"].value == 10
    assert [write[2] for write in history["delta"]["writes"]] == [1, 2, 3]

    await async_saver.aprune([thread_id], strategy="keep_latest")
    history_after_prune = await async_saver.aget_delta_channel_history(
        config=configs[-1], channels=["delta"]
    )
    assert history_after_prune == history
    assert len([item async for item in async_saver.alist(_config(thread_id))]) == 4


@pytest.mark.integration
async def test_sync_async_parity_and_recreated_client_persistence(
    clickhouse_kwargs: dict[str, Any],
) -> None:
    prefix = unique_prefix("persistence")
    sync_client = clickhouse_connect.get_client(**clickhouse_kwargs)
    sync_saver = ClickHouseSaver(sync_client, table_prefix=prefix)
    sync_saver.setup()
    config = _config()
    checkpoint = _checkpoint({"state": "durable"})
    stored = sync_saver.put(config, checkpoint, {"source": "input", "step": -1}, {"state": 1})
    sync_client.close()

    async_client = await clickhouse_connect.get_async_client(**clickhouse_kwargs)
    async_saver = AsyncClickHouseSaver(async_client, table_prefix=prefix)
    try:
        await async_saver.setup()
        loaded = await async_saver.aget_tuple(stored)
        assert loaded is not None
        assert loaded.checkpoint["channel_values"] == {"state": "durable"}
    finally:
        await async_saver.drop_tables()
        await async_client.close()
