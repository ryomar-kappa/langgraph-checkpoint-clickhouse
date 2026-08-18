from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

import clickhouse_connect
from clickhouse_connect.driver.asyncclient import AsyncClient
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    DeltaChannelHistory,
)
from langgraph.checkpoint.serde.base import SerializerProtocol

from langgraph.checkpoint.clickhouse._internal import (
    BLOB_QUERY_FORMATS,
    CHECKPOINT_COLUMN_TYPES,
    CHECKPOINT_COLUMNS,
    CHECKPOINT_DDL,
    CHECKPOINT_DEFAULTS,
    CHECKPOINT_SELECT,
    DELETE_SETTINGS,
    WRITE_COLUMN_TYPES,
    WRITE_COLUMNS,
    WRITE_DEFAULTS,
    WRITES_DDL,
    SaverCodec,
    checkpoint_config,
    config_values,
    prepare_insert_settings,
    quote_identifier,
    validate_schema,
    validate_table_prefix,
)


class _AsyncRLock:
    """Small task-reentrant lock used for nested saver operations."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._owner: asyncio.Task[Any] | None = None
        self._depth = 0

    async def acquire(self) -> None:
        task = asyncio.current_task()
        if task is None:  # pragma: no cover - a coroutine always has a task here
            raise RuntimeError("AsyncClickHouseSaver requires a running asyncio task")
        if self._owner is task:
            self._depth += 1
            return
        await self._lock.acquire()
        self._owner = task
        self._depth = 1

    def release(self) -> None:
        task = asyncio.current_task()
        if task is not self._owner:
            raise RuntimeError("async saver lock released by a non-owner task")
        self._depth -= 1
        if self._depth == 0:
            self._owner = None
            self._lock.release()


class AsyncClickHouseSaver(SaverCodec, BaseCheckpointSaver[int]):
    """Native-async LangGraph checkpointer backed by ClickHouse."""

    def __init__(
        self,
        client: AsyncClient,
        *,
        serde: SerializerProtocol | None = None,
        table_prefix: str = "langgraph_checkpoint",
        insert_settings: Mapping[str, Any] | None = None,
        legacy_delta_channels: Sequence[str] = (),
    ) -> None:
        super().__init__(serde=serde)
        prefix = validate_table_prefix(table_prefix)
        if (
            getattr(client, "_session_id_param", None)
            or getattr(client, "_autogenerate_session_id_param", False)
            or client.get_client_setting("session_id")
        ):
            raise ValueError(
                "AsyncClient session_id is incompatible with concurrent saver operations; "
                "create it with session_id=None and autogenerate_session_id=False"
            )
        self.client = client
        self.checkpoints_table_name = f"{prefix}_checkpoints"
        self.writes_table_name = f"{prefix}_writes"
        self.checkpoints_table = quote_identifier(self.checkpoints_table_name)
        self.writes_table = quote_identifier(self.writes_table_name)
        self.insert_settings = prepare_insert_settings(insert_settings)
        self.legacy_delta_channels = frozenset(map(str, legacy_delta_channels))
        self._locks = tuple(_AsyncRLock() for _ in range(64))
        self._is_setup = False

    @asynccontextmanager
    async def _locked(self, thread_ids: Sequence[str] | None) -> AsyncIterator[None]:
        indexes = (
            range(len(self._locks))
            if thread_ids is None
            else sorted({hash(thread_id) % len(self._locks) for thread_id in thread_ids})
        )
        locks = [self._locks[index] for index in indexes]
        acquired: list[_AsyncRLock] = []
        try:
            for lock in locks:
                await lock.acquire()
                acquired.append(lock)
            yield
        finally:
            for lock in reversed(acquired):
                lock.release()

    @classmethod
    @asynccontextmanager
    async def from_conn_string(
        cls,
        conn_string: str,
        *,
        serde: SerializerProtocol | None = None,
        table_prefix: str = "langgraph_checkpoint",
        insert_settings: Mapping[str, Any] | None = None,
        legacy_delta_channels: Sequence[str] = (),
        **client_kwargs: Any,
    ) -> AsyncIterator[AsyncClickHouseSaver]:
        client = await clickhouse_connect.get_async_client(dsn=conn_string, **client_kwargs)
        try:
            yield cls(
                client,
                serde=serde,
                table_prefix=table_prefix,
                insert_settings=insert_settings,
                legacy_delta_channels=legacy_delta_channels,
            )
        finally:
            await client.close()

    async def setup(self) -> None:
        async with self._locked(None):
            await self.client.command(CHECKPOINT_DDL.format(table=self.checkpoints_table))
            await self.client.command(WRITES_DDL.format(table=self.writes_table))
            await self._validate_schema()
            self._is_setup = True

    async def _validate_schema(self) -> None:
        specs = (
            (
                self.checkpoints_table,
                self.checkpoints_table_name,
                CHECKPOINT_COLUMN_TYPES,
                CHECKPOINT_DEFAULTS,
                "thread_id, checkpoint_ns, checkpoint_id",
                "thread_id, checkpoint_ns, checkpoint_id",
            ),
            (
                self.writes_table,
                self.writes_table_name,
                WRITE_COLUMN_TYPES,
                WRITE_DEFAULTS,
                "thread_id, checkpoint_ns, checkpoint_id, task_id, idx",
                "thread_id, checkpoint_ns, checkpoint_id",
            ),
        )
        for table, table_name, types, defaults, sorting_key, primary_key in specs:
            rows = (await self.client.query(f"DESCRIBE TABLE {table}")).result_rows
            info_rows = (
                await self.client.query(
                    """
                    SELECT engine, engine_full, sorting_key, primary_key,
                           partition_key, create_table_query
                    FROM system.tables
                    WHERE database = currentDatabase() AND name = %(table_name)s
                    """,
                    parameters={"table_name": table_name},
                )
            ).result_rows
            validate_schema(
                table=table,
                description_rows=rows,
                table_info=info_rows[0] if info_rows else None,
                expected_types=types,
                expected_defaults=defaults,
                expected_sorting_key=sorting_key,
                expected_primary_key=primary_key,
            )

    async def drop_tables(self) -> None:
        async with self._locked(None):
            await self.client.command(f"DROP TABLE IF EXISTS {self.checkpoints_table} SYNC")
            await self.client.command(f"DROP TABLE IF EXISTS {self.writes_table} SYNC")
            self._is_setup = False

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        del new_versions
        row = self.dump_checkpoint_row(config, checkpoint, metadata)
        thread_id, checkpoint_ns, _ = config_values(config)
        async with self._locked([thread_id]):
            await self.client.insert(
                self.checkpoints_table,
                [row],
                column_names=CHECKPOINT_COLUMNS,
                settings=self.insert_settings,
            )
        return checkpoint_config(thread_id, checkpoint_ns, checkpoint["id"])

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if not writes:
            return
        rows = self.dump_write_rows(config, writes, task_id, task_path)
        thread_id, _, _ = config_values(config)
        async with self._locked([thread_id]):
            await self.client.insert(
                self.writes_table,
                rows,
                column_names=WRITE_COLUMNS,
                settings=self.insert_settings,
            )

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        thread_id, checkpoint_ns, checkpoint_id = config_values(config)
        predicates = ["thread_id = %(thread_id)s", "checkpoint_ns = %(checkpoint_ns)s"]
        params: dict[str, Any] = {"thread_id": thread_id, "checkpoint_ns": checkpoint_ns}
        if checkpoint_id is not None:
            predicates.append("checkpoint_id = %(checkpoint_id)s")
            params["checkpoint_id"] = checkpoint_id
        query = (
            CHECKPOINT_SELECT.format(table=self.checkpoints_table)
            + " WHERE "
            + " AND ".join(predicates)
            + " ORDER BY checkpoint_id DESC LIMIT 1"
        )
        async with self._locked([thread_id]):
            rows = (
                await self.client.query(query, parameters=params, column_formats=BLOB_QUERY_FORMATS)
            ).result_rows
            if not rows:
                return None
            row = rows[0]
            pending = (await self._fetch_writes([(row[0], row[1], row[2])]))[
                (row[0], row[1], row[2])
            ]
            return self.load_checkpoint_tuple(row, pending)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        if limit is not None and limit <= 0:
            return
        where, params = self.checkpoint_search(config, before)
        query = CHECKPOINT_SELECT.format(table=self.checkpoints_table) + where
        query += " ORDER BY checkpoint_id DESC"
        if filter is None and limit is not None:
            query += " LIMIT %(limit)s"
            params["limit"] = int(limit)
        thread_ids = [config_values(config)[0]] if config is not None else None
        async with self._locked(thread_ids):
            rows = (
                await self.client.query(query, parameters=params, column_formats=BLOB_QUERY_FORMATS)
            ).result_rows
            decoded: list[Sequence[Any]] = []
            for row in rows:
                metadata = self.serde.loads_typed((row[6], bytes(row[7])))
                if filter and not self.metadata_matches(metadata, filter):
                    continue
                decoded.append(row)
                if limit is not None and len(decoded) >= limit:
                    break
            keys = [(row[0], row[1], row[2]) for row in decoded]
            writes = await self._fetch_writes(keys)
            values = [
                self.load_checkpoint_tuple(row, writes[(row[0], row[1], row[2])]) for row in decoded
            ]
        for value in values:
            yield value

    async def _fetch_writes(
        self, keys: Sequence[tuple[str, str, str]]
    ) -> dict[tuple[str, str, str], list[Sequence[Any]]]:
        grouped: dict[tuple[str, str, str], list[Sequence[Any]]] = {key: [] for key in keys}
        for predicate, params in self.key_delete_batches(keys):
            rows = (
                await self.client.query(
                    f"""
                    SELECT thread_id, checkpoint_ns, checkpoint_id,
                           task_id, channel, value_type, value_blob
                    FROM {self.writes_table} FINAL
                    WHERE {predicate}
                    ORDER BY thread_id, checkpoint_ns, checkpoint_id, task_id, idx
                    """,
                    parameters=params,
                    column_formats=BLOB_QUERY_FORMATS,
                )
            ).result_rows
            for row in rows:
                grouped[(row[0], row[1], row[2])].append(row[3:])
        return grouped

    async def adelete_thread(self, thread_id: str) -> None:
        thread_id = str(thread_id)
        params = {"thread_id": thread_id}
        async with self._locked([thread_id]):
            await self.client.command(
                f"DELETE FROM {self.checkpoints_table} WHERE thread_id = %(thread_id)s",
                parameters=params,
                settings=DELETE_SETTINGS,
            )
            await self.client.command(
                f"DELETE FROM {self.writes_table} WHERE thread_id = %(thread_id)s",
                parameters=params,
                settings=DELETE_SETTINGS,
            )

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        if not run_ids:
            return
        async with self._locked(None):
            predicate, params = self.values_predicate("run_id", [str(run_id) for run_id in run_ids])
            rows = (
                await self.client.query(
                    f"SELECT thread_id, checkpoint_ns, checkpoint_id "
                    f"FROM {self.checkpoints_table} FINAL WHERE {predicate}",
                    parameters=params,
                )
            ).result_rows
            await self._delete_keys([(row[0], row[1], row[2]) for row in rows])

    async def _delete_keys(self, keys: Sequence[tuple[str, str, str]]) -> None:
        for predicate, params in self.key_delete_batches(keys):
            await self.client.command(
                f"DELETE FROM {self.checkpoints_table} WHERE {predicate}",
                parameters=params,
                settings=DELETE_SETTINGS,
            )
            await self.client.command(
                f"DELETE FROM {self.writes_table} WHERE {predicate}",
                parameters=params,
                settings=DELETE_SETTINGS,
            )

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        source, target = str(source_thread_id), str(target_thread_id)
        if source == target:
            return
        async with self._locked([source, target]):
            source_exists = (
                await self.client.query(
                    f"SELECT 1 FROM {self.checkpoints_table} FINAL "
                    "WHERE thread_id = %(source)s LIMIT 1",
                    parameters={"source": source},
                )
            ).result_rows
            if not source_exists:
                return
            await self.adelete_thread(target)
            await self.client.command(
                f"""
                INSERT INTO {self.writes_table} ({", ".join(WRITE_COLUMNS)})
                SELECT %(target)s, checkpoint_ns, checkpoint_id, task_id, task_path,
                       idx, channel, value_type, value_blob
                FROM {self.writes_table} FINAL
                WHERE thread_id = %(source)s
                """,
                parameters={"source": source, "target": target},
            )
            await self.client.command(
                f"""
                INSERT INTO {self.checkpoints_table} ({", ".join(CHECKPOINT_COLUMNS)})
                SELECT %(target)s, checkpoint_ns, checkpoint_id, parent_checkpoint_id,
                       checkpoint_type, checkpoint_blob, metadata_type, metadata_blob,
                       run_id
                FROM {self.checkpoints_table} FINAL
                WHERE thread_id = %(source)s
                """,
                parameters={"source": source, "target": target},
            )

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        if strategy not in {"keep_latest", "delete"}:
            raise ValueError("strategy must be 'keep_latest' or 'delete'")
        ids = [str(thread_id) for thread_id in thread_ids]
        if not ids:
            return
        async with self._locked(ids):
            if strategy == "delete":
                for thread_id in ids:
                    await self.adelete_thread(thread_id)
                return
            predicate, params = self.values_predicate("thread_id", ids)
            rows = (
                await self.client.query(
                    CHECKPOINT_SELECT.format(table=self.checkpoints_table) + f" WHERE {predicate}",
                    parameters=params,
                    column_formats=BLOB_QUERY_FORMATS,
                )
            ).result_rows
            retained = self.retained_keys_for_prune(rows)
            stale = [
                (row[0], row[1], row[2]) for row in rows if (row[0], row[1], row[2]) not in retained
            ]
            await self._delete_keys(stale)

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        thread_id, _, _ = config_values(config)
        async with self._locked([thread_id]):
            return await super().aget_delta_channel_history(config=config, channels=channels)


__all__ = ["AsyncClickHouseSaver"]
