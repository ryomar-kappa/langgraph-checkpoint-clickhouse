from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import clickhouse_connect
from clickhouse_connect.driver.client import Client
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
from langgraph.checkpoint.clickhouse.aio import AsyncClickHouseSaver


class ClickHouseSaver(SaverCodec, BaseCheckpointSaver[int]):
    """Synchronous LangGraph checkpointer backed by ClickHouse.

    Call :meth:`setup` once before first use. Every read uses ``FINAL`` so that
    rows inserted as replacements are immediately observed without waiting for a
    background merge.
    """

    def __init__(
        self,
        client: Client,
        *,
        serde: SerializerProtocol | None = None,
        table_prefix: str = "langgraph_checkpoint",
        insert_settings: Mapping[str, Any] | None = None,
        legacy_delta_channels: Sequence[str] = (),
    ) -> None:
        super().__init__(serde=serde)
        prefix = validate_table_prefix(table_prefix)
        self.client = client
        self.checkpoints_table_name = f"{prefix}_checkpoints"
        self.writes_table_name = f"{prefix}_writes"
        self.checkpoints_table = quote_identifier(self.checkpoints_table_name)
        self.writes_table = quote_identifier(self.writes_table_name)
        self.insert_settings = prepare_insert_settings(insert_settings)
        self.legacy_delta_channels = frozenset(map(str, legacy_delta_channels))
        self._lock = threading.RLock()
        self._is_setup = False

    @classmethod
    @contextmanager
    def from_conn_string(
        cls,
        conn_string: str,
        *,
        serde: SerializerProtocol | None = None,
        table_prefix: str = "langgraph_checkpoint",
        insert_settings: Mapping[str, Any] | None = None,
        legacy_delta_channels: Sequence[str] = (),
        **client_kwargs: Any,
    ) -> Iterator[ClickHouseSaver]:
        client = clickhouse_connect.get_client(dsn=conn_string, **client_kwargs)
        try:
            yield cls(
                client,
                serde=serde,
                table_prefix=table_prefix,
                insert_settings=insert_settings,
                legacy_delta_channels=legacy_delta_channels,
            )
        finally:
            client.close()

    def setup(self) -> None:
        with self._lock:
            self.client.command(CHECKPOINT_DDL.format(table=self.checkpoints_table))
            self.client.command(WRITES_DDL.format(table=self.writes_table))
            self._validate_schema()
            self._is_setup = True

    def _validate_schema(self) -> None:
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
            rows = self.client.query(f"DESCRIBE TABLE {table}").result_rows
            info_rows = self.client.query(
                """
                SELECT engine, engine_full, sorting_key, primary_key,
                       partition_key, create_table_query
                FROM system.tables
                WHERE database = currentDatabase() AND name = %(table_name)s
                """,
                parameters={"table_name": table_name},
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

    def drop_tables(self) -> None:
        """Drop this saver's tables. Intended for tests and explicit teardown."""
        with self._lock:
            self.client.command(f"DROP TABLE IF EXISTS {self.checkpoints_table} SYNC")
            self.client.command(f"DROP TABLE IF EXISTS {self.writes_table} SYNC")
            self._is_setup = False

    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        del new_versions  # Full snapshots are deliberately stored atomically in one row.
        row = self.dump_checkpoint_row(config, checkpoint, metadata)
        with self._lock:
            self.client.insert(
                self.checkpoints_table,
                [row],
                column_names=CHECKPOINT_COLUMNS,
                settings=self.insert_settings,
            )
        thread_id, checkpoint_ns, _ = config_values(config)
        return checkpoint_config(thread_id, checkpoint_ns, checkpoint["id"])

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        if not writes:
            return
        rows = self.dump_write_rows(config, writes, task_id, task_path)
        with self._lock:
            self.client.insert(
                self.writes_table,
                rows,
                column_names=WRITE_COLUMNS,
                settings=self.insert_settings,
            )

    def get_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
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
        with self._lock:
            rows = self.client.query(
                query, parameters=params, column_formats=BLOB_QUERY_FORMATS
            ).result_rows
            if not rows:
                return None
            row = rows[0]
            pending = self._fetch_writes([(row[0], row[1], row[2])])[(row[0], row[1], row[2])]
        return self.load_checkpoint_tuple(row, pending)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:
        if limit is not None and limit <= 0:
            return
        where, params = self.checkpoint_search(config, before)
        query = CHECKPOINT_SELECT.format(table=self.checkpoints_table) + where
        query += " ORDER BY checkpoint_id DESC"
        if filter is None and limit is not None:
            query += " LIMIT %(limit)s"
            params["limit"] = int(limit)
        with self._lock:
            rows = self.client.query(
                query, parameters=params, column_formats=BLOB_QUERY_FORMATS
            ).result_rows
            decoded: list[tuple[Sequence[Any], CheckpointMetadata]] = []
            for row in rows:
                metadata = self.serde.loads_typed((row[6], bytes(row[7])))
                if filter and not self.metadata_matches(metadata, filter):
                    continue
                decoded.append((row, metadata))
                if limit is not None and len(decoded) >= limit:
                    break
            keys = [(row[0], row[1], row[2]) for row, _ in decoded]
            writes = self._fetch_writes(keys)
        for row, _ in decoded:
            key = (row[0], row[1], row[2])
            yield self.load_checkpoint_tuple(row, writes[key])

    def _fetch_writes(
        self, keys: Sequence[tuple[str, str, str]]
    ) -> dict[tuple[str, str, str], list[Sequence[Any]]]:
        grouped: dict[tuple[str, str, str], list[Sequence[Any]]] = {key: [] for key in keys}
        for predicate, params in self.key_delete_batches(keys):
            query = f"""
                SELECT thread_id, checkpoint_ns, checkpoint_id,
                       task_id, channel, value_type, value_blob
                FROM {self.writes_table} FINAL
                WHERE {predicate}
                ORDER BY thread_id, checkpoint_ns, checkpoint_id, task_id, idx
            """
            rows = self.client.query(
                query, parameters=params, column_formats=BLOB_QUERY_FORMATS
            ).result_rows
            for row in rows:
                grouped[(row[0], row[1], row[2])].append(row[3:])
        return grouped

    def delete_thread(self, thread_id: str) -> None:
        params = {"thread_id": str(thread_id)}
        with self._lock:
            self.client.command(
                f"DELETE FROM {self.checkpoints_table} WHERE thread_id = %(thread_id)s",
                parameters=params,
                settings=DELETE_SETTINGS,
            )
            self.client.command(
                f"DELETE FROM {self.writes_table} WHERE thread_id = %(thread_id)s",
                parameters=params,
                settings=DELETE_SETTINGS,
            )

    def delete_for_runs(self, run_ids: Sequence[str]) -> None:
        if not run_ids:
            return
        predicate, params = self.values_predicate("run_id", [str(run_id) for run_id in run_ids])
        with self._lock:
            rows = self.client.query(
                f"SELECT thread_id, checkpoint_ns, checkpoint_id "
                f"FROM {self.checkpoints_table} FINAL WHERE {predicate}",
                parameters=params,
            ).result_rows
            self._delete_keys([(row[0], row[1], row[2]) for row in rows])

    def _delete_keys(self, keys: Sequence[tuple[str, str, str]]) -> None:
        for predicate, params in self.key_delete_batches(keys):
            self.client.command(
                f"DELETE FROM {self.checkpoints_table} WHERE {predicate}",
                parameters=params,
                settings=DELETE_SETTINGS,
            )
            self.client.command(
                f"DELETE FROM {self.writes_table} WHERE {predicate}",
                parameters=params,
                settings=DELETE_SETTINGS,
            )

    def copy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        source, target = str(source_thread_id), str(target_thread_id)
        if source == target:
            return
        with self._lock:
            source_exists = self.client.query(
                f"SELECT 1 FROM {self.checkpoints_table} FINAL "
                "WHERE thread_id = %(source)s LIMIT 1",
                parameters={"source": source},
            ).result_rows
            if not source_exists:
                return
            self.delete_thread(target)
            self.client.command(
                f"""
                INSERT INTO {self.writes_table} ({", ".join(WRITE_COLUMNS)})
                SELECT %(target)s, checkpoint_ns, checkpoint_id, task_id, task_path,
                       idx, channel, value_type, value_blob
                FROM {self.writes_table} FINAL
                WHERE thread_id = %(source)s
                """,
                parameters={"source": source, "target": target},
            )
            self.client.command(
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

    def prune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        if strategy not in {"keep_latest", "delete"}:
            raise ValueError("strategy must be 'keep_latest' or 'delete'")
        ids = [str(thread_id) for thread_id in thread_ids]
        if not ids:
            return
        if strategy == "delete":
            for thread_id in ids:
                self.delete_thread(thread_id)
            return
        predicate, params = self.values_predicate("thread_id", ids)
        with self._lock:
            rows = self.client.query(
                CHECKPOINT_SELECT.format(table=self.checkpoints_table) + f" WHERE {predicate}",
                parameters=params,
                column_formats=BLOB_QUERY_FORMATS,
            ).result_rows
            retained = self.retained_keys_for_prune(rows)
            stale = [
                (row[0], row[1], row[2]) for row in rows if (row[0], row[1], row[2]) not in retained
            ]
            self._delete_keys(stale)

    def get_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        with self._lock:
            return super().get_delta_channel_history(config=config, channels=channels)

    # Async wrappers make the sync saver safe to use with LangGraph's async graph API.
    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:
        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:
        values = await asyncio.to_thread(
            lambda: list(self.list(config, filter=filter, before=before, limit=limit))
        )
        for value in values:
            yield value

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        return await asyncio.to_thread(self.put, config, checkpoint, metadata, new_versions)

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:
        await asyncio.to_thread(self.put_writes, config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await asyncio.to_thread(self.delete_thread, thread_id)

    async def adelete_for_runs(self, run_ids: Sequence[str]) -> None:
        await asyncio.to_thread(self.delete_for_runs, run_ids)

    async def acopy_thread(self, source_thread_id: str, target_thread_id: str) -> None:
        await asyncio.to_thread(self.copy_thread, source_thread_id, target_thread_id)

    async def aprune(self, thread_ids: Sequence[str], *, strategy: str = "keep_latest") -> None:
        await asyncio.to_thread(self.prune, thread_ids, strategy=strategy)

    async def aget_delta_channel_history(
        self, *, config: RunnableConfig, channels: Sequence[str]
    ) -> Mapping[str, DeltaChannelHistory]:
        return await asyncio.to_thread(
            self.get_delta_channel_history, config=config, channels=channels
        )


__all__ = ["AsyncClickHouseSaver", "ClickHouseSaver"]
