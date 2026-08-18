from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
    get_checkpoint_id,
    get_checkpoint_metadata,
)
from langgraph.checkpoint.serde.base import SerializerProtocol
from langgraph.checkpoint.serde.types import _DeltaSnapshot

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_MAX_DELETE_KEYS = 200

CHECKPOINT_COLUMNS = (
    "thread_id",
    "checkpoint_ns",
    "checkpoint_id",
    "parent_checkpoint_id",
    "checkpoint_type",
    "checkpoint_blob",
    "metadata_type",
    "metadata_blob",
    "run_id",
)

WRITE_COLUMNS = (
    "thread_id",
    "checkpoint_ns",
    "checkpoint_id",
    "task_id",
    "task_path",
    "idx",
    "channel",
    "value_type",
    "value_blob",
)

CHECKPOINT_COLUMN_TYPES = {
    "thread_id": "String",
    "checkpoint_ns": "String",
    "checkpoint_id": "String",
    "parent_checkpoint_id": "String",
    "checkpoint_type": "LowCardinality(String)",
    "checkpoint_blob": "String",
    "metadata_type": "LowCardinality(String)",
    "metadata_blob": "String",
    "run_id": "String",
    "revision": "UInt128",
}

WRITE_COLUMN_TYPES = {
    "thread_id": "String",
    "checkpoint_ns": "String",
    "checkpoint_id": "String",
    "task_id": "String",
    "task_path": "String",
    "idx": "Int32",
    "channel": "String",
    "value_type": "LowCardinality(String)",
    "value_blob": "String",
    "revision": "UInt128",
}

CHECKPOINT_DEFAULTS = {
    "parent_checkpoint_id": "''",
    "run_id": "''",
    "revision": "toUInt128(generateUUIDv7())",
}

WRITE_DEFAULTS = {
    "task_path": "''",
    "revision": ("if(idx<0,toUInt128(generateUUIDv7()),bitNot(toUInt128(generateUUIDv7())))"),
}


def validate_table_prefix(prefix: str) -> str:
    if not _IDENTIFIER.fullmatch(prefix):
        raise ValueError(
            "table_prefix must start with a letter or underscore and contain only "
            "ASCII letters, digits, and underscores"
        )
    if len(prefix) > 160:
        raise ValueError("table_prefix must be at most 160 characters")
    return prefix


def quote_identifier(identifier: str) -> str:
    """Quote an identifier that has already passed ``validate_table_prefix``."""
    return f"`{identifier}`"


def prepare_insert_settings(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return settings that always preserve read-after-write semantics."""
    prepared = dict(DEFAULT_INSERT_SETTINGS)
    if settings is not None:
        prepared.update(settings)
    if _setting_enabled(prepared.get("async_insert", 0)):
        if "wait_for_async_insert" in prepared and not _setting_enabled(
            prepared["wait_for_async_insert"]
        ):
            raise ValueError("wait_for_async_insert=0 is unsafe for a checkpoint saver")
        prepared["wait_for_async_insert"] = 1
    return prepared


def validate_schema(
    *,
    table: str,
    description_rows: Sequence[Sequence[Any]],
    table_info: Sequence[Any] | None,
    expected_types: Mapping[str, str],
    expected_defaults: Mapping[str, str],
    expected_sorting_key: str,
    expected_primary_key: str,
) -> None:
    """Reject pre-existing tables that would silently corrupt saver semantics."""
    problems: list[str] = []
    described = {str(row[0]): row for row in description_rows}
    expected_columns = set(expected_types)
    actual_columns = set(described)
    if actual_columns != expected_columns:
        missing = sorted(expected_columns - actual_columns)
        extra = sorted(actual_columns - expected_columns)
        if missing:
            problems.append(f"missing columns {missing}")
        if extra:
            problems.append(f"unexpected columns {extra}")

    for column, expected_type in expected_types.items():
        row = described.get(column)
        if row is not None and row[1] != expected_type:
            problems.append(f"{column} has type {row[1]!r}, expected {expected_type!r}")

    for column, expected_expression in expected_defaults.items():
        row = described.get(column)
        if row is None:
            continue
        if row[2] != "DEFAULT" or _compact_sql(str(row[3])) != _compact_sql(expected_expression):
            problems.append(
                f"{column} has default ({row[2]!r}, {row[3]!r}), "
                f"expected DEFAULT {expected_expression}"
            )
    ttl_columns = sorted(column for column, row in described.items() if len(row) > 6 and row[6])
    if ttl_columns:
        problems.append(f"column TTL is not supported on {ttl_columns}")

    if table_info is None:
        problems.append("table metadata is missing from system.tables")
    else:
        engine, engine_full, sorting_key, primary_key, partition_key, create_query = map(
            str, table_info
        )
        compatible_engines = {
            "ReplacingMergeTree",
            "ReplicatedReplacingMergeTree",
            "SharedReplacingMergeTree",
        }
        if engine not in compatible_engines:
            problems.append(
                f"engine is {engine!r}, expected a ReplacingMergeTree-compatible engine"
            )
        compact_engine = _compact_sql(engine_full)
        engine_arguments = re.match(rf"^{re.escape(engine)}\(([^)]*)\)", compact_engine)
        if engine_arguments is None or "revision" not in engine_arguments.group(1).split(","):
            problems.append("ReplacingMergeTree must use revision as a version argument")
        if sorting_key != expected_sorting_key:
            problems.append(f"sorting key is {sorting_key!r}, expected {expected_sorting_key!r}")
        if primary_key != expected_primary_key:
            problems.append(f"primary key is {primary_key!r}, expected {expected_primary_key!r}")
        if partition_key:
            problems.append(f"partition key must be empty, got {partition_key!r}")
        if re.search(r"\bTTL\b", create_query, flags=re.IGNORECASE):
            problems.append("table TTL is not supported because checkpoint history is durable")

    if problems:
        raise RuntimeError(f"Incompatible ClickHouse table {table}: " + "; ".join(problems))


def _compact_sql(expression: str) -> str:
    return re.sub(r"\s+", "", expression)


def _setting_enabled(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "0", "false", "off", "no"}
    return bool(value)


def checkpoint_config(thread_id: str, checkpoint_ns: str, checkpoint_id: str) -> RunnableConfig:
    return {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
        }
    }


def config_values(config: RunnableConfig) -> tuple[str, str, str | None]:
    configurable = config["configurable"]
    return (
        str(configurable["thread_id"]),
        str(configurable.get("checkpoint_ns", "")),
        get_checkpoint_id(config),
    )


class SaverCodec:
    """Pure serialization and SQL-construction helpers shared by both savers."""

    serde: SerializerProtocol
    checkpoints_table: str
    writes_table: str
    legacy_delta_channels: frozenset[str]

    def dump_checkpoint_row(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
    ) -> tuple[Any, ...]:
        thread_id, checkpoint_ns, parent_checkpoint_id = config_values(config)
        checkpoint_type, checkpoint_blob = self.serde.dumps_typed(checkpoint)
        stored_metadata = get_checkpoint_metadata(config, metadata)
        metadata_type, metadata_blob = self.serde.dumps_typed(stored_metadata)
        run_id = stored_metadata.get("run_id", "")
        return (
            thread_id,
            checkpoint_ns,
            checkpoint["id"],
            parent_checkpoint_id or "",
            checkpoint_type,
            checkpoint_blob,
            metadata_type,
            metadata_blob,
            str(run_id) if run_id is not None else "",
        )

    def dump_write_rows(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str,
    ) -> list[tuple[Any, ...]]:
        thread_id, checkpoint_ns, checkpoint_id = config_values(config)
        if checkpoint_id is None:
            raise ValueError("config.configurable.checkpoint_id is required for put_writes")
        rows: list[tuple[Any, ...]] = []
        for position, (channel, value) in enumerate(writes):
            value_type, value_blob = self.serde.dumps_typed(value)
            rows.append(
                (
                    thread_id,
                    checkpoint_ns,
                    checkpoint_id,
                    task_id,
                    task_path,
                    WRITES_IDX_MAP.get(channel, position),
                    channel,
                    value_type,
                    value_blob,
                )
            )
        return rows

    def load_checkpoint_tuple(
        self,
        row: Sequence[Any],
        pending_rows: Sequence[Sequence[Any]],
    ) -> CheckpointTuple:
        (
            thread_id,
            checkpoint_ns,
            checkpoint_id,
            parent_checkpoint_id,
            checkpoint_type,
            checkpoint_blob,
            metadata_type,
            metadata_blob,
        ) = row
        config = checkpoint_config(thread_id, checkpoint_ns, checkpoint_id)
        parent_config = (
            checkpoint_config(thread_id, checkpoint_ns, parent_checkpoint_id)
            if parent_checkpoint_id
            else None
        )
        return CheckpointTuple(
            config,
            cast(
                Checkpoint,
                self.serde.loads_typed((checkpoint_type, bytes(checkpoint_blob))),
            ),
            cast(
                CheckpointMetadata,
                self.serde.loads_typed((metadata_type, bytes(metadata_blob))),
            ),
            parent_config,
            [
                (
                    task_id,
                    channel,
                    self.serde.loads_typed((value_type, bytes(value_blob))),
                )
                for task_id, channel, value_type, value_blob in pending_rows
            ],
        )

    def checkpoint_search(
        self,
        config: RunnableConfig | None,
        before: RunnableConfig | None,
    ) -> tuple[str, dict[str, Any]]:
        predicates: list[str] = []
        parameters: dict[str, Any] = {}
        if config is not None:
            configurable = config["configurable"]
            predicates.append("thread_id = %(thread_id)s")
            parameters["thread_id"] = str(configurable["thread_id"])
            if "checkpoint_ns" in configurable and configurable["checkpoint_ns"] is not None:
                predicates.append("checkpoint_ns = %(checkpoint_ns)s")
                parameters["checkpoint_ns"] = str(configurable["checkpoint_ns"])
            if checkpoint_id := get_checkpoint_id(config):
                predicates.append("checkpoint_id = %(checkpoint_id)s")
                parameters["checkpoint_id"] = checkpoint_id
        if before is not None and (before_id := get_checkpoint_id(before)) is not None:
            predicates.append("checkpoint_id < %(before_id)s")
            parameters["before_id"] = before_id
        clause = f" WHERE {' AND '.join(predicates)}" if predicates else ""
        return clause, parameters

    @staticmethod
    def metadata_matches(metadata: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
        return all(_json_contains(metadata.get(key), value) for key, value in expected.items())

    def retained_keys_for_prune(self, rows: Sequence[Sequence[Any]]) -> set[tuple[str, str, str]]:
        """Select latest checkpoints plus any DeltaChannel reconstruction path.

        A regular namespace retains only its newest checkpoint. If real graph
        metadata or a stored ``_DeltaSnapshot`` identifies delta-backed channels,
        ancestors are retained until the nearest seed for every such channel.
        """
        grouped: dict[tuple[str, str], dict[str, Sequence[Any]]] = {}
        decoded: dict[tuple[str, str, str], tuple[Checkpoint, CheckpointMetadata]] = {}
        delta_channels: dict[tuple[str, str], set[str]] = {}

        for row in rows:
            thread_id, checkpoint_ns, checkpoint_id = row[0], row[1], row[2]
            namespace_key = (thread_id, checkpoint_ns)
            checkpoint_key = (thread_id, checkpoint_ns, checkpoint_id)
            checkpoint = cast(
                Checkpoint,
                self.serde.loads_typed((row[4], bytes(row[5]))),
            )
            metadata = cast(
                CheckpointMetadata,
                self.serde.loads_typed((row[6], bytes(row[7]))),
            )
            grouped.setdefault(namespace_key, {})[checkpoint_id] = row
            decoded[checkpoint_key] = (checkpoint, metadata)
            channels = delta_channels.setdefault(namespace_key, set(self.legacy_delta_channels))
            counters = metadata.get("counters_since_delta_snapshot", {})
            if isinstance(counters, Mapping):
                channels.update(str(channel) for channel in counters)
            channels.update(
                channel
                for channel, value in checkpoint.get("channel_values", {}).items()
                if isinstance(value, _DeltaSnapshot)
            )

        retained: set[tuple[str, str, str]] = set()
        for (thread_id, checkpoint_ns), by_id in grouped.items():
            cursor = max(by_id)
            remaining = set(delta_channels[(thread_id, checkpoint_ns)])
            visited: set[str] = set()
            while cursor in by_id and cursor not in visited:
                visited.add(cursor)
                key = (thread_id, checkpoint_ns, cursor)
                retained.add(key)
                checkpoint, _ = decoded[key]
                remaining.difference_update(checkpoint.get("channel_values", {}))
                if not remaining:
                    break
                parent_checkpoint_id = by_id[cursor][3]
                if not parent_checkpoint_id:
                    break
                cursor = parent_checkpoint_id
        return retained

    @staticmethod
    def key_delete_batches(
        keys: Sequence[tuple[str, str, str]],
    ) -> Iterable[tuple[str, dict[str, Any]]]:
        for offset in range(0, len(keys), _MAX_DELETE_KEYS):
            chunk = keys[offset : offset + _MAX_DELETE_KEYS]
            predicates: list[str] = []
            params: dict[str, Any] = {}
            for index, (thread_id, checkpoint_ns, checkpoint_id) in enumerate(chunk):
                predicates.append(
                    f"(thread_id = %(t{index})s AND checkpoint_ns = %(n{index})s "
                    f"AND checkpoint_id = %(c{index})s)"
                )
                params[f"t{index}"] = thread_id
                params[f"n{index}"] = checkpoint_ns
                params[f"c{index}"] = checkpoint_id
            yield " OR ".join(predicates), params

    @staticmethod
    def values_predicate(column: str, values: Sequence[str]) -> tuple[str, dict[str, Any]]:
        params = {f"value_{index}": value for index, value in enumerate(values)}
        placeholders = ", ".join(f"%(value_{index})s" for index in range(len(values)))
        return f"{column} IN ({placeholders})", params


def _json_contains(actual: Any, expected: Any) -> bool:
    """Approximate PostgreSQL JSONB containment for metadata filters."""
    if isinstance(expected, Mapping):
        return isinstance(actual, Mapping) and all(
            key in actual and _json_contains(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and all(item in actual for item in expected)
    return actual == expected


CHECKPOINT_DDL = """
CREATE TABLE IF NOT EXISTS {table}
(
    thread_id String,
    checkpoint_ns String,
    checkpoint_id String,
    parent_checkpoint_id String DEFAULT '',
    checkpoint_type LowCardinality(String),
    checkpoint_blob String CODEC(ZSTD(3)),
    metadata_type LowCardinality(String),
    metadata_blob String CODEC(ZSTD(3)),
    run_id String DEFAULT '',
    revision UInt128 DEFAULT toUInt128(generateUUIDv7())
)
ENGINE = ReplacingMergeTree(revision)
ORDER BY (thread_id, checkpoint_ns, checkpoint_id)
"""

WRITES_DDL = """
CREATE TABLE IF NOT EXISTS {table}
(
    thread_id String,
    checkpoint_ns String,
    checkpoint_id String,
    task_id String,
    task_path String DEFAULT '',
    idx Int32,
    channel String,
    value_type LowCardinality(String),
    value_blob String CODEC(ZSTD(3)),
    revision UInt128 DEFAULT if(
        idx < 0,
        toUInt128(generateUUIDv7()),
        bitNot(toUInt128(generateUUIDv7()))
    )
)
ENGINE = ReplacingMergeTree(revision)
PRIMARY KEY (thread_id, checkpoint_ns, checkpoint_id)
ORDER BY (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
"""

CHECKPOINT_SELECT = """
SELECT
    thread_id,
    checkpoint_ns,
    checkpoint_id,
    parent_checkpoint_id,
    checkpoint_type,
    checkpoint_blob,
    metadata_type,
    metadata_blob
FROM {table} FINAL
"""

BLOB_QUERY_FORMATS = {
    "checkpoint_blob": "bytes",
    "metadata_blob": "bytes",
    "value_blob": "bytes",
}

DELETE_SETTINGS = {"lightweight_deletes_sync": 2}
DEFAULT_INSERT_SETTINGS = {"async_insert": 0}
