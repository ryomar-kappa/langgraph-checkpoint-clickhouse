from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path
from typing import cast

from alembic import command
from alembic.config import CommandLine, Config
from clickhouse_connect.cc_sqlalchemy import engines, types
from sqlalchemy import Column, MetaData, Table, create_engine, text
from sqlalchemy.engine import URL, Connection
from sqlalchemy.pool import NullPool
from sqlalchemy.types import TypeEngine

from langgraph.checkpoint.clickhouse._internal import (
    CHECKPOINT_COLUMN_TYPES,
    CHECKPOINT_DEFAULTS,
    WRITE_COLUMN_TYPES,
    WRITE_DEFAULTS,
    quote_identifier,
    validate_schema,
    validate_table_prefix,
)

ALEMBIC_URL_ENV = "CLICKHOUSE_ALEMBIC_URL"
TABLE_PREFIX_ENV = "LANGGRAPH_CHECKPOINT_TABLE_PREFIX"
BASELINE_REVISION = "0001_initial"

# Immutable 0001 snapshot. Future saver schemas must update the current specs in
# _internal.py without changing these, so a fresh database can still traverse 0001.
_BASELINE_CHECKPOINT_COLUMN_TYPES = {
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
_BASELINE_WRITE_COLUMN_TYPES = {
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
_BASELINE_CHECKPOINT_DEFAULTS = {
    "parent_checkpoint_id": "''",
    "run_id": "''",
    "revision": "toUInt128(generateUUIDv7())",
}
_BASELINE_WRITE_DEFAULTS = {
    "task_path": "''",
    "revision": "if(idx<0,toUInt128(generateUUIDv7()),bitNot(toUInt128(generateUUIDv7())))",
}


def migration_table_names(table_prefix: str) -> tuple[str, str]:
    """Return checkpoint and write table names for a validated prefix."""
    prefix = validate_table_prefix(table_prefix)
    return f"{prefix}_checkpoints", f"{prefix}_writes"


def migration_version_table(table_prefix: str) -> str:
    """Use an independent Alembic history for every saver table prefix."""
    prefix = validate_table_prefix(table_prefix)
    suffix = "_alembic_version"
    if len(prefix) + len(suffix) <= 160:
        return f"{prefix}{suffix}"

    # Saver prefixes can themselves use the full 160-character allowance. Keep
    # their migration histories independent without exceeding that same bound.
    digest = sha256(prefix.encode()).hexdigest()[:12]
    shortened = prefix[: 160 - len(suffix) - len(digest) - 1]
    return f"{shortened}_{digest}{suffix}"


def resolve_version_table(table_prefix: str, version_table: str | None = None) -> str:
    """Validate an override or derive a bounded, prefix-specific history table."""
    return validate_table_prefix(version_table or migration_version_table(table_prefix))


def _low_cardinality_string() -> TypeEngine[str]:
    # clickhouse-connect returns a valid SQLAlchemy UserDefinedType at runtime,
    # but its public return annotation is the broader ChSqlaType protocol.
    return cast(TypeEngine[str], types.LowCardinality(types.String()))


def build_migration_metadata(table_prefix: str) -> MetaData:
    """Describe the current saver schema for Alembic autogeneration."""
    checkpoints_name, writes_name = migration_table_names(table_prefix)
    metadata = MetaData()

    Table(
        checkpoints_name,
        metadata,
        Column("thread_id", types.String(), nullable=False),
        Column("checkpoint_ns", types.String(), nullable=False),
        Column("checkpoint_id", types.String(), nullable=False),
        Column("parent_checkpoint_id", types.String(), nullable=False, server_default=text("''")),
        Column("checkpoint_type", _low_cardinality_string(), nullable=False),
        Column(
            "checkpoint_blob",
            types.String(),
            nullable=False,
            clickhousedb_codec="ZSTD(3)",
        ),
        Column("metadata_type", _low_cardinality_string(), nullable=False),
        Column(
            "metadata_blob",
            types.String(),
            nullable=False,
            clickhousedb_codec="ZSTD(3)",
        ),
        Column("run_id", types.String(), nullable=False, server_default=text("''")),
        Column(
            "revision",
            types.UInt128(),
            nullable=False,
            server_default=text("toUInt128(generateUUIDv7())"),
        ),
        engines.ReplacingMergeTree(
            version="revision",
            order_by=("thread_id", "checkpoint_ns", "checkpoint_id"),
        ),
    )

    Table(
        writes_name,
        metadata,
        Column("thread_id", types.String(), nullable=False),
        Column("checkpoint_ns", types.String(), nullable=False),
        Column("checkpoint_id", types.String(), nullable=False),
        Column("task_id", types.String(), nullable=False),
        Column("task_path", types.String(), nullable=False, server_default=text("''")),
        Column("idx", types.Int32(), nullable=False),
        Column("channel", types.String(), nullable=False),
        Column("value_type", _low_cardinality_string(), nullable=False),
        Column(
            "value_blob",
            types.String(),
            nullable=False,
            clickhousedb_codec="ZSTD(3)",
        ),
        Column(
            "revision",
            types.UInt128(),
            nullable=False,
            server_default=text(
                "if(idx < 0, toUInt128(generateUUIDv7()), bitNot(toUInt128(generateUUIDv7())))"
            ),
        ),
        engines.ReplacingMergeTree(
            version="revision",
            primary_key=("thread_id", "checkpoint_ns", "checkpoint_id"),
            order_by=("thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"),
        ),
    )
    return metadata


def make_alembic_config(
    *,
    table_prefix: str = "langgraph_checkpoint",
    version_table: str | None = None,
    url: str | URL | None = None,
    config_path: str | Path | None = None,
) -> Config:
    """Create an Alembic config without placing credentials in ``alembic.ini``."""
    prefix = validate_table_prefix(table_prefix)
    path = Path(config_path) if config_path is not None else _default_config_path()
    config = Config(str(path))
    config.attributes["table_prefix"] = prefix
    config.attributes["version_table"] = resolve_version_table(prefix, version_table)
    if url is not None:
        config.attributes["connection_url"] = url
    return config


def upgrade_schema(
    *,
    table_prefix: str = "langgraph_checkpoint",
    version_table: str | None = None,
    url: str | URL | None = None,
    revision: str = "head",
) -> None:
    """Apply schema revisions with ClickHouse's non-transactional Alembic dialect."""
    config = make_alembic_config(
        table_prefix=table_prefix,
        version_table=version_table,
        url=url,
    )
    command.upgrade(config, revision)
    if revision == "head":
        validate_schema_at_url(url=_resolve_url(url), table_prefix=table_prefix)


def downgrade_schema(
    *,
    table_prefix: str = "langgraph_checkpoint",
    version_table: str | None = None,
    url: str | URL | None = None,
    revision: str = "base",
) -> None:
    """Apply explicit downgrade DDL; ClickHouse cannot roll a migration back atomically."""
    command.downgrade(
        make_alembic_config(
            table_prefix=table_prefix,
            version_table=version_table,
            url=url,
        ),
        revision,
    )


def adopt_existing_schema(
    *,
    table_prefix: str = "langgraph_checkpoint",
    version_table: str | None = None,
    url: str | URL | None = None,
) -> None:
    """Validate setup-created tables and stamp them at the immutable baseline.

    This deliberately refuses to overwrite a different Alembic revision. The caller
    must ensure that only one migration or adoption process runs for a database.
    """
    prefix = validate_table_prefix(table_prefix)
    connection_url = _resolve_url(url)
    history_table = resolve_version_table(prefix, version_table)
    engine = create_engine(
        connection_url,
        poolclass=NullPool,
        max_identifier_length=255,
    )
    try:
        with engine.connect() as connection:
            validate_baseline_schema(connection, prefix)
            versions = read_version_rows(connection, history_table)
    finally:
        engine.dispose()

    if versions == (BASELINE_REVISION,):
        return
    if versions:
        raise RuntimeError(
            f"Cannot adopt {prefix!r}: {history_table} contains revisions {versions!r}"
        )

    command.stamp(
        make_alembic_config(
            table_prefix=prefix,
            version_table=history_table,
            url=connection_url,
        ),
        BASELINE_REVISION,
    )


def validate_schema_at_url(
    *,
    table_prefix: str = "langgraph_checkpoint",
    url: str | URL | None = None,
) -> None:
    """Connect and verify that the migrated tables match saver invariants."""
    prefix = validate_table_prefix(table_prefix)
    engine = create_engine(
        _resolve_url(url),
        poolclass=NullPool,
        max_identifier_length=255,
    )
    try:
        with engine.connect() as connection:
            validate_existing_schema(connection, prefix)
    finally:
        engine.dispose()


def validate_existing_schema(connection: Connection, table_prefix: str) -> None:
    """Run the same semantic schema checks used by ``ClickHouseSaver.setup``."""
    expected_names = set(migration_table_names(table_prefix))
    existing_names = _existing_application_tables(connection, table_prefix)
    missing_names = sorted(expected_names - existing_names)
    if missing_names:
        raise RuntimeError(f"Missing ClickHouse checkpoint tables: {missing_names!r}")
    _validate_schema_tables(
        connection,
        table_prefix,
        expected_names,
        checkpoint_types=CHECKPOINT_COLUMN_TYPES,
        checkpoint_defaults=CHECKPOINT_DEFAULTS,
        write_types=WRITE_COLUMN_TYPES,
        write_defaults=WRITE_DEFAULTS,
    )


def validate_baseline_schema(connection: Connection, table_prefix: str) -> None:
    """Require both tables to match the immutable ``0001_initial`` schema."""
    expected_names = set(migration_table_names(table_prefix))
    existing_names = _existing_application_tables(connection, table_prefix)
    missing_names = sorted(expected_names - existing_names)
    if missing_names:
        raise RuntimeError(f"Missing ClickHouse checkpoint tables: {missing_names!r}")
    _validate_baseline_tables(connection, table_prefix, expected_names)


def validate_present_schema(connection: Connection, table_prefix: str) -> None:
    """Validate any saver tables that exist, allowing a partial DDL retry."""
    existing_names = _existing_application_tables(connection, table_prefix)
    _validate_schema_tables(
        connection,
        table_prefix,
        existing_names,
        checkpoint_types=CHECKPOINT_COLUMN_TYPES,
        checkpoint_defaults=CHECKPOINT_DEFAULTS,
        write_types=WRITE_COLUMN_TYPES,
        write_defaults=WRITE_DEFAULTS,
    )


def validate_present_baseline_schema(connection: Connection, table_prefix: str) -> None:
    """Validate existing 0001 tables while allowing a partial DDL retry."""
    existing_names = _existing_application_tables(connection, table_prefix)
    _validate_baseline_tables(connection, table_prefix, existing_names)


def _validate_baseline_tables(
    connection: Connection,
    table_prefix: str,
    table_names: set[str],
) -> None:
    _validate_schema_tables(
        connection,
        table_prefix,
        table_names,
        checkpoint_types=_BASELINE_CHECKPOINT_COLUMN_TYPES,
        checkpoint_defaults=_BASELINE_CHECKPOINT_DEFAULTS,
        write_types=_BASELINE_WRITE_COLUMN_TYPES,
        write_defaults=_BASELINE_WRITE_DEFAULTS,
    )


def _existing_application_tables(
    connection: Connection,
    table_prefix: str,
) -> set[str]:
    checkpoints_name, writes_name = migration_table_names(table_prefix)
    return {
        str(row[0])
        for row in connection.exec_driver_sql(
            """
            SELECT name
            FROM system.tables
            WHERE database = currentDatabase()
              AND name IN (%(checkpoints_table)s, %(writes_table)s)
            """,
            {
                "checkpoints_table": checkpoints_name,
                "writes_table": writes_name,
            },
        ).fetchall()
    }


def _validate_schema_tables(
    connection: Connection,
    table_prefix: str,
    table_names: set[str],
    *,
    checkpoint_types: dict[str, str],
    checkpoint_defaults: dict[str, str],
    write_types: dict[str, str],
    write_defaults: dict[str, str],
) -> None:
    checkpoints_name, writes_name = migration_table_names(table_prefix)
    specs = (
        (
            checkpoints_name,
            checkpoint_types,
            checkpoint_defaults,
            "thread_id, checkpoint_ns, checkpoint_id",
            "thread_id, checkpoint_ns, checkpoint_id",
        ),
        (
            writes_name,
            write_types,
            write_defaults,
            "thread_id, checkpoint_ns, checkpoint_id, task_id, idx",
            "thread_id, checkpoint_ns, checkpoint_id",
        ),
    )
    for table_name, expected_types, defaults, sorting_key, primary_key in specs:
        if table_name not in table_names:
            continue
        quoted_table = quote_identifier(table_name)
        description_rows = [
            tuple(row)
            for row in connection.exec_driver_sql(f"DESCRIBE TABLE {quoted_table}").fetchall()
        ]
        table_info_rows = connection.exec_driver_sql(
            """
            SELECT engine, engine_full, sorting_key, primary_key,
                   partition_key, create_table_query
            FROM system.tables
            WHERE database = currentDatabase() AND name = %(table_name)s
            """,
            {"table_name": table_name},
        ).fetchall()
        validate_schema(
            table=quoted_table,
            description_rows=description_rows,
            table_info=tuple(table_info_rows[0]) if table_info_rows else None,
            expected_types=expected_types,
            expected_defaults=defaults,
            expected_sorting_key=sorting_key,
            expected_primary_key=primary_key,
        )


def read_version_rows(connection: Connection, version_table: str) -> tuple[str, ...]:
    """Return the migration revisions recorded in a validated history table."""
    history_table = validate_table_prefix(version_table)
    quoted_table = quote_identifier(history_table)
    exists = connection.exec_driver_sql(f"EXISTS TABLE {quoted_table}").scalar()
    if not exists:
        return ()
    return tuple(
        str(row[0])
        for row in connection.exec_driver_sql(
            f"SELECT version_num FROM {quoted_table} ORDER BY version_num"
        ).fetchall()
    )


def _resolve_url(url: str | URL | None) -> str | URL:
    value: str | URL | None = url or os.getenv(ALEMBIC_URL_ENV)
    if value is None or (isinstance(value, str) and not value.strip()):
        raise RuntimeError(f"Set {ALEMBIC_URL_ENV} to a clickhousedb:// SQLAlchemy URL")
    return value


def _default_config_path() -> Path:
    packaged = Path(__file__).with_name("alembic.ini")
    if packaged.is_file():
        return packaged
    repository = Path(__file__).resolve().parents[4] / "alembic.ini"
    if repository.is_file():
        return repository
    raise RuntimeError("Bundled Alembic configuration could not be located")


def main(argv: Sequence[str] | None = None) -> None:
    """Run Alembic using the configuration bundled in the installed wheel."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    has_config = any(
        argument in {"-c", "--config"} or argument.startswith("--config=") for argument in arguments
    )
    if not has_config:
        arguments[:0] = ["-c", str(_default_config_path())]
    CommandLine(prog="langgraph-checkpoint-clickhouse-migrate").main(arguments)


__all__ = [
    "ALEMBIC_URL_ENV",
    "BASELINE_REVISION",
    "TABLE_PREFIX_ENV",
    "adopt_existing_schema",
    "build_migration_metadata",
    "downgrade_schema",
    "make_alembic_config",
    "migration_table_names",
    "migration_version_table",
    "read_version_rows",
    "resolve_version_table",
    "upgrade_schema",
    "validate_baseline_schema",
    "validate_existing_schema",
    "validate_present_baseline_schema",
    "validate_present_schema",
    "validate_schema_at_url",
]
