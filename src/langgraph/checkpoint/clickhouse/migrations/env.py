from __future__ import annotations

import os
from logging.config import fileConfig
from typing import Any

from alembic import context
from alembic.script import ScriptDirectory
from clickhouse_connect.cc_sqlalchemy import alembic as ch_alembic
from sqlalchemy import create_engine, pool
from sqlalchemy.engine import URL, make_url

from langgraph.checkpoint.clickhouse._internal import validate_table_prefix
from langgraph.checkpoint.clickhouse.migration import (
    ALEMBIC_URL_ENV,
    TABLE_PREFIX_ENV,
    build_migration_metadata,
    migration_version_table,
    read_version_rows,
    validate_existing_schema,
    validate_present_baseline_schema,
)

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)


def _arguments() -> tuple[str, str, str | URL]:
    x_args: dict[str, str] = context.get_x_argument(as_dictionary=True)
    prefix = validate_table_prefix(
        x_args.get("table_prefix")
        or config.attributes.get("table_prefix")
        or os.getenv(TABLE_PREFIX_ENV, "langgraph_checkpoint")
    )
    version_table = validate_table_prefix(
        x_args.get("version_table")
        or config.attributes.get("version_table")
        or migration_version_table(prefix)
    )
    url: Any = config.attributes.get("connection_url") or os.getenv(ALEMBIC_URL_ENV)
    if url is None:
        configured_url = config.get_main_option("sqlalchemy.url") or ""
        url = configured_url if configured_url.strip() else None
    if url is None:
        raise RuntimeError(f"Set {ALEMBIC_URL_ENV} to a clickhousedb:// SQLAlchemy URL")
    return prefix, version_table, url


table_prefix, version_table, connection_url = _arguments()
target_metadata = build_migration_metadata(table_prefix)


def _include_object(database: str | None):
    schemas = frozenset({database}) if database else None
    return ch_alembic.make_include_object(
        exclude_tables=frozenset({version_table}),
        include_schemas=schemas,
        base_include_object_fn=ch_alembic.include_object,
    )


def run_migrations_offline() -> None:
    database = make_url(connection_url).database
    context.configure(
        url=connection_url,
        dialect_name="clickhousedb",
        dialect_opts={"max_identifier_length": 255},
        target_metadata=target_metadata,
        literal_binds=True,
        compare_type=True,
        compare_server_default=True,
        include_object=_include_object(database),
        transactional_ddl=False,
        version_table=version_table,
        version_table_schema=database,
        table_prefix=table_prefix,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = create_engine(
        connection_url,
        poolclass=pool.NullPool,
        max_identifier_length=255,
    )
    try:
        with connectable.connect() as connection:
            database = str(connection.exec_driver_sql("SELECT currentDatabase()").scalar_one())
            if not read_version_rows(connection, version_table):
                # Initial adoption/retry only: validate tables already present.
                # Later revisions own their own pre/postconditions and must be
                # allowed to migrate a schema that intentionally differs from head.
                validate_present_baseline_schema(connection, table_prefix)
            context.configure(
                connection=connection,
                target_metadata=target_metadata,
                include_schemas=True,
                include_name=ch_alembic.make_include_name(
                    include_schemas=frozenset({database}),
                    default_schema=database,
                ),
                include_object=_include_object(database),
                process_revision_directives=ch_alembic.clickhouse_writer,
                compare_type=True,
                compare_server_default=True,
                transactional_ddl=False,
                version_table=version_table,
                version_table_schema=database,
                table_prefix=table_prefix,
            )
            with context.begin_transaction():
                context.run_migrations()
            if set(read_version_rows(connection, version_table)) == set(
                ScriptDirectory.from_config(config).get_heads()
            ):
                # Validate the current target after CLI and programmatic upgrades,
                # including the no-op case where the database was already at head.
                validate_existing_schema(connection, table_prefix)
    finally:
        connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
