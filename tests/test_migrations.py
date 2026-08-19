from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import clickhouse_connect
import pytest
from alembic import command
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import CheckpointMetadata, empty_checkpoint
from sqlalchemy.engine import URL

from langgraph.checkpoint.clickhouse import ClickHouseSaver
from langgraph.checkpoint.clickhouse._internal import (
    CHECKPOINT_DDL,
    WRITES_DDL,
    quote_identifier,
)
from langgraph.checkpoint.clickhouse.migration import (
    ALEMBIC_URL_ENV,
    BASELINE_REVISION,
    adopt_existing_schema,
    downgrade_schema,
    make_alembic_config,
    migration_table_names,
    migration_version_table,
    upgrade_schema,
)

from .conftest import unique_prefix


def _sqlalchemy_url(clickhouse_kwargs: dict[str, Any]) -> URL:
    return URL.create(
        "clickhousedb",
        username=str(clickhouse_kwargs["username"]),
        password=str(clickhouse_kwargs["password"]),
        host=str(clickhouse_kwargs["host"]),
        port=int(clickhouse_kwargs["port"]),
        database=str(clickhouse_kwargs["database"]),
    )


def _drop_tables(client: Any, *table_names: str) -> None:
    for table_name in table_names:
        client.command(f"DROP TABLE IF EXISTS {quote_identifier(table_name)} SYNC")


def _exists(client: Any, table_name: str) -> bool:
    return bool(client.command(f"EXISTS TABLE {quote_identifier(table_name)}"))


def _versions(client: Any, version_table: str) -> tuple[str, ...]:
    if not _exists(client, version_table):
        return ()
    return tuple(
        str(row[0])
        for row in client.query(
            f"SELECT version_num FROM {quote_identifier(version_table)} ORDER BY version_num"
        ).result_rows
    )


def _roundtrip(saver: ClickHouseSaver, value: str) -> RunnableConfig:
    checkpoint = empty_checkpoint()
    checkpoint["channel_values"] = {"migration": value}
    checkpoint["channel_versions"] = {"migration": 1}
    config = cast(
        RunnableConfig,
        {"configurable": {"thread_id": str(uuid4()), "checkpoint_ns": ""}},
    )
    metadata = cast(CheckpointMetadata, {"source": "input", "step": -1})
    stored = saver.put(config, checkpoint, metadata, checkpoint["channel_versions"])
    loaded = saver.get_tuple(stored)
    assert loaded is not None
    assert loaded.checkpoint["channel_values"] == {"migration": value}
    return stored


@pytest.mark.integration
def test_fresh_upgrade_downgrade_reupgrade_and_saver_roundtrip(
    clickhouse_kwargs: dict[str, Any],
) -> None:
    prefix = unique_prefix("migration_lifecycle")
    version_table = f"{prefix}_history"
    checkpoints_table, writes_table = migration_table_names(prefix)
    url = _sqlalchemy_url(clickhouse_kwargs)
    client = clickhouse_connect.get_client(**clickhouse_kwargs)
    try:
        upgrade_schema(table_prefix=prefix, version_table=version_table, url=url)
        assert _versions(client, version_table) == (BASELINE_REVISION,)
        command.check(
            make_alembic_config(
                table_prefix=prefix,
                version_table=version_table,
                url=url,
            )
        )

        saver = ClickHouseSaver(client, table_prefix=prefix)
        saver.setup()
        _roundtrip(saver, "fresh-upgrade")

        downgrade_schema(table_prefix=prefix, version_table=version_table, url=url)
        assert not _exists(client, checkpoints_table)
        assert not _exists(client, writes_table)
        assert _versions(client, version_table) == ()

        upgrade_schema(table_prefix=prefix, version_table=version_table, url=url)
        saver.setup()
        _roundtrip(saver, "re-upgrade")
        assert _versions(client, version_table) == (BASELINE_REVISION,)
    finally:
        _drop_tables(client, checkpoints_table, writes_table, version_table)
        client.close()


@pytest.mark.integration
def test_setup_schema_can_be_adopted_without_data_loss(
    clickhouse_kwargs: dict[str, Any],
) -> None:
    prefix = unique_prefix("migration_adopt")
    version_table = f"{prefix}_history"
    checkpoints_table, writes_table = migration_table_names(prefix)
    url = _sqlalchemy_url(clickhouse_kwargs)
    client = clickhouse_connect.get_client(**clickhouse_kwargs)
    saver = ClickHouseSaver(client, table_prefix=prefix)
    try:
        saver.setup()
        stored = _roundtrip(saver, "before-adoption")

        adopt_existing_schema(table_prefix=prefix, version_table=version_table, url=url)
        adopt_existing_schema(table_prefix=prefix, version_table=version_table, url=url)
        assert _versions(client, version_table) == (BASELINE_REVISION,)

        upgrade_schema(table_prefix=prefix, version_table=version_table, url=url)
        loaded = saver.get_tuple(stored)
        assert loaded is not None
        assert loaded.checkpoint["channel_values"] == {"migration": "before-adoption"}
    finally:
        _drop_tables(client, checkpoints_table, writes_table, version_table)
        client.close()


@pytest.mark.integration
def test_partial_nontransactional_ddl_can_be_retried(
    clickhouse_kwargs: dict[str, Any],
) -> None:
    prefix = unique_prefix("migration_partial")
    version_table = f"{prefix}_history"
    checkpoints_table, writes_table = migration_table_names(prefix)
    url = _sqlalchemy_url(clickhouse_kwargs)
    client = clickhouse_connect.get_client(**clickhouse_kwargs)
    try:
        client.command(CHECKPOINT_DDL.format(table=quote_identifier(checkpoints_table)))
        assert _exists(client, checkpoints_table)
        assert not _exists(client, writes_table)

        upgrade_schema(table_prefix=prefix, version_table=version_table, url=url)
        assert _exists(client, writes_table)
        assert _versions(client, version_table) == (BASELINE_REVISION,)
        ClickHouseSaver(client, table_prefix=prefix).setup()
    finally:
        _drop_tables(client, checkpoints_table, writes_table, version_table)
        client.close()


@pytest.mark.integration
def test_incompatible_partial_schema_is_not_stamped(
    clickhouse_kwargs: dict[str, Any],
) -> None:
    prefix = unique_prefix("migration_invalid")
    version_table = f"{prefix}_history"
    checkpoints_table, writes_table = migration_table_names(prefix)
    url = _sqlalchemy_url(clickhouse_kwargs)
    client = clickhouse_connect.get_client(**clickhouse_kwargs)
    try:
        invalid_ddl = WRITES_DDL.format(table=quote_identifier(writes_table)).replace(
            "ENGINE = ReplacingMergeTree(revision)",
            "ENGINE = MergeTree",
        )
        client.command(invalid_ddl)
        with pytest.raises(RuntimeError, match="Incompatible ClickHouse table"):
            upgrade_schema(table_prefix=prefix, version_table=version_table, url=url)
        assert _versions(client, version_table) == ()
    finally:
        _drop_tables(client, checkpoints_table, writes_table, version_table)
        client.close()


@pytest.mark.integration
def test_installed_cli_works_outside_repository(
    clickhouse_kwargs: dict[str, Any],
    tmp_path: Path,
) -> None:
    prefix = unique_prefix("migration_cli")
    version_table = f"{prefix}_history"
    checkpoints_table, writes_table = migration_table_names(prefix)
    url = _sqlalchemy_url(clickhouse_kwargs)
    client = clickhouse_connect.get_client(**clickhouse_kwargs)
    executable = Path(sys.executable).with_name("langgraph-checkpoint-clickhouse-migrate")
    environment = os.environ.copy()
    environment[ALEMBIC_URL_ENV] = url.render_as_string(hide_password=False)
    common_arguments = [
        str(executable),
        "-x",
        f"table_prefix={prefix}",
        "-x",
        f"version_table={version_table}",
    ]
    try:
        heads = subprocess.run(
            [str(executable), "heads"],
            cwd=tmp_path,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        assert BASELINE_REVISION in heads.stdout

        subprocess.run(
            [*common_arguments, "upgrade", "head"],
            cwd=tmp_path,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        saver = ClickHouseSaver(client, table_prefix=prefix)
        saver.setup()
        _roundtrip(saver, "installed-wheel-cli")
        assert _versions(client, version_table) == (BASELINE_REVISION,)

        subprocess.run(
            [*common_arguments, "downgrade", "base"],
            cwd=tmp_path,
            env=environment,
            check=True,
            capture_output=True,
            text=True,
        )
        assert not _exists(client, checkpoints_table)
        assert not _exists(client, writes_table)
    finally:
        _drop_tables(client, checkpoints_table, writes_table, version_table)
        client.close()


def test_long_prefix_gets_a_bounded_unique_version_table() -> None:
    first = migration_version_table("a" * 160)
    second = migration_version_table("a" * 159 + "b")
    assert len(first) == 160
    assert len(second) == 160
    assert first != second
    assert first.endswith("_alembic_version")


def test_offline_upgrade_sql_can_be_generated_outside_repository(tmp_path: Path) -> None:
    prefix = unique_prefix("migration_offline")
    version_table = f"{prefix}_history"
    executable = Path(sys.executable).with_name("langgraph-checkpoint-clickhouse-migrate")
    environment = os.environ.copy()
    environment[ALEMBIC_URL_ENV] = "clickhousedb://default@localhost/default"
    generated = subprocess.run(
        [
            str(executable),
            "-x",
            f"table_prefix={prefix}",
            "-x",
            f"version_table={version_table}",
            "upgrade",
            "head",
            "--sql",
        ],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    assert f"CREATE TABLE IF NOT EXISTS `{prefix}_checkpoints`" in generated.stdout
    assert f"CREATE TABLE IF NOT EXISTS `{prefix}_writes`" in generated.stdout
    assert version_table in generated.stdout
