from __future__ import annotations

import os
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any
from uuid import uuid4

import clickhouse_connect
import pytest

from langgraph.checkpoint.clickhouse import AsyncClickHouseSaver, ClickHouseSaver


@pytest.fixture(scope="session")
def clickhouse_kwargs() -> dict[str, Any]:
    return {
        "host": os.getenv("CLICKHOUSE_HOST", "127.0.0.1"),
        "port": int(os.getenv("CLICKHOUSE_PORT", "18123")),
        "username": os.getenv("CLICKHOUSE_USER", "langgraph"),
        "password": os.getenv("CLICKHOUSE_PASSWORD", "langgraph_test_password"),
        "database": os.getenv("CLICKHOUSE_DATABASE", "langgraph_test"),
        "connect_timeout": 5,
        "send_receive_timeout": 60,
    }


@pytest.fixture(scope="session", autouse=True)
def wait_for_clickhouse(clickhouse_kwargs: dict[str, Any]) -> Iterator[None]:
    deadline = time.monotonic() + 45
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        client = None
        try:
            client = clickhouse_connect.get_client(**clickhouse_kwargs)
            assert client.command("SELECT 1") == 1
            client.close()
            yield
            return
        except Exception as error:  # pragma: no cover - only used during container startup
            last_error = error
            if client is not None:
                client.close()
            time.sleep(0.5)
    pytest.fail(f"ClickHouse did not become ready: {last_error!r}")


def unique_prefix(label: str) -> str:
    return f"lgcp_{label}_{uuid4().hex}"


@pytest.fixture
def sync_saver(clickhouse_kwargs: dict[str, Any]) -> Iterator[ClickHouseSaver]:
    client = clickhouse_connect.get_client(**clickhouse_kwargs)
    saver = ClickHouseSaver(client, table_prefix=unique_prefix("sync"))
    saver.setup()
    try:
        yield saver
    finally:
        saver.drop_tables()
        client.close()


@pytest.fixture
async def async_saver(
    clickhouse_kwargs: dict[str, Any],
) -> AsyncIterator[AsyncClickHouseSaver]:
    client = await clickhouse_connect.get_async_client(**clickhouse_kwargs)
    saver = AsyncClickHouseSaver(client, table_prefix=unique_prefix("async"))
    await saver.setup()
    try:
        yield saver
    finally:
        await saver.drop_tables()
        await client.close()
