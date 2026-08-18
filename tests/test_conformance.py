from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import clickhouse_connect
import pytest
from langgraph.checkpoint.conformance import checkpointer_test, validate
from langgraph.checkpoint.conformance.report import ProgressCallbacks

from langgraph.checkpoint.clickhouse import AsyncClickHouseSaver, ClickHouseSaver

from .conftest import unique_prefix


@pytest.mark.integration
@pytest.mark.conformance
async def test_official_langgraph_checkpointer_conformance(
    clickhouse_kwargs: dict[str, Any],
) -> None:
    @checkpointer_test(name="AsyncClickHouseSaver")
    async def clickhouse_checkpointer() -> AsyncIterator[AsyncClickHouseSaver]:
        client = await clickhouse_connect.get_async_client(**clickhouse_kwargs)
        saver = AsyncClickHouseSaver(client, table_prefix=unique_prefix("conformance"))
        await saver.setup()
        try:
            yield saver
        finally:
            await saver.drop_tables()
            await client.close()

    report = await validate(clickhouse_checkpointer, progress=ProgressCallbacks.default())
    report.print_report()

    assert report.passed_all_base(), report.to_dict()
    assert report.passed_all(), report.to_dict()
    # Released suite 0.0.2: 58 required + 23 extended tests.
    assert sum(result.tests_passed for result in report.results.values()) == 81


@pytest.mark.integration
@pytest.mark.conformance
async def test_sync_saver_async_wrappers_conformance(
    clickhouse_kwargs: dict[str, Any],
) -> None:
    @checkpointer_test(name="ClickHouseSaver async wrappers")
    async def clickhouse_checkpointer() -> AsyncIterator[ClickHouseSaver]:
        client = clickhouse_connect.get_client(**clickhouse_kwargs)
        saver = ClickHouseSaver(client, table_prefix=unique_prefix("sync_conformance"))
        saver.setup()
        try:
            yield saver
        finally:
            saver.drop_tables()
            client.close()

    report = await validate(clickhouse_checkpointer, progress=ProgressCallbacks.default())
    report.print_report()

    assert report.passed_all_base(), report.to_dict()
    assert report.passed_all(), report.to_dict()
    assert sum(result.tests_passed for result in report.results.values()) == 81
