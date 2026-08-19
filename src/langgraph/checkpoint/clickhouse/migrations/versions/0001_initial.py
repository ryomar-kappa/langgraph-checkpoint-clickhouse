"""Create the initial ClickHouse checkpoint schema.

Revision ID: 0001_initial
Revises:
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from alembic import op

from langgraph.checkpoint.clickhouse.migration import validate_baseline_schema

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _prefix() -> str:
    prefix = str(op.get_context().opts["table_prefix"])
    if not _IDENTIFIER.fullmatch(prefix) or len(prefix) > 160:
        raise RuntimeError("invalid table_prefix in Alembic migration context")
    return prefix


def _tables() -> tuple[str, str]:
    prefix = _prefix()
    return f"`{prefix}_checkpoints`", f"`{prefix}_writes`"


def upgrade() -> None:
    checkpoints, writes = _tables()
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {checkpoints}
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
    )
    op.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {writes}
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
    )
    # ClickHouse DDL is non-transactional. Validate before Alembic records this
    # revision so a pre-existing incompatible table cannot be silently adopted.
    if not op.get_context().as_sql:
        validate_baseline_schema(op.get_bind(), _prefix())


def downgrade() -> None:
    checkpoints, writes = _tables()
    op.execute(f"DROP TABLE IF EXISTS {writes} SYNC")
    op.execute(f"DROP TABLE IF EXISTS {checkpoints} SYNC")
