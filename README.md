# LangGraph Checkpoint ClickHouse

`langgraph-checkpoint` 4系の `BaseCheckpointSaver` をClickHouse向けに実装したPythonパッケージです。
同期用の `ClickHouseSaver` と、`clickhouse-connect` のnative async clientを使う
`AsyncClickHouseSaver` を提供します。

## 対応範囲

- checkpointの保存、最新版・ID指定取得、履歴一覧
- pending writesと特殊write（error / interrupt / resume / scheduled）
- namespace、metadata filter、`before`、`limit`
- thread削除、run ID単位削除、threadコピー、prune
- 同期・非同期LangGraph実行
- DeltaChannelの親checkpoint履歴復元
- typed serializer（既定はLangGraphの `JsonPlusSerializer`）

テストは `langgraph-checkpoint-conformance==0.0.2` の必須58件と拡張23件をnative async saverと
sync saverのasync wrapperの両方でそのまま実行します。配布版conformanceにはまだ含まれていないDeltaChannel、同期API、
実グラフ、interrupt/resume、再接続後の永続性、並行writeについても統合テストを追加しています。

## インストール

GitHubから直接インストールできます。Alembicでschemaを管理する場合は
`migration` extraも指定してください。

```bash
uv add \
  "langgraph-checkpoint-clickhouse[migration] @ git+https://github.com/ryomar-kappa/langgraph-checkpoint-clickhouse.git"
```

`pip`を使う場合は次の通りです。

```bash
python -m pip install \
  "langgraph-checkpoint-clickhouse[migration] @ git+https://github.com/ryomar-kappa/langgraph-checkpoint-clickhouse.git"
```

ソースコードをコピーして使う場合は、`src/langgraph/checkpoint/clickhouse/`を
`alembic.ini`と`migrations/`も含めてディレクトリごとコピーしてください。migrationを使う環境には
`clickhouse-connect[async,alembic]`、`alembic>=1.16,<2`、`SQLAlchemy>=1.4.40,<3`が必要です。

## Alembicによるschema管理

本パッケージは`clickhouse-connect`の
[Alembic連携](https://github.com/ClickHouse/clickhouse-connect/blob/main/clickhouse_connect/cc_sqlalchemy/alembic/WORKED_EXAMPLE.md)
を使います。本番ではアプリ起動前のrelease stepやmigration jobとして、1プロセスだけで
Alembicを実行してください。Saverのconstructor、worker起動、最初のrequestからは自動実行
されません。

AlembicはSQLAlchemy dialect用の `clickhousedb://` URLを使います。Saverへ渡す
`http://` / `https://` DSNとはschemeが異なります。

```bash
export CLICKHOUSE_ALEMBIC_URL="clickhousedb://langgraph:langgraph_test_password@localhost:18123/langgraph_test"
export CLICKHOUSE_DSN="http://langgraph:langgraph_test_password@localhost:18123/langgraph_test"
```

パスワードに `@`、`:`、`/` などが含まれる場合はURL encodeしてください。

### 新しいdatabaseへ適用

database自体は先に作成しておきます。migration資材はwheelに同梱されているため、
repositoryをcloneした場所以外からも専用CLIで実行できます。

```bash
uv run langgraph-checkpoint-clickhouse-migrate upgrade head
uv run langgraph-checkpoint-clickhouse-migrate current
```

初期migrationは次のtableを作成します。

```text
langgraph_checkpoint_checkpoints
langgraph_checkpoint_writes
```

Pythonからの明示実行も可能です。

```python
from langgraph.checkpoint.clickhouse.migration import upgrade_schema

upgrade_schema()  # CLICKHOUSE_ALEMBIC_URLを使用
```

ClickHouseのDDLはrevision全体をtransactionでrollbackできません。初期migrationは、1表だけ
作成された途中状態から再実行で復旧できます。一方、既存表のengine、key、列型、defaultが
非互換な場合はrevisionを記録せず失敗します。途中失敗時は状態を確認し、同じ`upgrade head`を
再実行してください。

`upgrade head --sql`によるoffline SQL生成も可能ですが、既存schemaの互換性は検査できないため、
空のdatabase用として扱い、適用後に通常の`current`と`setup()`で確認してください。

### `setup()`との関係

現行の`setup()`は`CREATE TABLE IF NOT EXISTS`と互換schemaの検証を行いますが、
Alembicのversion tableは更新しません。本番では先に`upgrade head`を実行します。
application用credentialにDDL権限がある場合は、その後`setup()`を起動時の追加schema検証として
呼べます。DML専用credentialで運用する場合は、migration後の`setup()`は不要です。

```python
with ClickHouseSaver.from_conn_string(dsn) as checkpointer:
    checkpointer.setup()  # migrationは進めず、作成済みschemaを検証
    graph = builder.compile(checkpointer=checkpointer)
```

`setup()`が成功してもAlembic revisionがheadとは限りません。`setup()`を
`alembic upgrade head`の代わりにしないでください。

### 既存の`setup()`環境をbaselineへ登録

すでに`setup()`で表を作成しcheckpointを保存している環境では、backup後に
`adopt_existing_schema()`を実行します。これは両表を厳密に検証し、互換性がある場合だけ
固定baseline revisionをstampします。checkpoint dataは保持されます。

```bash
uv run python - <<'PY'
from langgraph.checkpoint.clickhouse.migration import adopt_existing_schema

adopt_existing_schema()
PY

uv run langgraph-checkpoint-clickhouse-migrate current
uv run langgraph-checkpoint-clickhouse-migrate upgrade head
```

手動の`alembic stamp head`はschemaを検証しないため、既存環境のadoptionには使わないでください。

### custom table prefix

Saverで`table_prefix`を変更する場合はmigrationにも同じ値を渡します。version table名は
prefixから自動導出されるため、異なるprefixの履歴は独立します。

```bash
uv run langgraph-checkpoint-clickhouse-migrate \
  -x table_prefix=my_app_checkpoint \
  upgrade head
```

```python
checkpointer = ClickHouseSaver(client, table_prefix="my_app_checkpoint")
checkpointer.setup()
```

`upgrade`、`current`、`downgrade`のすべてで同じ`table_prefix`を使用してください。
必要な場合だけ`-x version_table=...`で自動導出名をoverrideできます。両値とも英数字と
underscoreのみを使用できます。

### downgrade

initial revisionからbaseへのdowngradeは、writes tableを先に、checkpoints tableを後に削除します。
保存済みのcheckpoint、pending write、履歴はすべて失われます。アプリケーション起動時に自動実行せず、
backupと対象database・prefixを確認したうえで明示的に実行してください。

```bash
uv run langgraph-checkpoint-clickhouse-migrate downgrade base
```

custom prefixの場合はupgrade時と同じ`table_prefix`を指定します。
`version_table`をoverrideした場合はその値も同じにしてください。

## Dockerで全テストを実行

Docker DesktopまたはDocker Engine + Compose v2が必要です。

```bash
docker compose up --build --abort-on-container-exit --exit-code-from tests
```

ClickHouseだけを起動してホスト側からテストする場合は次の通りです。

```bash
docker compose up -d --wait --wait-timeout 120 clickhouse
uv sync --extra test
uv run pytest -q
```

migrationの新規適用、downgrade/re-upgrade、既存dataのadoption、途中失敗からの再実行、
不正schemaの拒否、repository外からのwheel同梱CLIをまとめて確認する場合は次の通りです。

```bash
uv run pytest -q tests/test_migrations.py
```

既定のHTTPポートは `127.0.0.1:18123` です。変更する場合は
`CLICKHOUSE_HTTP_PORT` を指定します。

```bash
CLICKHOUSE_HTTP_PORT=28123 docker compose up -d --wait clickhouse
CLICKHOUSE_PORT=28123 uv run pytest -q
```

停止は `docker compose down` です。テスト用ClickHouseの全データも消す場合のみ
`docker compose down -v` を使ってください。

## 使用例（同期）

```python
from langgraph.checkpoint.clickhouse import ClickHouseSaver

dsn = "http://langgraph:password@clickhouse.example:8123/langgraph"

with ClickHouseSaver.from_conn_string(dsn) as checkpointer:
    checkpointer.setup()  # Alembic適用後の互換schema検証。起動時に1回実行
    graph = builder.compile(checkpointer=checkpointer)
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "hello"}]},
        {"configurable": {"thread_id": "conversation-1"}},
    )
```

既存clientを注入することもできます。

```python
import clickhouse_connect
from langgraph.checkpoint.clickhouse import ClickHouseSaver

client = clickhouse_connect.get_client(
    host="localhost",
    port=18123,
    username="langgraph",
    password="langgraph_test_password",
    database="langgraph_test",
)
checkpointer = ClickHouseSaver(client, table_prefix="my_app_checkpoint")
checkpointer.setup()
```

## 使用例（非同期）

```python
from langgraph.checkpoint.clickhouse import AsyncClickHouseSaver

dsn = "http://langgraph:password@clickhouse.example:8123/langgraph"

async with AsyncClickHouseSaver.from_conn_string(dsn) as checkpointer:
    await checkpointer.setup()
    graph = builder.compile(checkpointer=checkpointer)
    result = await graph.ainvoke(
        {"messages": [{"role": "user", "content": "hello"}]},
        {"configurable": {"thread_id": "conversation-1"}},
    )
```

小さいINSERTをサーバー側でまとめたい場合は、書込み完了を待つ設定を必ず併用します。

```python
checkpointer = AsyncClickHouseSaver(
    client,
    insert_settings={"async_insert": 1, "wait_for_async_insert": 1},
)
```

`wait_for_async_insert=0` はread-after-writeとエラー通知を保証できないため、constructorで
`ValueError` にします。

## 保存設計

`{table_prefix}_checkpoints` と `{table_prefix}_writes` の2表を作成します。
checkpoint全体、metadata、write valueはLangGraph serializerの `(type, bytes)` のまま保存します。
checkpointをチャネル別の複数行に分割しないため、1回の `put()` はClickHouseの単一INSERTで
原子的です。

両表は `ReplacingMergeTree(UInt128 revision)` です。revisionはClickHouseサーバーの
`generateUUIDv7()` で採番します。write表では通常indexのrevisionだけ `bitNot` で反転するため、
同じテーブルを使う複数client間でも通常writeのretryはfirst-write-wins、特殊writeは
last-write-winsになります。ClickHouseの背景merge時期には依存せず、全読取りで `FINAL` を
適用して同一論理キーの正しい行を直ちに取得します。

削除には `lightweight_deletes_sync=2` を明示し、メソッド完了直後のreadから不可視になることを
保証します。TTLは設定しません。古いcheckpointはtime travelやDeltaChannel復元に必要です。

## 運用上の注意

- 同じthreadへの `put` と `delete` を複数プロセスから同時実行する順序はClickHouseだけでは
  線形化できません。単一 `AsyncClickHouseSaver` 内はthread単位で直列化しますが、複数workerでは
  アプリケーション側でもthread単位に直列化してください。`copy_thread` のsource更新も同様です。
- UUIDv7の並行呼出し順序保証は1台のClickHouseサーバー内です。複数書込みreplicaへ同じlogical keyを
  送る構成では、同一keyを常に同じ書込みreplicaへrouteしてください。別server間のUUIDv7順序には
  依存できません。
- native async clientはsession IDなしで作成してください。`get_async_client` の既定値は安全です。
  固定または自動生成session IDは並行queryと両立しないためconstructorで拒否します。
- `delete_for_runs` と `prune(strategy="delete")` は明示的な履歴削除です。生存checkpointが
  参照するDeltaChannel祖先を削除しないよう、呼出し側で対象を管理してください。
- `prune(strategy="keep_latest")` は通常threadでは最新checkpointだけを残します。
  DeltaChannelを検出したthreadでは、復元に必要な直近snapshotまでの祖先を保持します。
  旧版から移行したplain value形式のDelta seedは型情報だけでは自動判定できないため、該当channel名を
  `legacy_delta_channels={"channel_name"}` でsaver作成時に指定してください。
- ClickHouse Cloudや複数replicaでより強いread-your-write保証が必要な場合は、環境に合わせて
  `insert_quorum` やsequential consistencyも検討してください。
- table名へのSQL injectionを防ぐため、`table_prefix` は英数字とunderscoreだけを受け付けます。
- `setup()` は既存表の列型、default式、engine/version、sorting key、primary keyを検査し、互換性が
  なければデータ欠落を起こす前に失敗します。ただしAlembic migrationやbaseline stampは実行しません。
- metadata filterはtyped metadataをPython側で照合するため、非常に長い履歴では `thread_id`、
  namespace、`before` を併用して走査範囲を絞ってください。

## バージョン

- Python 3.10–3.14
- `langgraph-checkpoint >=4.1,<5`（Dockerテストは4.2.0）
- `langgraph 1.2.11`（統合テスト）
- `clickhouse-connect >=1.7.1,<2`
- schema migration: `clickhouse-connect[alembic]`、`Alembic >=1.16,<2`、
  `SQLAlchemy >=1.4.40,<3`
- ClickHouse `>=24.5`（UUIDv7が必要。Docker環境は `26.3.18.32` LTS）

DockerではClickHouseとPython base imageのmulti-architecture manifest digestを固定し、主要Python
テスト依存も `constraints-test.txt` で固定しています。
