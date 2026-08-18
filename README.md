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

GitHubから直接インストールできます。

```bash
python -m pip install \
  "git+https://github.com/ryomar-kappa/langgraph-checkpoint-clickhouse.git"
```

## Dockerで全テストを実行

Docker DesktopまたはDocker Engine + Compose v2が必要です。

```bash
docker compose up --build --abort-on-container-exit --exit-code-from tests
```

ClickHouseだけを起動してホスト側からテストする場合は次の通りです。

```bash
docker compose up -d --wait --wait-timeout 120 clickhouse
python -m pip install --index-url https://pypi.org/simple -c constraints-test.txt -e '.[test]'
python -m pytest -q
```

既定のHTTPポートは `127.0.0.1:18123` です。変更する場合は
`CLICKHOUSE_HTTP_PORT` を指定します。

```bash
CLICKHOUSE_HTTP_PORT=28123 docker compose up -d --wait clickhouse
CLICKHOUSE_PORT=28123 python -m pytest -q
```

停止は `docker compose down` です。テスト用ClickHouseの全データも消す場合のみ
`docker compose down -v` を使ってください。

## 使用例（同期）

```python
from langgraph.checkpoint.clickhouse import ClickHouseSaver

dsn = "http://langgraph:password@clickhouse.example:8123/langgraph"

with ClickHouseSaver.from_conn_string(dsn) as checkpointer:
    checkpointer.setup()  # table作成と互換schema検証。起動時に1回実行
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
  なければデータ欠落を起こす前に失敗します。
- metadata filterはtyped metadataをPython側で照合するため、非常に長い履歴では `thread_id`、
  namespace、`before` を併用して走査範囲を絞ってください。

## バージョン

- Python 3.10–3.14
- `langgraph-checkpoint >=4.1,<5`（Dockerテストは4.2.0）
- `langgraph 1.2.11`（統合テスト）
- `clickhouse-connect >=1.7.1,<2`
- ClickHouse `>=24.5`（UUIDv7が必要。Docker環境は `26.3.18.32` LTS）

DockerではClickHouseとPython base imageのmulti-architecture manifest digestを固定し、主要Python
テスト依存も `constraints-test.txt` で固定しています。
