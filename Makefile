.PHONY: test test-local lint up down

test:
	docker compose up --build --abort-on-container-exit --exit-code-from tests

test-local:
	python -m pytest -q

lint:
	python -m ruff check src tests
	python -m ruff format --check src tests

up:
	docker compose up -d --wait --wait-timeout 120 clickhouse

down:
	docker compose down --remove-orphans
