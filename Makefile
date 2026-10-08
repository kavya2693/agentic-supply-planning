.PHONY: run check test lint

run:
	uv run supply-planning --refresh

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy

test:
	uv run pytest -q

check: lint test
