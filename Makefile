.DEFAULT_GOAL := help

.PHONY: help install run debug lint lint-strict clean

help:
	@sed -n 's/^\.PHONY: *//p' $(MAKEFILE_LIST) | tr ' ' '\n' | sort

install:
	uv sync

run:
	uv run python -m src

debug:
	uv run python -m pdb -m src

lint:
	uv run flake8 .
	uv run mypy .

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

clean:
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .mypy_cache .pytest_cache data/output
