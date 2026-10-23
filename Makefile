.PHONY: install dev lint format typecheck test check ingest serve ui eval

install:
	pip install -e ".[local,embeddings,ui]"

dev:
	pip install -e ".[dev]"

lint:
	ruff check src tests ui
	ruff format --check src tests ui

format:
	ruff format src tests ui
	ruff check --fix src tests ui

typecheck:
	mypy src

test:
	pytest --cov=jolt --cov-report=term-missing

check: lint typecheck test

ingest:
	jolt ingest data/sample

serve:
	jolt serve --reload

ui:
	streamlit run ui/streamlit_app.py

eval:
	jolt eval eval/questions.jsonl --no-web
