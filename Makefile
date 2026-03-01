.PHONY: create-venv lock install-deps build install install-editable style test clean

MACHINE := $(shell uname -m)

create-venv:
	uv venv --seed --prompt nasa --python 3.13

lock:
	uv pip compile pyproject.toml -o requirements_${MACHINE}.txt

install-deps:
	python3 -m pip install --upgrade -r requirements_${MACHINE}.txt

build:
	python3 -m build --wheel .

install:
	python3 -m pip install --no-deps --force-reinstall dist/nasa_rag-0.2.0-py3-none-any.whl

install-editable:
	python3 -m pip install --no-deps -e .

style:
	python3 -m ruff format .
	python3 -m ruff check --fix .

test:
	python3 -m pytest -v -rP tests/

run-llm-client:
	python3 -m nasa_rag.llm_client

run-embedding-pipeline:
	python3 -m nasa_rag.embedding_pipeline --data-path ./data_text/ --openai-key ${OPENAI_API_KEY} --chroma-dir ./chroma_db_openai --chunk-size 1024 --chunk-overlap 64 --batch-size 100 --update-mode skip

run-rag-client:
	python3 -m nasa_rag.rag_client

run-ragas-eval:
	python3 -m nasa_rag.ragas_evaluator

run-app:
	python3 -m streamlit run app/chat.py

clean:
	rm *.~