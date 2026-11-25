.PHONY: install test ablation serve docker-build docker-run

install:
	pip install -r requirements.txt

test:
	PYTHONPATH=src pytest -v

ablation:
	python3 scripts/run_ablation.py

serve:
	PYTHONPATH=src uvicorn biomed_rag.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker build -t biomed-rag-doc-intel:latest .

docker-run:
	docker run --rm -p 8000:8000 biomed-rag-doc-intel:latest
