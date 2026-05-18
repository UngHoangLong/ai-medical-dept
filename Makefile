.PHONY: install lint test train-radiology train-cardiology train-oncology serve docker-train docker-serve

install:
	pip install -e ".[dev,serving,training,segmentation]"

lint:
	ruff check .
	ruff format --check .

format:
	ruff check --fix .
	ruff format .

test:
	pytest tests/ -v

# Fine-tuning (run on VastAI)
train-radiology:
	python training/finetune_radiology.py --config configs/training/radiology.yaml

train-cardiology:
	python training/finetune_cardiology.py --config configs/training/cardiology.yaml

train-oncology:
	python training/finetune_oncology.py --config configs/training/oncology.yaml

train-synthesis:
	python training/train_synthesis.py --config configs/training/synthesis.yaml

# Build FAISS index from embeddings
build-index:
	python scripts/build_index.py --config configs/serving.yaml

# Inference server
serve:
	uvicorn serving.server:app --host $${API_HOST:-0.0.0.0} --port $${API_PORT:-8000} --reload

# Docker (VastAI)
docker-train:
	docker build -f docker/Dockerfile.train -t ai-medical-dept:train .

docker-serve:
	docker build -f docker/Dockerfile.serve -t ai-medical-dept:serve .
