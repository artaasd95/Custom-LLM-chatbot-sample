# Custom LLM Chatbot - Makefile
# Provides convenient commands for development, testing, and deployment

.PHONY: help install install-dev install-prod clean test lint format type-check
.PHONY: train serve docker-build docker-run docker-compose-up docker-compose-down
.PHONY: docs docs-serve backup restore monitoring setup-env

# Default target
help:
	@echo "Custom LLM Chatbot - Available Commands:"
	@echo ""
	@echo "Setup and Installation:"
	@echo "  install          Install basic dependencies"
	@echo "  install-dev      Install development dependencies"
	@echo "  install-prod     Install production dependencies"
	@echo "  setup-env        Setup environment and directories"
	@echo ""
	@echo "Development:"
	@echo "  clean            Clean build artifacts and cache"
	@echo "  test             Run tests"
	@echo "  lint             Run linting checks"
	@echo "  format           Format code with black"
	@echo "  type-check       Run type checking with mypy"
	@echo "  pre-commit       Run pre-commit hooks"
	@echo ""
	@echo "Training and Serving:"
	@echo "  train            Start training with default config"
	@echo "  train-custom     Start training with custom config"
	@echo "  serve            Start serving with default config"
	@echo "  serve-api        Start API server"
	@echo "  serve-vllm       Start vLLM server"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build     Build Docker images"
	@echo "  docker-run       Run Docker container"
	@echo "  docker-dev       Run development container"
	@echo "  docker-compose-up    Start all services with docker-compose"
	@echo "  docker-compose-down  Stop all services"
	@echo ""
	@echo "Documentation:"
	@echo "  docs             Build documentation"
	@echo "  docs-serve       Serve documentation locally"
	@echo ""
	@echo "Utilities:"
	@echo "  backup           Backup models and data"
	@echo "  restore          Restore from backup"
	@echo "  monitoring       Start monitoring stack"
	@echo "  example          Run example usage"

# Variables
PYTHON := python
PIP := pip
DOCKER := docker
DOCKER_COMPOSE := docker-compose
CONFIG_FILE := config.yaml
BACKUP_DIR := ./backups
DATE := $(shell date +%Y%m%d_%H%M%S)

# Setup and Installation
install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .[dev]
	$(PIP) install pre-commit
	pre-commit install

install-prod:
	$(PIP) install -r requirements.txt
	$(PIP) install -e .[serving,optimized]

setup-env:
	@echo "Setting up environment..."
	mkdir -p data models outputs logs backups
	mkdir -p data/train data/eval data/test
	mkdir -p models/checkpoints models/final
	mkdir -p outputs/experiments outputs/results
	mkdir -p logs/training logs/serving logs/monitoring
	mkdir -p backups/models backups/data backups/configs
	@if [ ! -f .env ]; then cp .env.example .env; echo "Created .env file from template"; fi
	@echo "Environment setup complete!"

# Development
clean:
	@echo "Cleaning build artifacts and cache..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf build/ dist/ .coverage htmlcov/
	@echo "Clean complete!"

test:
	@echo "Running tests..."
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

lint:
	@echo "Running linting checks..."
	flake8 src/ tests/ --max-line-length=88 --extend-ignore=E203,W503
	black --check src/ tests/
	mypy src/

format:
	@echo "Formatting code..."
	black src/ tests/ *.py
	isort src/ tests/ *.py

type-check:
	@echo "Running type checks..."
	mypy src/ --strict

pre-commit:
	@echo "Running pre-commit hooks..."
	pre-commit run --all-files

# Training and Serving
train:
	@echo "Starting training with default config..."
	$(PYTHON) train.py --config $(CONFIG_FILE)

train-custom:
	@echo "Starting training with custom config..."
	@read -p "Enter config file path: " config; \
	$(PYTHON) train.py --config $$config

train-resume:
	@echo "Resuming training from checkpoint..."
	@read -p "Enter checkpoint path: " checkpoint; \
	$(PYTHON) train.py --config $(CONFIG_FILE) --resume $$checkpoint

serve:
	@echo "Starting serving with default config..."
	$(PYTHON) serve.py --config $(CONFIG_FILE)

serve-api:
	@echo "Starting API server..."
	$(PYTHON) serve.py --config $(CONFIG_FILE) --backend pytorch --port 8000

serve-vllm:
	@echo "Starting vLLM server..."
	$(PYTHON) serve.py --config $(CONFIG_FILE) --backend vllm --port 8001

example:
	@echo "Running example usage..."
	$(PYTHON) example_usage.py

# Docker
docker-build:
	@echo "Building Docker images..."
	$(DOCKER) build -t custom-llm-chatbot:latest .
	$(DOCKER) build -t custom-llm-chatbot:training --target training .
	$(DOCKER) build -t custom-llm-chatbot:serving --target serving .
	$(DOCKER) build -t custom-llm-chatbot:development --target development .

docker-run:
	@echo "Running Docker container..."
	$(DOCKER) run -it --rm --gpus all \
		-v $(PWD)/models:/app/models \
		-v $(PWD)/data:/app/data \
		-v $(PWD)/config.yaml:/app/config.yaml \
		-p 8000:8000 \
		custom-llm-chatbot:serving

docker-dev:
	@echo "Running development container..."
	$(DOCKER) run -it --rm --gpus all \
		-v $(PWD):/app \
		-p 8000:8000 -p 8888:8888 \
		custom-llm-chatbot:development bash

docker-compose-up:
	@echo "Starting all services with docker-compose..."
	$(DOCKER_COMPOSE) up -d

docker-compose-down:
	@echo "Stopping all services..."
	$(DOCKER_COMPOSE) down

docker-compose-logs:
	@echo "Showing service logs..."
	$(DOCKER_COMPOSE) logs -f

# Monitoring
monitoring:
	@echo "Starting monitoring stack..."
	$(DOCKER_COMPOSE) --profile monitoring up -d
	@echo "Prometheus: http://localhost:9090"
	@echo "Grafana: http://localhost:3000 (admin/admin)"

monitoring-down:
	@echo "Stopping monitoring stack..."
	$(DOCKER_COMPOSE) --profile monitoring down

# Documentation
docs:
	@echo "Building documentation..."
	mkdir -p docs/build
	@if command -v sphinx-build >/dev/null 2>&1; then \
		sphinx-build -b html docs/source docs/build; \
	else \
		echo "Sphinx not installed. Install with: pip install sphinx"; \
	fi

docs-serve:
	@echo "Serving documentation locally..."
	@if [ -d "docs/build" ]; then \
		cd docs/build && $(PYTHON) -m http.server 8080; \
	else \
		echo "Documentation not built. Run 'make docs' first."; \
	fi

# Backup and Restore
backup:
	@echo "Creating backup..."
	mkdir -p $(BACKUP_DIR)/$(DATE)
	@if [ -d "models" ]; then cp -r models $(BACKUP_DIR)/$(DATE)/; fi
	@if [ -d "data" ]; then cp -r data $(BACKUP_DIR)/$(DATE)/; fi
	@if [ -d "outputs" ]; then cp -r outputs $(BACKUP_DIR)/$(DATE)/; fi
	cp $(CONFIG_FILE) $(BACKUP_DIR)/$(DATE)/ 2>/dev/null || true
	cp .env $(BACKUP_DIR)/$(DATE)/ 2>/dev/null || true
	@echo "Backup created: $(BACKUP_DIR)/$(DATE)"

restore:
	@echo "Available backups:"
	@ls -la $(BACKUP_DIR)/ 2>/dev/null || echo "No backups found"
	@read -p "Enter backup date (YYYYMMDD_HHMMSS): " backup_date; \
	if [ -d "$(BACKUP_DIR)/$$backup_date" ]; then \
		cp -r $(BACKUP_DIR)/$$backup_date/* .; \
		echo "Restored from backup: $$backup_date"; \
	else \
		echo "Backup not found: $$backup_date"; \
	fi

# Database operations
db-setup:
	@echo "Setting up database..."
	$(DOCKER_COMPOSE) --profile database up -d postgres
	@echo "Waiting for database to be ready..."
	sleep 10
	@echo "Database setup complete!"

db-migrate:
	@echo "Running database migrations..."
	# Add your migration commands here
	@echo "Migrations complete!"

db-backup:
	@echo "Backing up database..."
	mkdir -p $(BACKUP_DIR)/db
	$(DOCKER_COMPOSE) exec postgres pg_dump -U llm_user llm_experiments > $(BACKUP_DIR)/db/backup_$(DATE).sql
	@echo "Database backup created: $(BACKUP_DIR)/db/backup_$(DATE).sql"

# Performance testing
benchmark:
	@echo "Running performance benchmarks..."
	$(PYTHON) -m pytest benchmarks/ -v --benchmark-only

load-test:
	@echo "Running load tests..."
	@if command -v locust >/dev/null 2>&1; then \
		locust -f tests/load_test.py --host=http://localhost:8000; \
	else \
		echo "Locust not installed. Install with: pip install locust"; \
	fi

# Security
security-scan:
	@echo "Running security scans..."
	@if command -v bandit >/dev/null 2>&1; then \
		bandit -r src/; \
	else \
		echo "Bandit not installed. Install with: pip install bandit"; \
	fi
	@if command -v safety >/dev/null 2>&1; then \
		safety check; \
	else \
		echo "Safety not installed. Install with: pip install safety"; \
	fi

# Deployment
deploy-staging:
	@echo "Deploying to staging..."
	# Add your staging deployment commands here
	@echo "Staging deployment complete!"

deploy-prod:
	@echo "Deploying to production..."
	@read -p "Are you sure you want to deploy to production? (y/N): " confirm; \
	if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
		echo "Deploying to production..."; \
		# Add your production deployment commands here \
		echo "Production deployment complete!"; \
	else \
		echo "Deployment cancelled."; \
	fi

# Health checks
health-check:
	@echo "Running health checks..."
	@if curl -f http://localhost:8000/health >/dev/null 2>&1; then \
		echo "✓ API server is healthy"; \
	else \
		echo "✗ API server is not responding"; \
	fi

# Quick start
quick-start: setup-env install
	@echo "Quick start setup complete!"
	@echo "Next steps:"
	@echo "1. Edit .env file with your configuration"
	@echo "2. Edit config.yaml with your model settings"
	@echo "3. Add your training data to data/ directory"
	@echo "4. Run 'make train' to start training"
	@echo "5. Run 'make serve' to start serving"

# Development workflow
dev-setup: install-dev setup-env
	@echo "Development environment setup complete!"
	@echo "Run 'make test' to verify everything works"

dev-check: lint type-check test
	@echo "All development checks passed!"

# Production workflow
prod-setup: install-prod setup-env
	@echo "Production environment setup complete!"

# All-in-one commands
full-test: clean lint type-check test
	@echo "Full test suite completed!"

full-setup: setup-env install-dev docs
	@echo "Full setup completed!"