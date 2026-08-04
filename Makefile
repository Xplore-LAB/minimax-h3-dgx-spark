.DEFAULT_GOAL := help
SHELL := /usr/bin/env bash

DATA_ROOT ?=

.PHONY: help init preflight build rebuild gpu-check download download-r2v verify-models up stop restart logs status diagnostics clean-container

help:
	@echo "MiniMax H3 / DGX Spark commands"
	@echo ""
	@echo "  make init DATA_ROOT=/mnt/<4tb>/minimax-h3"
	@echo "  make preflight"
	@echo "  make build"
	@echo "  make gpu-check"
	@echo "  make download"
	@echo "  make up"
	@echo "  make logs"
	@echo "  make status"
	@echo "  make download-r2v"
	@echo "  make diagnostics"

init:
	@if [[ -z "$(DATA_ROOT)" ]]; then \
		echo "ERROR: use make init DATA_ROOT=/mnt/<4tb>/minimax-h3"; exit 2; \
	fi
	@./scripts/init_storage.sh "$(DATA_ROOT)"

preflight:
	@./scripts/preflight.sh

build:
	@docker compose build

rebuild:
	@docker compose build --no-cache

gpu-check:
	@docker compose run --rm --no-deps --entrypoint python comfyui -c \
	  'import platform, torch; print("arch =", platform.machine()); print("torch =", torch.__version__); print("cuda =", torch.version.cuda); print("torch.cuda.is_available() =", torch.cuda.is_available()); assert torch.cuda.is_available(); print("GPU:", torch.cuda.get_device_name(0)); print("capability:", torch.cuda.get_device_capability(0))'

download:
	@./scripts/download_fl2va.sh

download-r2v:
	@./scripts/download_ref2va.sh

verify-models:
	@./scripts/verify_models.sh

up:
	@docker compose up -d
	@echo "ComfyUI: http://127.0.0.1:$${COMFYUI_PORT:-8188}"

stop:
	@docker compose down

restart:
	@docker compose down
	@docker compose up -d

logs:
	@docker compose logs -f --tail=200 comfyui

status:
	@./scripts/status.sh

diagnostics:
	@./scripts/collect_diagnostics.sh

clean-container:
	@docker compose down --remove-orphans
	@docker image rm "local/minimax-h3-comfyui:$${COMFYUI_REF:-v0.30.0}" 2>/dev/null || true
