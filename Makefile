# ============================================================================
# MODELOX QUANT STATION - Makefile
# One command to rule them all
# ============================================================================

.PHONY: help start stop restart build logs clean dev prod shell

# Colors for output
CYAN := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Default target
help:
	@echo ""
	@echo "$(CYAN)╔══════════════════════════════════════════════════════════════════╗$(RESET)"
	@echo "$(CYAN)║$(RESET)     $(GREEN)MODELOX QUANT STATION$(RESET) - Command Reference                  $(CYAN)║$(RESET)"
	@echo "$(CYAN)╠══════════════════════════════════════════════════════════════════╣$(RESET)"
	@echo "$(CYAN)║$(RESET)                                                                  $(CYAN)║$(RESET)"
	@echo "$(CYAN)║$(RESET)   $(YELLOW)make start$(RESET)      Start all services (development mode)       $(CYAN)║$(RESET)"
	@echo "$(CYAN)║$(RESET)   $(YELLOW)make stop$(RESET)       Stop all services                           $(CYAN)║$(RESET)"
	@echo "$(CYAN)║$(RESET)   $(YELLOW)make restart$(RESET)    Restart all services                        $(CYAN)║$(RESET)"
	@echo "$(CYAN)║$(RESET)   $(YELLOW)make build$(RESET)      Rebuild all containers                      $(CYAN)║$(RESET)"
	@echo "$(CYAN)║$(RESET)   $(YELLOW)make logs$(RESET)       View all logs (follow mode)                 $(CYAN)║$(RESET)"
	@echo "$(CYAN)║$(RESET)   $(YELLOW)make clean$(RESET)      Stop and remove all containers/volumes      $(CYAN)║$(RESET)"
	@echo "$(CYAN)║$(RESET)                                                                  $(CYAN)║$(RESET)"
	@echo "$(CYAN)║$(RESET)   $(YELLOW)make dev$(RESET)        Start in development mode (with hot-reload) $(CYAN)║$(RESET)"
	@echo "$(CYAN)║$(RESET)   $(YELLOW)make prod$(RESET)       Start in production mode (optimized)        $(CYAN)║$(RESET)"
	@echo "$(CYAN)║$(RESET)                                                                  $(CYAN)║$(RESET)"
	@echo "$(CYAN)║$(RESET)   $(YELLOW)make shell-backend$(RESET)   Open shell in backend container        $(CYAN)║$(RESET)"
	@echo "$(CYAN)║$(RESET)   $(YELLOW)make shell-frontend$(RESET)  Open shell in frontend container       $(CYAN)║$(RESET)"
	@echo "$(CYAN)║$(RESET)                                                                  $(CYAN)║$(RESET)"
	@echo "$(CYAN)╚══════════════════════════════════════════════════════════════════╝$(RESET)"
	@echo ""

# Start all services
start: dev

# Development mode
dev:
	@echo "$(GREEN)🚀 Starting MODELOX Quant Station (Development)...$(RESET)"
	@docker compose up -d --build
	@echo ""
	@echo "$(GREEN)✅ MODELOX is running!$(RESET)"
	@echo "$(CYAN)   Open: http://localhost:8080$(RESET)"
	@echo ""

# Production mode
prod:
	@echo "$(GREEN)🚀 Starting MODELOX Quant Station (Production)...$(RESET)"
	@docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d --build
	@echo ""
	@echo "$(GREEN)✅ MODELOX is running in production mode!$(RESET)"
	@echo "$(CYAN)   Open: http://localhost:8080$(RESET)"
	@echo ""

# Stop services
stop:
	@echo "$(YELLOW)⏹️  Stopping MODELOX...$(RESET)"
	@docker compose down
	@echo "$(GREEN)✅ Stopped$(RESET)"

# Restart services
restart: stop start

# Build/rebuild containers
build:
	@echo "$(CYAN)🔨 Building containers...$(RESET)"
	@docker compose build --no-cache
	@echo "$(GREEN)✅ Build complete$(RESET)"

# View logs
logs:
	@docker compose logs -f

# Backend logs only
logs-backend:
	@docker compose logs -f backend

# Frontend logs only
logs-frontend:
	@docker compose logs -f frontend

# Clean everything
clean:
	@echo "$(RED)🧹 Cleaning up...$(RESET)"
	@docker compose down -v --remove-orphans
	@docker system prune -f
	@echo "$(GREEN)✅ Cleaned$(RESET)"

# Shell access
shell-backend:
	@docker compose exec backend /bin/bash

shell-frontend:
	@docker compose exec frontend /bin/sh

# Status
status:
	@echo "$(CYAN)📊 Container Status:$(RESET)"
	@docker compose ps

# Install dependencies locally (for IDE support)
install-local:
	@echo "$(CYAN)📦 Installing local dependencies...$(RESET)"
	@cd docker/frontend/app && npm install
	@echo "$(GREEN)✅ Dependencies installed$(RESET)"

# Quick test
test:
	@echo "$(CYAN)🧪 Running health checks...$(RESET)"
	@curl -s http://localhost:8080/api/health | jq . || echo "$(RED)Backend not responding$(RESET)"
	@curl -s http://localhost:8080/health || echo "$(RED)Nginx not responding$(RESET)"
