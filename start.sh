#!/bin/bash

# ============================================================================
# MODELOX QUANT STATION - Quick Start Script
# Works on: Mac, Linux, Windows (Git Bash/WSL), Cloud, Project IDX
# ============================================================================

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo ""
echo -e "${CYAN}╔══════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║${NC}                                                                  ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}     ${GREEN}███╗   ███╗ ██████╗ ██████╗ ███████╗██╗      ██████╗ ██╗  ██╗${NC}    ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}     ${GREEN}████╗ ████║██╔═══██╗██╔══██╗██╔════╝██║     ██╔═══██╗╚██╗██╔╝${NC}    ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}     ${GREEN}██╔████╔██║██║   ██║██║  ██║█████╗  ██║     ██║   ██║ ╚███╔╝${NC}     ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}     ${GREEN}██║╚██╔╝██║██║   ██║██║  ██║██╔══╝  ██║     ██║   ██║ ██╔██╗${NC}     ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}     ${GREEN}██║ ╚═╝ ██║╚██████╔╝██████╔╝███████╗███████╗╚██████╔╝██╔╝ ██╗${NC}    ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}     ${GREEN}╚═╝     ╚═╝ ╚═════╝ ╚═════╝ ╚══════╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝${NC}    ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}                                                                  ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}                    ${YELLOW}Q U A N T   S T A T I O N${NC}                      ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}                        ${CYAN}v2.0 - Docker Edition${NC}                       ${CYAN}║${NC}"
echo -e "${CYAN}║${NC}                                                                  ${CYAN}║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for Docker
echo -e "${CYAN}[1/4]${NC} Checking Docker..."
if ! command_exists docker; then
    echo -e "${RED}✗ Docker is not installed!${NC}"
    echo ""
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}✗ Docker daemon is not running!${NC}"
    echo ""
    echo "Please start Docker Desktop or the Docker daemon."
    exit 1
fi
echo -e "${GREEN}✓ Docker is ready${NC}"

# Check for Docker Compose
echo -e "${CYAN}[2/4]${NC} Checking Docker Compose..."
if ! command_exists docker-compose && ! docker compose version >/dev/null 2>&1; then
    echo -e "${RED}✗ Docker Compose is not installed!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Docker Compose is ready${NC}"

# Create .env if not exists
echo -e "${CYAN}[3/4]${NC} Checking configuration..."
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${GREEN}✓ Created .env from .env.example${NC}"
    else
        echo "APP_PORT=8080" > .env
        echo "MODELOX_ENV=development" >> .env
        echo -e "${GREEN}✓ Created default .env${NC}"
    fi
else
    echo -e "${GREEN}✓ Configuration exists${NC}"
fi

# Determine which docker compose command to use
DOCKER_COMPOSE="docker compose"
if ! docker compose version >/dev/null 2>&1; then
    DOCKER_COMPOSE="docker-compose"
fi

# Handle arguments
case "${1:-start}" in
    start|up)
        echo -e "${CYAN}[4/4]${NC} Starting MODELOX Quant Station..."
        echo ""
        $DOCKER_COMPOSE up -d --build
        echo ""
        echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║${NC}                                                                  ${GREEN}║${NC}"
        echo -e "${GREEN}║${NC}   ${GREEN}✅ MODELOX is now running!${NC}                                    ${GREEN}║${NC}"
        echo -e "${GREEN}║${NC}                                                                  ${GREEN}║${NC}"
        echo -e "${GREEN}║${NC}   ${CYAN}🌐 Open in browser: http://localhost:8080${NC}                    ${GREEN}║${NC}"
        echo -e "${GREEN}║${NC}                                                                  ${GREEN}║${NC}"
        echo -e "${GREEN}║${NC}   ${YELLOW}📋 Commands:${NC}                                                   ${GREEN}║${NC}"
        echo -e "${GREEN}║${NC}      ${CYAN}./start.sh stop${NC}    - Stop all services                     ${GREEN}║${NC}"
        echo -e "${GREEN}║${NC}      ${CYAN}./start.sh logs${NC}    - View logs                             ${GREEN}║${NC}"
        echo -e "${GREEN}║${NC}      ${CYAN}./start.sh restart${NC} - Restart services                      ${GREEN}║${NC}"
        echo -e "${GREEN}║${NC}                                                                  ${GREEN}║${NC}"
        echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════╝${NC}"
        ;;
    
    stop|down)
        echo -e "${YELLOW}⏹️  Stopping MODELOX...${NC}"
        $DOCKER_COMPOSE down
        echo -e "${GREEN}✓ Stopped${NC}"
        ;;
    
    restart)
        echo -e "${YELLOW}🔄 Restarting MODELOX...${NC}"
        $DOCKER_COMPOSE down
        $DOCKER_COMPOSE up -d --build
        echo -e "${GREEN}✓ Restarted${NC}"
        echo -e "${CYAN}🌐 Open: http://localhost:8080${NC}"
        ;;
    
    logs)
        echo -e "${CYAN}📜 Showing logs (Ctrl+C to exit)...${NC}"
        $DOCKER_COMPOSE logs -f
        ;;
    
    build)
        echo -e "${CYAN}🔨 Rebuilding containers...${NC}"
        $DOCKER_COMPOSE build --no-cache
        echo -e "${GREEN}✓ Build complete${NC}"
        ;;
    
    status)
        echo -e "${CYAN}📊 Status:${NC}"
        $DOCKER_COMPOSE ps
        ;;
    
    clean)
        echo -e "${RED}🧹 Cleaning up everything...${NC}"
        $DOCKER_COMPOSE down -v --remove-orphans
        docker system prune -f
        echo -e "${GREEN}✓ Cleaned${NC}"
        ;;
    
    # Legacy mode - run without Docker
    legacy)
        echo -e "${YELLOW}⚠️  Running in legacy mode (without Docker)...${NC}"
        fuser -k 8000/tcp 2>/dev/null || true
        fuser -k 5173/tcp 2>/dev/null || true
        cd backend && python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ..

        # Start Frontend
        echo "Starting Frontend..."
        cd frontend
        npm run dev -- --host 0.0.0.0 &
        FRONTEND_PID=$!
        cd ..

        echo "Services started."
        echo "Backend PID: $BACKEND_PID"
        echo "Frontend PID: $FRONTEND_PID"
        wait
        ;;
    
    help|--help|-h)
        echo "Usage: ./start.sh [command]"
        echo ""
        echo "Commands:"
        echo "  start    Start with Docker (default)"
        echo "  stop     Stop all services"
        echo "  restart  Restart all services"
        echo "  logs     View logs"
        echo "  build    Rebuild containers"
        echo "  status   Show container status"
        echo "  clean    Remove all containers and volumes"
        echo "  legacy   Run without Docker (old mode)"
        echo "  help     Show this help"
        ;;
    
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo "Use './start.sh help' for available commands"
        exit 1
        ;;
esac
