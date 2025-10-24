#!/bin/bash

# Local Docker Deployment Script for Crypto Arbitrage Platform
# This script sets up and runs the application locally using Docker Compose

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Crypto Arbitrage - Local Deployment${NC}"
echo "=========================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if .env file exists
if [ ! -f .env ]; then
    echo -e "${YELLOW}⚠️  Warning: .env file not found${NC}"
    echo "Creating .env from .env.example..."

    if [ -f .env.example ]; then
        cp .env.example .env
        echo -e "${YELLOW}📝 Please edit .env file with your configuration before deploying${NC}"
        echo -e "${YELLOW}Press Enter to continue or Ctrl+C to cancel...${NC}"
        read
    else
        echo -e "${RED}❌ Error: .env.example not found${NC}"
        exit 1
    fi
fi

# Parse command line arguments
ACTION=${1:-up}
BUILD_FLAG=""

case $ACTION in
    up)
        echo -e "${GREEN}📦 Starting services...${NC}"
        ;;
    build)
        echo -e "${GREEN}🔨 Building and starting services...${NC}"
        BUILD_FLAG="--build"
        ACTION="up"
        ;;
    down)
        echo -e "${YELLOW}🛑 Stopping services...${NC}"
        ;;
    restart)
        echo -e "${GREEN}🔄 Restarting services...${NC}"
        docker-compose down
        ACTION="up"
        ;;
    logs)
        echo -e "${GREEN}📋 Showing logs...${NC}"
        docker-compose logs -f
        exit 0
        ;;
    ps)
        echo -e "${GREEN}📊 Service status:${NC}"
        docker-compose ps
        exit 0
        ;;
    clean)
        echo -e "${RED}🧹 Cleaning up (WARNING: This will remove volumes and delete data)${NC}"
        echo -e "${RED}Are you sure? Type 'yes' to continue:${NC}"
        read -r confirmation
        if [ "$confirmation" = "yes" ]; then
            docker-compose down -v
            docker system prune -f
            echo -e "${GREEN}✅ Cleanup complete${NC}"
        else
            echo -e "${YELLOW}Cancelled${NC}"
        fi
        exit 0
        ;;
    *)
        echo -e "${RED}Unknown command: $ACTION${NC}"
        echo "Usage: $0 {up|build|down|restart|logs|ps|clean}"
        exit 1
        ;;
esac

# Execute docker-compose command
docker-compose $ACTION -d $BUILD_FLAG

if [ "$ACTION" = "up" ]; then
    echo ""
    echo -e "${GREEN}✅ Services started successfully!${NC}"
    echo ""
    echo "📱 Access the application:"
    echo -e "  Frontend:  ${GREEN}http://localhost:3000${NC}"
    echo -e "  Backend:   ${GREEN}http://localhost:8080${NC}"
    echo -e "  Swagger:   ${GREEN}http://localhost:8080/swagger${NC}"
    echo ""
    echo "📊 Check status with: ./deploy-local.sh ps"
    echo "📋 View logs with:    ./deploy-local.sh logs"
    echo "🛑 Stop services with: ./deploy-local.sh down"
    echo ""

    # Wait a moment for services to start
    sleep 3

    # Check service health
    echo "🔍 Checking service health..."

    if docker-compose ps | grep -q "Up"; then
        echo -e "${GREEN}✅ Containers are running${NC}"

        # Check PostgreSQL
        if docker-compose exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
            echo -e "${GREEN}✅ PostgreSQL is ready${NC}"
        else
            echo -e "${YELLOW}⚠️  PostgreSQL is starting...${NC}"
        fi
    else
        echo -e "${RED}❌ Some containers failed to start${NC}"
        echo "Run './deploy-local.sh logs' to see details"
    fi
fi
