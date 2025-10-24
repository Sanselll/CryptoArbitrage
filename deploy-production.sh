#!/bin/bash

# Production Docker Deployment Script for Crypto Arbitrage Platform
# This script handles production deployment with safety checks

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 Crypto Arbitrage - Production Deployment${NC}"
echo "=============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Error: Docker is not running. Please start Docker and try again.${NC}"
    exit 1
fi

# Check if .env.production exists
if [ ! -f .env.production ]; then
    echo -e "${RED}❌ Error: .env.production file not found${NC}"
    echo ""
    echo "To create it:"
    echo "  1. cp .env.production.example .env.production"
    echo "  2. Edit .env.production with your production values"
    echo "  3. IMPORTANT: Change all default passwords and secret keys!"
    exit 1
fi

# Check for default/weak secrets
echo -e "${BLUE}🔒 Checking for default secrets...${NC}"

if grep -q "CHANGE_THIS" .env.production; then
    echo -e "${RED}❌ Error: Found 'CHANGE_THIS' in .env.production${NC}"
    echo -e "${RED}Please update all placeholder values before deploying to production${NC}"
    exit 1
fi

if grep -q "your_secure_password_here" .env.production; then
    echo -e "${RED}❌ Error: Found default password in .env.production${NC}"
    echo -e "${RED}Please set a strong, unique password${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Security check passed${NC}"

# Parse command line arguments
ACTION=${1:-deploy}

case $ACTION in
    deploy)
        echo -e "${YELLOW}⚠️  Production Deployment Warning${NC}"
        echo "This will deploy to production. Make sure you have:"
        echo "  1. ✅ Backed up the database"
        echo "  2. ✅ Reviewed recent code changes"
        echo "  3. ✅ Tested in a staging environment"
        echo "  4. ✅ Updated .env.production with correct values"
        echo ""
        echo -e "${YELLOW}Type 'deploy' to continue or Ctrl+C to cancel:${NC}"
        read -r confirmation

        if [ "$confirmation" != "deploy" ]; then
            echo -e "${YELLOW}Deployment cancelled${NC}"
            exit 0
        fi

        echo -e "${GREEN}📦 Pulling latest images...${NC}"
        docker-compose -f docker-compose.production.yml --env-file .env.production pull

        echo -e "${GREEN}🔄 Restarting services with zero downtime...${NC}"
        docker-compose -f docker-compose.production.yml --env-file .env.production up -d --remove-orphans

        echo ""
        echo -e "${GREEN}✅ Production deployment complete!${NC}"
        ;;

    backup)
        echo -e "${BLUE}💾 Creating database backup...${NC}"
        BACKUP_FILE="backup-$(date +%Y%m%d-%H%M%S).sql.gz"

        docker-compose -f docker-compose.production.yml exec -T postgres \
            pg_dump -U postgres crypto_arbitrage_prod | gzip > "$BACKUP_FILE"

        echo -e "${GREEN}✅ Backup created: $BACKUP_FILE${NC}"
        ;;

    restore)
        if [ -z "$2" ]; then
            echo -e "${RED}❌ Error: Please specify backup file${NC}"
            echo "Usage: $0 restore backup-YYYYMMDD-HHMMSS.sql.gz"
            exit 1
        fi

        BACKUP_FILE=$2

        if [ ! -f "$BACKUP_FILE" ]; then
            echo -e "${RED}❌ Error: Backup file not found: $BACKUP_FILE${NC}"
            exit 1
        fi

        echo -e "${RED}⚠️  WARNING: This will replace the current database!${NC}"
        echo -e "${YELLOW}Type 'restore' to continue:${NC}"
        read -r confirmation

        if [ "$confirmation" != "restore" ]; then
            echo -e "${YELLOW}Restore cancelled${NC}"
            exit 0
        fi

        echo -e "${BLUE}🔄 Stopping backend...${NC}"
        docker-compose -f docker-compose.production.yml stop backend

        echo -e "${BLUE}📥 Restoring database...${NC}"
        gunzip < "$BACKUP_FILE" | docker-compose -f docker-compose.production.yml exec -T postgres \
            psql -U postgres crypto_arbitrage_prod

        echo -e "${BLUE}🔄 Restarting backend...${NC}"
        docker-compose -f docker-compose.production.yml start backend

        echo -e "${GREEN}✅ Database restored${NC}"
        ;;

    logs)
        SERVICE=${2:-}
        if [ -z "$SERVICE" ]; then
            docker-compose -f docker-compose.production.yml logs -f
        else
            docker-compose -f docker-compose.production.yml logs -f "$SERVICE"
        fi
        ;;

    ps)
        docker-compose -f docker-compose.production.yml ps
        ;;

    down)
        echo -e "${RED}⚠️  This will stop all production services${NC}"
        echo -e "${YELLOW}Type 'stop' to continue:${NC}"
        read -r confirmation

        if [ "$confirmation" = "stop" ]; then
            docker-compose -f docker-compose.production.yml down
            echo -e "${GREEN}✅ Services stopped${NC}"
        else
            echo -e "${YELLOW}Cancelled${NC}"
        fi
        ;;

    health)
        echo -e "${BLUE}🏥 Checking service health...${NC}"
        echo ""

        # Check containers
        echo "📦 Container Status:"
        docker-compose -f docker-compose.production.yml ps
        echo ""

        # Check PostgreSQL
        echo -n "🐘 PostgreSQL: "
        if docker-compose -f docker-compose.production.yml exec -T postgres pg_isready -U postgres > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Healthy${NC}"
        else
            echo -e "${RED}❌ Not responding${NC}"
        fi

        # Check Backend API
        echo -n "⚙️  Backend API: "
        if curl -f -s http://localhost:8080/health > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Healthy${NC}"
        else
            echo -e "${RED}❌ Not responding${NC}"
        fi

        # Check Frontend
        echo -n "🌐 Frontend: "
        if curl -f -s http://localhost:80 > /dev/null 2>&1; then
            echo -e "${GREEN}✅ Healthy${NC}"
        else
            echo -e "${RED}❌ Not responding${NC}"
        fi
        ;;

    *)
        echo "Usage: $0 {deploy|backup|restore|logs|ps|down|health}"
        echo ""
        echo "Commands:"
        echo "  deploy         - Deploy to production"
        echo "  backup         - Create database backup"
        echo "  restore <file> - Restore database from backup"
        echo "  logs [service] - View logs (all or specific service)"
        echo "  ps             - Show service status"
        echo "  down           - Stop all services"
        echo "  health         - Check service health"
        exit 1
        ;;
esac
