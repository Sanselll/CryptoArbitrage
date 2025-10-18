# Crypto Arbitrage Platform - Deployment Guide

This guide covers deploying the Crypto Arbitrage platform using Docker and Docker Compose.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start - Local Development](#quick-start---local-development)
- [Production Deployment](#production-deployment)
- [Manual Deployment](#manual-deployment)
- [CI/CD with GitHub Actions](#cicd-with-github-actions)
- [Cloud Platform Guides](#cloud-platform-guides)
- [Database Management](#database-management)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- Git
- For production: Domain name and SSL certificates (optional but recommended)

## Quick Start - Local Development

### 1. Clone and Configure

```bash
# Clone the repository
git clone <your-repo-url>
cd CryptoArbitrage

# Copy environment template
cp .env.example .env

# Edit .env and fill in your configuration
nano .env
```

### 2. Required Environment Variables

Edit `.env` and set at minimum:

```env
# Database
POSTGRES_PASSWORD=your_secure_password_here

# JWT
JWT_SECRET_KEY=your_256_bit_secret_key_minimum_32_characters_long

# Google OAuth
GOOGLE_CLIENT_ID=your_google_client_id
GOOGLE_CLIENT_SECRET=your_google_client_secret

# Encryption
ENCRYPTION_KEY=your_32_character_encryption_key
```

### 3. Start Services

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check status
docker-compose ps
```

### 4. Access the Application

- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8080
- **Swagger UI**: http://localhost:8080/swagger

### 5. Stop Services

```bash
# Stop all services
docker-compose down

# Stop and remove volumes (WARNING: deletes database)
docker-compose down -v
```

## Production Deployment

### 1. Prepare Production Environment

```bash
# Copy production template
cp .env.production.example .env.production

# Edit and fill in production values
nano .env.production
```

**CRITICAL**: Change all default passwords and secret keys!

### 2. Deploy with Docker Compose

```bash
# Build and start production services
docker-compose -f docker-compose.production.yml --env-file .env.production up -d

# Check logs
docker-compose -f docker-compose.production.yml logs -f
```

### 3. Run Database Migrations

```bash
# Access the backend container
docker exec -it crypto-arbitrage-api-prod bash

# Run migrations
dotnet ef database update

# Exit container
exit
```

### 4. Set Up SSL/TLS (Recommended)

For production, use a reverse proxy like Nginx or Caddy:

```nginx
# nginx.conf example
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /api {
        proxy_pass http://localhost:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location /hubs {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

## Manual Deployment

### Building Docker Images Manually

```bash
# Build backend image
cd src/CryptoArbitrage.API
docker build -t crypto-arbitrage-api:latest .

# Build frontend image
cd ../../client
docker build -t crypto-arbitrage-frontend:latest .
```

### Running Individual Containers

```bash
# Start PostgreSQL
docker run -d \
  --name postgres \
  -e POSTGRES_DB=crypto_arbitrage \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=yourpassword \
  -p 5432:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  postgres:16-alpine

# Start Backend
docker run -d \
  --name api \
  -p 8080:8080 \
  -e ConnectionStrings__DefaultConnection="Host=postgres;Database=crypto_arbitrage;Username=postgres;Password=yourpassword" \
  --link postgres \
  crypto-arbitrage-api:latest

# Start Frontend
docker run -d \
  --name frontend \
  -p 3000:80 \
  --link api \
  crypto-arbitrage-frontend:latest
```

## CI/CD with GitHub Actions

The repository includes GitHub Actions workflows for automated builds and deployments.

### Setup GitHub Container Registry

1. Enable GitHub Container Registry in your repository settings
2. Create a Personal Access Token with `write:packages` permission
3. Add secrets to your repository:
   - `GHCR_TOKEN`: Your GitHub token
   - Production environment variables

### Workflows

- **ci.yml**: Runs on every push/PR - builds and tests both backend and frontend
- **docker-build-push.yml**: Builds Docker images and pushes to GitHub Container Registry

### Trigger Manual Deployment

```bash
# Via GitHub UI: Actions tab > Docker Build and Push > Run workflow

# Or via gh CLI
gh workflow run docker-build-push.yml
```

## Cloud Platform Guides

### AWS Deployment (ECS/Fargate)

```bash
# Install AWS CLI and configure
aws configure

# Create ECR repositories
aws ecr create-repository --repository-name crypto-arbitrage-api
aws ecr create-repository --repository-name crypto-arbitrage-frontend

# Tag and push images
docker tag crypto-arbitrage-api:latest <aws-account>.dkr.ecr.<region>.amazonaws.com/crypto-arbitrage-api:latest
docker push <aws-account>.dkr.ecr.<region>.amazonaws.com/crypto-arbitrage-api:latest

# Create ECS cluster, task definitions, and services
# (Use AWS Console or Terraform for full setup)
```

### Azure Container Instances

```bash
# Install Azure CLI
az login

# Create resource group
az group create --name crypto-arbitrage --location eastus

# Create container registry
az acr create --resource-group crypto-arbitrage --name cryptoarbitrage --sku Basic

# Deploy with Azure Container Instances
az container create \
  --resource-group crypto-arbitrage \
  --name crypto-arbitrage-app \
  --image cryptoarbitrage.azurecr.io/api:latest \
  --dns-name-label crypto-arbitrage \
  --ports 80 443
```

### DigitalOcean Droplet

```bash
# SSH into droplet
ssh root@your-droplet-ip

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Clone repository
git clone <your-repo>
cd CryptoArbitrage

# Deploy
cp .env.production.example .env.production
nano .env.production  # Fill in values
docker compose -f docker-compose.production.yml up -d
```

## Database Management

### Backup Database

```bash
# Create backup
docker exec crypto-arbitrage-db-prod pg_dump -U postgres crypto_arbitrage > backup.sql

# Or use automatic backups
docker exec crypto-arbitrage-db-prod pg_dump -U postgres crypto_arbitrage | gzip > backup-$(date +%Y%m%d).sql.gz
```

### Restore Database

```bash
# Stop application
docker-compose -f docker-compose.production.yml stop backend

# Restore
docker exec -i crypto-arbitrage-db-prod psql -U postgres crypto_arbitrage < backup.sql

# Restart application
docker-compose -f docker-compose.production.yml start backend
```

### Migrations

```bash
# Create new migration
cd src/CryptoArbitrage.API
dotnet ef migrations add MigrationName

# Apply migrations
dotnet ef database update

# Or inside Docker
docker exec crypto-arbitrage-api-prod dotnet ef database update
```

## Troubleshooting

### Check Logs

```bash
# All services
docker-compose logs -f

# Specific service
docker-compose logs -f backend
docker-compose logs -f postgres
docker-compose logs -f frontend

# Last 100 lines
docker-compose logs --tail=100 backend
```

### Common Issues

#### Port Already in Use

```bash
# Find process using port
lsof -i :8080

# Kill process
kill -9 <PID>

# Or change port in .env
BACKEND_PORT=8081
```

#### Database Connection Failed

```bash
# Check if PostgreSQL is healthy
docker-compose ps postgres

# Check connection string
docker-compose exec backend env | grep ConnectionStrings

# Test connection
docker-compose exec postgres psql -U postgres -d crypto_arbitrage
```

#### Frontend Can't Reach Backend

- Ensure CORS is configured correctly in `Program.cs`
- Check backend URL in frontend env variables
- Verify network connectivity between containers

### Health Checks

```bash
# Backend health
curl http://localhost:8080/health

# PostgreSQL health
docker-compose exec postgres pg_isready -U postgres

# Frontend health
curl http://localhost:3000
```

### Reset Everything

```bash
# WARNING: This deletes all data!
docker-compose down -v
docker system prune -a
docker-compose up -d --build
```

## Security Considerations

1. **Never commit** `.env` or `.env.production` files
2. Use strong, unique passwords for database and encryption
3. Rotate JWT secret keys regularly
4. Use HTTPS in production
5. Keep Docker images updated
6. Implement rate limiting
7. Regular security audits
8. Monitor logs for suspicious activity

## Performance Tuning

### Database Optimization

```sql
-- Create indexes for better performance
CREATE INDEX idx_funding_rates_symbol_time ON FundingRates(Symbol, RecordedAt);
CREATE INDEX idx_positions_user_status ON Positions(UserId, Status);
```

### Docker Resource Limits

Edit `docker-compose.production.yml` to adjust resources:

```yaml
deploy:
  resources:
    limits:
      cpus: '2'
      memory: 4G
```

## Monitoring

Consider adding monitoring tools:

- Prometheus + Grafana for metrics
- ELK Stack for log aggregation
- Sentry for error tracking
- Uptime monitoring (e.g., UptimeRobot)

## Support

For issues and questions:
- Check logs first
- Review this documentation
- Check GitHub Issues
- Contact support team

## License

[Your License Here]
