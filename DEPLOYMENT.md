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

## CI/CD with GitHub Actions and DigitalOcean

### Complete Setup Guide

#### Step 1: Create DigitalOcean Droplet

1. **Create Droplet**:
   - Go to DigitalOcean Console → Create → Droplets
   - Choose: Ubuntu 22.04 LTS, $6+/month plan
   - Select region closest to you
   - Add SSH key (create one if needed)
   - Name: `crypto-arbitrage-prod`
   - Click Create

2. **Note your droplet IP** - you'll need this for GitHub secrets

3. **Initial SSH Setup**:
   ```bash
   ssh root@your_droplet_ip

   # Update system
   sudo apt update && sudo apt upgrade -y

   # Install Docker
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh

   # Install Docker Compose
   sudo apt-get install -y docker-compose-plugin

   # Add user to docker group
   sudo usermod -aG docker $USER

   # Create deploy user (more secure than root)
   sudo useradd -m -s /bin/bash deploy
   sudo usermod -aG docker deploy
   sudo usermod -aG sudo deploy

   # Set password for deploy user
   sudo passwd deploy
   ```

4. **Setup SSH for deployment**:
   ```bash
   # Create .ssh directory for deploy user
   sudo -u deploy mkdir -p /home/deploy/.ssh

   # Copy your public key
   cat ~/.ssh/id_rsa.pub | sudo tee -a /home/deploy/.ssh/authorized_keys
   sudo chmod 700 /home/deploy/.ssh
   sudo chmod 600 /home/deploy/.ssh/authorized_keys
   sudo chown -R deploy:deploy /home/deploy/.ssh
   ```

#### Step 2: Setup GitHub Repository Secrets

1. **Go to GitHub Settings** → Secrets and variables → Actions → New repository secret

2. **Add these secrets**:
   ```
   DEPLOY_HOST=your_droplet_ip
   DEPLOY_USER=deploy
   DEPLOY_KEY=<your-ssh-private-key-content>

   # Production Environment Variables
   PROD_POSTGRES_PASSWORD=<strong-random-password>
   PROD_JWT_SECRET=<32-char-random-string>
   PROD_ENCRYPTION_KEY=<32-char-random-string>
   PROD_GOOGLE_CLIENT_ID=<your-google-oauth-id>
   PROD_GOOGLE_CLIENT_SECRET=<your-google-oauth-secret>
   PROD_DOMAIN=your-domain.com

   # GitHub Container Registry
   GHCR_TOKEN=<github-pat-token>
   ```

3. **Get SSH Private Key**:
   ```bash
   # On your local machine
   cat ~/.ssh/id_rsa
   # Copy entire key including -----BEGIN... and -----END...
   ```

#### Step 3: Create GitHub Actions Workflows

**File: `.github/workflows/ci.yml`**
```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main, develop]

jobs:
  build:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup .NET
        uses: actions/setup-dotnet@v4
        with:
          dotnet-version: '8.0.x'

      - name: Restore dependencies
        run: |
          cd src/CryptoArbitrage.API
          dotnet restore

      - name: Build
        run: |
          cd src/CryptoArbitrage.API
          dotnet build --no-restore --configuration Release

      - name: Run tests
        run: |
          cd src/CryptoArbitrage.API
          dotnet test --no-build --verbosity normal

  frontend:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Setup Node.js
        uses: actions/setup-node@v4
        with:
          node-version: '18'
          cache: 'npm'
          cache-dependency-path: 'client/package-lock.json'

      - name: Install dependencies
        run: |
          cd client
          npm install

      - name: Lint
        run: |
          cd client
          npm run lint

      - name: Build
        run: |
          cd client
          npm run build
```

**File: `.github/workflows/deploy.yml`**
```yaml
name: Build and Deploy

on:
  push:
    branches: [main]
  workflow_dispatch:

jobs:
  build-and-push:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write

    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Log in to GitHub Container Registry
        uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GHCR_TOKEN }}

      - name: Build and push backend image
        uses: docker/build-push-action@v5
        with:
          context: ./src/CryptoArbitrage.API
          push: true
          tags: |
            ghcr.io/${{ github.repository }}/api:latest
            ghcr.io/${{ github.repository }}/api:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

      - name: Build and push frontend image
        uses: docker/build-push-action@v5
        with:
          context: ./client
          push: true
          tags: |
            ghcr.io/${{ github.repository }}/web:latest
            ghcr.io/${{ github.repository }}/web:${{ github.sha }}
          cache-from: type=gha
          cache-to: type=gha,mode=max

  deploy:
    needs: build-and-push
    runs-on: ubuntu-latest
    if: success()

    steps:
      - uses: actions/checkout@v4

      - name: Deploy to DigitalOcean
        env:
          DEPLOY_KEY: ${{ secrets.DEPLOY_KEY }}
          DEPLOY_HOST: ${{ secrets.DEPLOY_HOST }}
          DEPLOY_USER: ${{ secrets.DEPLOY_USER }}
        run: |
          # Create SSH key file
          mkdir -p ~/.ssh
          echo "$DEPLOY_KEY" > ~/.ssh/deploy_key
          chmod 600 ~/.ssh/deploy_key

          # Add host to known_hosts
          ssh-keyscan -H $DEPLOY_HOST >> ~/.ssh/known_hosts 2>/dev/null

          # Deploy script
          ssh -i ~/.ssh/deploy_key $DEPLOY_USER@$DEPLOY_HOST << 'EOF'
          set -e

          # Create app directory
          APP_DIR="/home/deploy/crypto-arbitrage"
          mkdir -p $APP_DIR
          cd $APP_DIR

          # Clone or pull latest code
          if [ -d .git ]; then
            git pull origin main
          else
            git clone $GITHUB_SERVER_URL/$GITHUB_REPOSITORY.git .
          fi

          # Create .env.production from template
          if [ ! -f .env.production ]; then
            cp .env.production.example .env.production
          fi

          # Update environment variables
          cat > .env.production << 'ENVEOF'
          POSTGRES_DB=crypto_arbitrage
          POSTGRES_USER=postgres
          POSTGRES_PASSWORD=${{ secrets.PROD_POSTGRES_PASSWORD }}
          BACKEND_PORT=8080
          FRONTEND_PORT=3000
          JWT_SECRET_KEY=${{ secrets.PROD_JWT_SECRET }}
          ENCRYPTION_KEY=${{ secrets.PROD_ENCRYPTION_KEY }}
          GOOGLE_CLIENT_ID=${{ secrets.PROD_GOOGLE_CLIENT_ID }}
          GOOGLE_CLIENT_SECRET=${{ secrets.PROD_GOOGLE_CLIENT_SECRET }}
          ASPNETCORE_ENVIRONMENT=Production
          IMAGE_TAG=${{ github.sha }}
          ENVEOF

          # Pull latest images
          docker pull ghcr.io/${{ github.repository }}/api:latest
          docker pull ghcr.io/${{ github.repository }}/web:latest

          # Update docker-compose with latest images
          sed -i 's|image:.*api:.*|image: ghcr.io/${{ github.repository }}/api:${{ github.sha }}|' docker-compose.production.yml
          sed -i 's|image:.*web:.*|image: ghcr.io/${{ github.repository }}/web:${{ github.sha }}|' docker-compose.production.yml

          # Stop old containers
          docker compose -f docker-compose.production.yml down || true

          # Start new containers
          docker compose -f docker-compose.production.yml --env-file .env.production up -d

          # Wait for services to be ready
          sleep 10

          # Run database migrations
          docker compose -f docker-compose.production.yml exec -T api dotnet ef database update || true

          # Verify deployment
          docker compose -f docker-compose.production.yml ps
          EOF

          # Clean up SSH key
          rm ~/.ssh/deploy_key
```

**File: `.github/workflows/scheduled-backup.yml`**
```yaml
name: Scheduled Database Backup

on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM UTC daily
  workflow_dispatch:

jobs:
  backup:
    runs-on: ubuntu-latest
    env:
      DEPLOY_KEY: ${{ secrets.DEPLOY_KEY }}
      DEPLOY_HOST: ${{ secrets.DEPLOY_HOST }}
      DEPLOY_USER: ${{ secrets.DEPLOY_USER }}

    steps:
      - name: Create SSH key
        run: |
          mkdir -p ~/.ssh
          echo "$DEPLOY_KEY" > ~/.ssh/deploy_key
          chmod 600 ~/.ssh/deploy_key
          ssh-keyscan -H $DEPLOY_HOST >> ~/.ssh/known_hosts 2>/dev/null

      - name: Backup database
        run: |
          ssh -i ~/.ssh/deploy_key $DEPLOY_USER@$DEPLOY_HOST << 'EOF'
          cd /home/deploy/crypto-arbitrage
          BACKUP_FILE="backups/backup-$(date +%Y%m%d_%H%M%S).sql.gz"
          mkdir -p backups
          docker compose -f docker-compose.production.yml exec -T postgres pg_dump -U postgres crypto_arbitrage | gzip > $BACKUP_FILE
          echo "Backup created: $BACKUP_FILE"

          # Keep only last 7 backups
          ls -t backups/backup-*.sql.gz | tail -n +8 | xargs -r rm
          EOF
```

#### Step 4: Create `.env.production.example`

```env
# Production Database
POSTGRES_DB=crypto_arbitrage
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your_secure_password_here
POSTGRES_PORT=5432

# Backend
BACKEND_PORT=8080
ASPNETCORE_ENVIRONMENT=Production

# Frontend
FRONTEND_PORT=3000

# JWT
JWT_SECRET_KEY=your_256_bit_secret_key_minimum_32_characters_long
JWT_ISSUER=CryptoArbitrage
JWT_AUDIENCE=CryptoArbitrageClient

# Google OAuth
GOOGLE_CLIENT_ID=your_google_client_id_here
GOOGLE_CLIENT_SECRET=your_google_client_secret_here

# Encryption
ENCRYPTION_KEY=your_32_character_encryption_key

# GitHub Container Registry
IMAGE_TAG=latest
```

#### Step 5: Update `docker-compose.production.yml`

Add image tags that reference the environment variable:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "${POSTGRES_PORT}:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
      interval: 10s
      timeout: 5s
      retries: 5

  api:
    image: ghcr.io/${GITHUB_REPOSITORY}/api:${IMAGE_TAG}
    environment:
      ConnectionStrings__DefaultConnection: Host=postgres;Database=${POSTGRES_DB};Username=${POSTGRES_USER};Password=${POSTGRES_PASSWORD};
      JWT_SECRET_KEY: ${JWT_SECRET_KEY}
      ENCRYPTION_KEY: ${ENCRYPTION_KEY}
      GOOGLE_CLIENT_ID: ${GOOGLE_CLIENT_ID}
      GOOGLE_CLIENT_SECRET: ${GOOGLE_CLIENT_SECRET}
      ASPNETCORE_ENVIRONMENT: ${ASPNETCORE_ENVIRONMENT}
    depends_on:
      postgres:
        condition: service_healthy
    ports:
      - "${BACKEND_PORT}:8080"
    restart: unless-stopped

  web:
    image: ghcr.io/${GITHUB_REPOSITORY}/web:${IMAGE_TAG}
    environment:
      VITE_API_BASE_URL: https://${PROD_DOMAIN}/api
    depends_on:
      - api
    ports:
      - "${FRONTEND_PORT}:80"
    restart: unless-stopped

volumes:
  postgres_data:
```

### Workflows Overview

- **ci.yml**: Runs on every push and PR to main/develop - builds and tests both backend and frontend
- **deploy.yml**: Triggered on push to main - builds Docker images, pushes to GHCR, and deploys to DigitalOcean
- **scheduled-backup.yml**: Daily backups at 2 AM UTC

### Manual Deployment Trigger

```bash
# Via GitHub UI: Actions tab > Build and Deploy > Run workflow

# Or via gh CLI
gh workflow run deploy.yml
```

### Monitoring Deployments

1. **Check GitHub Actions logs** in repository Actions tab
2. **SSH into droplet** and check container logs:
   ```bash
   ssh deploy@your_droplet_ip
   cd /home/deploy/crypto-arbitrage
   docker compose -f docker-compose.production.yml logs -f
   ```
3. **Health checks**:
   ```bash
   curl https://your-domain.com/health
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
