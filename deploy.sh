#!/bin/bash
set -e

APP_DIR="/home/deploy/crypto-arbitrage"
mkdir -p "$APP_DIR"
cd "$APP_DIR"

# Clone or update repo
if [ ! -d .git ]; then
  git clone "$1" .
fi

git fetch origin
git checkout develop
git pull origin develop

# Extract owner name (lowercase)
REPO_OWNER=$(echo "$GITHUB_REPOSITORY" | cut -d'/' -f1 | tr '[:upper:]' '[:lower:]')

# Create/update .env.production
cat > .env.production << EOF
POSTGRES_DB=crypto_arbitrage
POSTGRES_USER=postgres
POSTGRES_PASSWORD=$PROD_POSTGRES_PASSWORD
BACKEND_PORT=8080
FRONTEND_PORT=3000
JWT_SECRET_KEY=$PROD_JWT_SECRET
JWT_ISSUER=CryptoArbitrage
JWT_AUDIENCE=CryptoArbitrageClient
ENCRYPTION_KEY=$PROD_ENCRYPTION_KEY
GOOGLE_CLIENT_ID=$PROD_GOOGLE_CLIENT_ID
GOOGLE_CLIENT_SECRET=$PROD_GOOGLE_CLIENT_SECRET
PROD_DOMAIN=$PROD_DOMAIN
ASPNETCORE_ENVIRONMENT=Production
IMAGE_TAG=$GITHUB_SHA
GITHUB_REPOSITORY_OWNER=$REPO_OWNER
EOF

# Extract owner and pull images
REPO_OWNER=$(echo "$GITHUB_REPOSITORY" | cut -d'/' -f1 | tr '[:upper:]' '[:lower:]')
docker pull ghcr.io/$REPO_OWNER/crypto-arbitrage/api:latest || true
docker pull ghcr.io/$REPO_OWNER/crypto-arbitrage/web:latest || true

# Deploy
docker compose -f docker-compose.production.yml down || true
docker compose -f docker-compose.production.yml --env-file .env.production up -d

sleep 10

# Run migrations
docker compose -f docker-compose.production.yml exec -T api dotnet ef database update || true

# Show status
docker compose -f docker-compose.production.yml ps
