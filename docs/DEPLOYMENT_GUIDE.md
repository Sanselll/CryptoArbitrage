# Deployment Guide

This guide covers various deployment scenarios for the Crypto Funding Arbitrage Platform, from local development to production cloud deployments.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Development Deployment](#development-deployment)
3. [Production Deployment Options](#production-deployment-options)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployments](#cloud-deployments)
6. [Environment Configuration](#environment-configuration)
7. [Database Migration](#database-migration)
8. [SSL/TLS Configuration](#ssltls-configuration)
9. [Monitoring & Logging](#monitoring--logging)
10. [Backup & Recovery](#backup--recovery)
11. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Development Environment

- **.NET 8 SDK**: Download from [dotnet.microsoft.com](https://dotnet.microsoft.com/download/dotnet/8.0)
- **Node.js 18+**: Download from [nodejs.org](https://nodejs.org/)
- **Git**: Version control
- **Code Editor**: VS Code, Rider, or Visual Studio

### Production Environment

- **Linux Server** (Ubuntu 22.04 LTS recommended) or Windows Server
- **Nginx or IIS**: Reverse proxy
- **systemd or Windows Service**: Process management
- **SSL Certificate**: Let's Encrypt or commercial certificate
- **Firewall**: Configure appropriate ports

### Exchange Requirements

- **Binance Futures Account**: [futures.binance.com](https://www.binance.com/en/futures)
- **Bybit Account**: [bybit.com](https://www.bybit.com/)
- **API Keys**: With futures trading permissions
- **IP Whitelist**: (Optional) Restrict API access

---

## Development Deployment

### 1. Clone Repository

```bash
cd ~/Projects
git clone <repository-url> CryptoArbitrage
cd CryptoArbitrage
```

### 2. Backend Setup

```bash
cd src/CryptoArbitrage.API

# Restore packages
dotnet restore

# Build
dotnet build

# Run database migrations (if any)
dotnet ef database update

# Run application
dotnet run
```

**Backend URL**: `http://localhost:5000` or `https://localhost:5001`

**Verify**:
```bash
curl http://localhost:5000/health
```

### 3. Frontend Setup

```bash
cd ../../client

# Install dependencies
npm install

# Start development server
npm run dev
```

**Frontend URL**: `http://localhost:5173`

### 4. Configure Exchange API Keys

**Option A: Direct Database Update**

Use a SQLite browser (e.g., DB Browser for SQLite):

1. Open `src/CryptoArbitrage.API/arbitrage.db`
2. Navigate to `Exchanges` table
3. Update `ApiKey` and `ApiSecret` fields
4. Set `IsEnabled` to `1`

**Option B: Via API**

```bash
curl -X PUT "http://localhost:5000/api/exchange/1" \
  -H "Content-Type: application/json" \
  -d '{
    "id": 1,
    "name": "Binance",
    "apiKey": "your-binance-api-key",
    "apiSecret": "your-binance-api-secret",
    "isEnabled": true
  }'
```

### 5. Enable Exchanges

```bash
# Enable Binance
curl -X POST "http://localhost:5000/api/exchange/1/toggle"

# Enable Bybit
curl -X POST "http://localhost:5000/api/exchange/2/toggle"
```

### 6. Verify Operation

- Open browser to `http://localhost:5173`
- Check connection status (green dot in header)
- Verify funding rates are appearing
- Check for arbitrage opportunities

---

## Production Deployment Options

### Option 1: Single VPS (Recommended for Start)

**Pros**:
- Simple setup
- Low cost
- Full control

**Cons**:
- Single point of failure
- Manual scaling

**Recommended Providers**:
- DigitalOcean ($12-24/month)
- Linode ($12-24/month)
- Vultr ($12-24/month)
- AWS Lightsail ($10-20/month)

### Option 2: Cloud Platform (Azure/AWS/GCP)

**Pros**:
- Auto-scaling
- Managed services
- High availability

**Cons**:
- Higher cost
- More complexity

### Option 3: Docker Compose

**Pros**:
- Consistent environment
- Easy deployment
- Version control

**Cons**:
- Requires Docker knowledge
- Additional overhead

---

## Docker Deployment

### 1. Create Dockerfiles

**Backend Dockerfile**:

Create `src/CryptoArbitrage.API/Dockerfile`:

```dockerfile
FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

# Copy csproj and restore
COPY ["CryptoArbitrage.API.csproj", "./"]
RUN dotnet restore

# Copy everything and build
COPY . .
RUN dotnet build -c Release -o /app/build
RUN dotnet publish -c Release -o /app/publish

# Runtime image
FROM mcr.microsoft.com/dotnet/aspnet:8.0
WORKDIR /app
COPY --from=build /app/publish .

# Create data directory for SQLite
RUN mkdir -p /app/data

EXPOSE 8080
ENTRYPOINT ["dotnet", "CryptoArbitrage.API.dll"]
```

**Frontend Dockerfile**:

Create `client/Dockerfile`:

```dockerfile
FROM node:18-alpine AS build
WORKDIR /app

# Copy package files
COPY package*.json ./
RUN npm ci

# Copy source and build
COPY . .
RUN npm run build

# Production image with nginx
FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
CMD ["nginx", "-g", "daemon off;"]
```

**nginx.conf** (in `client/` directory):

```nginx
server {
    listen 80;
    server_name localhost;
    root /usr/share/nginx/html;
    index index.html;

    location / {
        try_files $uri $uri/ /index.html;
    }

    location /api {
        proxy_pass http://backend:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    location /arbitragehub {
        proxy_pass http://backend:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### 2. Create docker-compose.yml

Create `docker-compose.yml` in project root:

```yaml
version: '3.8'

services:
  backend:
    build:
      context: ./src/CryptoArbitrage.API
      dockerfile: Dockerfile
    container_name: arbitrage-backend
    environment:
      - ASPNETCORE_ENVIRONMENT=Production
      - ASPNETCORE_URLS=http://+:8080
      - ConnectionStrings__DefaultConnection=Data Source=/app/data/arbitrage.db
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    ports:
      - "5000:8080"
    restart: unless-stopped
    networks:
      - arbitrage-network

  frontend:
    build:
      context: ./client
      dockerfile: Dockerfile
    container_name: arbitrage-frontend
    ports:
      - "80:80"
    depends_on:
      - backend
    restart: unless-stopped
    networks:
      - arbitrage-network

networks:
  arbitrage-network:
    driver: bridge

volumes:
  arbitrage-data:
  arbitrage-logs:
```

### 3. Build and Run

```bash
# Build images
docker-compose build

# Start services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down

# Restart services
docker-compose restart
```

### 4. Update Configuration

```bash
# Access backend container
docker exec -it arbitrage-backend bash

# Or update via mounted volumes
# Edit files in ./data and ./logs directories
```

---

## Cloud Deployments

### Azure App Service

#### 1. Backend Deployment

```bash
# Install Azure CLI
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

# Login
az login

# Create resource group
az group create --name CryptoArbitrage --location eastus

# Create App Service plan
az appservice plan create \
  --name ArbitragePlan \
  --resource-group CryptoArbitrage \
  --sku B1 \
  --is-linux

# Create web app
az webapp create \
  --resource-group CryptoArbitrage \
  --plan ArbitragePlan \
  --name crypto-arbitrage-api \
  --runtime "DOTNET|8.0"

# Deploy from GitHub (or local)
cd src/CryptoArbitrage.API
az webapp up \
  --resource-group CryptoArbitrage \
  --name crypto-arbitrage-api
```

#### 2. Database Configuration

**Option A: Azure SQL Database**

```bash
az sql server create \
  --name arbitrage-sql \
  --resource-group CryptoArbitrage \
  --location eastus \
  --admin-user sqladmin \
  --admin-password YourPassword123!

az sql db create \
  --resource-group CryptoArbitrage \
  --server arbitrage-sql \
  --name ArbitrageDB \
  --service-objective S0
```

Update connection string:
```bash
az webapp config connection-string set \
  --resource-group CryptoArbitrage \
  --name crypto-arbitrage-api \
  --connection-string-type SQLAzure \
  --settings DefaultConnection="Server=tcp:arbitrage-sql.database.windows.net,1433;Database=ArbitrageDB;User ID=sqladmin;Password=YourPassword123!;Encrypt=True;TrustServerCertificate=False;"
```

**Option B: Keep SQLite with Persistent Storage**

```bash
az webapp config appsettings set \
  --resource-group CryptoArbitrage \
  --name crypto-arbitrage-api \
  --settings WEBSITES_ENABLE_APP_SERVICE_STORAGE=true
```

#### 3. Frontend Deployment (Static Web App)

```bash
# Create Static Web App
az staticwebapp create \
  --name crypto-arbitrage-ui \
  --resource-group CryptoArbitrage \
  --source https://github.com/yourrepo/CryptoArbitrage \
  --location eastus \
  --branch main \
  --app-location "/client" \
  --output-location "dist"
```

#### 4. Configure Environment Variables

```bash
az webapp config appsettings set \
  --resource-group CryptoArbitrage \
  --name crypto-arbitrage-api \
  --settings \
    ASPNETCORE_ENVIRONMENT=Production \
    ArbitrageConfig__MinSpreadPercentage=0.1 \
    ArbitrageConfig__MaxPositionSizeUsd=10000 \
    ArbitrageConfig__AutoExecute=false
```

### AWS (Elastic Beanstalk + S3)

#### 1. Install AWS CLI and EB CLI

```bash
# Install AWS CLI
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Install EB CLI
pip install awsebcli

# Configure
aws configure
```

#### 2. Initialize EB Application

```bash
cd src/CryptoArbitrage.API

# Initialize
eb init -p "64bit Amazon Linux 2023 v2.0.0 running .NET 8" crypto-arbitrage-api

# Create environment
eb create crypto-arbitrage-env \
  --instance-type t3.small \
  --envvars ASPNETCORE_ENVIRONMENT=Production

# Deploy
eb deploy
```

#### 3. Frontend to S3 + CloudFront

```bash
cd ../../client

# Build
npm run build

# Create S3 bucket
aws s3 mb s3://crypto-arbitrage-ui

# Enable static website hosting
aws s3 website s3://crypto-arbitrage-ui \
  --index-document index.html \
  --error-document index.html

# Upload
aws s3 sync dist/ s3://crypto-arbitrage-ui --acl public-read

# Create CloudFront distribution (optional for CDN)
aws cloudfront create-distribution \
  --origin-domain-name crypto-arbitrage-ui.s3.amazonaws.com
```

### Google Cloud Platform (App Engine + Cloud Run)

#### 1. Install gcloud CLI

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

#### 2. Deploy Backend to Cloud Run

Create `app.yaml`:

```yaml
runtime: aspnetcore
env: flex

manual_scaling:
  instances: 1

resources:
  cpu: 1
  memory_gb: 2
  disk_size_gb: 10
```

Deploy:

```bash
cd src/CryptoArbitrage.API

gcloud app deploy

# Or use Cloud Run
gcloud run deploy crypto-arbitrage-api \
  --source . \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

#### 3. Frontend to Firebase Hosting

```bash
cd ../../client

# Install Firebase CLI
npm install -g firebase-tools

# Login
firebase login

# Initialize
firebase init hosting

# Build
npm run build

# Deploy
firebase deploy --only hosting
```

---

## Environment Configuration

### Production appsettings.json

Create `src/CryptoArbitrage.API/appsettings.Production.json`:

```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Warning",
      "Microsoft": "Warning",
      "Microsoft.Hosting.Lifetime": "Information"
    }
  },
  "ConnectionStrings": {
    "DefaultConnection": "Data Source=/app/data/arbitrage.db"
  },
  "AllowedHosts": "*",
  "Kestrel": {
    "Endpoints": {
      "Http": {
        "Url": "http://0.0.0.0:8080"
      }
    }
  }
}
```

### Environment Variables

**Backend**:
```bash
export ASPNETCORE_ENVIRONMENT=Production
export ConnectionStrings__DefaultConnection="Data Source=/app/data/arbitrage.db"
export ArbitrageConfig__MinSpreadPercentage=0.1
export ArbitrageConfig__AutoExecute=false
```

**Frontend** (`.env.production`):
```bash
VITE_API_URL=https://api.yourdomain.com
VITE_SIGNALR_URL=https://api.yourdomain.com/arbitragehub
```

---

## Database Migration

### From SQLite to PostgreSQL

#### 1. Install PostgreSQL Package

```bash
cd src/CryptoArbitrage.API
dotnet add package Npgsql.EntityFrameworkCore.PostgreSQL
```

#### 2. Update Program.cs

```csharp
builder.Services.AddDbContext<ArbitrageDbContext>(options =>
{
    var provider = builder.Configuration["DatabaseProvider"];

    if (provider == "PostgreSQL")
    {
        options.UseNpgsql(
            builder.Configuration.GetConnectionString("PostgreSQL"));
    }
    else
    {
        options.UseSqlite(
            builder.Configuration.GetConnectionString("DefaultConnection"));
    }
});
```

#### 3. Create Migration

```bash
dotnet ef migrations add InitialPostgreSQL \
  --context ArbitrageDbContext \
  --output-dir Data/Migrations/PostgreSQL

dotnet ef database update --context ArbitrageDbContext
```

#### 4. Data Migration Script

```bash
# Export from SQLite
sqlite3 arbitrage.db .dump > dump.sql

# Convert and import to PostgreSQL
# (Manual conversion may be needed for data types)
psql -U postgres -d arbitrage -f dump_converted.sql
```

---

## SSL/TLS Configuration

### Let's Encrypt with Nginx

#### 1. Install Certbot

```bash
sudo apt-get update
sudo apt-get install certbot python3-certbot-nginx
```

#### 2. Obtain Certificate

```bash
sudo certbot --nginx -d yourdomain.com -d www.yourdomain.com
```

#### 3. Auto-Renewal

```bash
# Test renewal
sudo certbot renew --dry-run

# Cron job (already set up by certbot)
sudo crontab -l
```

### Nginx Configuration with SSL

Create `/etc/nginx/sites-available/arbitrage`:

```nginx
# HTTP - redirect to HTTPS
server {
    listen 80;
    server_name yourdomain.com www.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS
server {
    listen 443 ssl http2;
    server_name yourdomain.com www.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Frontend
    location / {
        root /var/www/arbitrage;
        try_files $uri $uri/ /index.html;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # SignalR WebSocket
    location /arbitragehub {
        proxy_pass http://localhost:5000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_cache_bypass $http_upgrade;
        proxy_read_timeout 86400;
    }
}
```

Enable and restart:

```bash
sudo ln -s /etc/nginx/sites-available/arbitrage /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

---

## Monitoring & Logging

### Application Insights (Azure)

```bash
dotnet add package Microsoft.ApplicationInsights.AspNetCore
```

In `Program.cs`:

```csharp
builder.Services.AddApplicationInsightsTelemetry(options =>
{
    options.ConnectionString = builder.Configuration["ApplicationInsights:ConnectionString"];
});
```

### Serilog

```bash
dotnet add package Serilog.AspNetCore
dotnet add package Serilog.Sinks.File
dotnet add package Serilog.Sinks.Console
```

In `Program.cs`:

```csharp
builder.Host.UseSerilog((context, config) =>
{
    config
        .ReadFrom.Configuration(context.Configuration)
        .Enrich.FromLogContext()
        .WriteTo.Console()
        .WriteTo.File("logs/app-.log", rollingInterval: RollingInterval.Day);
});
```

### Health Checks

```bash
# Install packages
dotnet add package Microsoft.Extensions.Diagnostics.HealthChecks.EntityFrameworkCore
```

In `Program.cs`:

```csharp
builder.Services.AddHealthChecks()
    .AddDbContextCheck<ArbitrageDbContext>();

app.MapHealthChecks("/health");
```

Monitor:

```bash
curl http://localhost:5000/health
```

---

## Backup & Recovery

### Database Backup

**SQLite**:

```bash
#!/bin/bash
# backup.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_FILE="/app/data/arbitrage.db"

sqlite3 $DB_FILE ".backup '$BACKUP_DIR/arbitrage_$DATE.db'"

# Keep only last 30 days
find $BACKUP_DIR -name "arbitrage_*.db" -mtime +30 -delete
```

**Cron job**:

```bash
# Daily backup at 2 AM
0 2 * * * /path/to/backup.sh
```

### Application Backup

```bash
#!/bin/bash
# Backup entire application

tar -czf "arbitrage_backup_$(date +%Y%m%d).tar.gz" \
  /app \
  /etc/nginx/sites-available/arbitrage \
  /etc/systemd/system/arbitrage.service
```

### Recovery

```bash
# Stop application
sudo systemctl stop arbitrage

# Restore database
cp /backups/arbitrage_20250115.db /app/data/arbitrage.db

# Restart application
sudo systemctl start arbitrage
```

---

## Troubleshooting

### Backend Won't Start

**Check logs**:
```bash
journalctl -u arbitrage -f
```

**Common issues**:
- Port 5000 already in use: `sudo lsof -i :5000`
- Database permissions: `ls -l /app/data/`
- .NET version: `dotnet --version`

### Frontend Can't Connect

**Check CORS**:
- Verify allowed origins in `Program.cs`
- Check browser console for CORS errors

**Check SignalR**:
```typescript
// Enable detailed logging
.configureLogging(signalR.LogLevel.Debug)
```

### High Memory Usage

**Check process**:
```bash
htop
dotnet-dump collect --process-id <pid>
dotnet-dump analyze <dump-file>
```

### Exchange API Errors

**Check API key permissions**:
- Futures trading enabled
- IP whitelist configured
- Rate limits not exceeded

**Test connection**:
```bash
curl -X GET "https://fapi.binance.com/fapi/v1/time"
```

### Database Locked (SQLite)

**Issue**: Multiple processes accessing same database

**Solution**:
- Use `Journal Mode = WAL` in connection string
- Or migrate to PostgreSQL for concurrent access

---

## Performance Tuning

### Backend

**Program.cs optimizations**:

```csharp
builder.Services.AddDbContext<ArbitrageDbContext>(options =>
{
    options.UseSqlite(connectionString)
           .UseQueryTrackingBehavior(QueryTrackingBehavior.NoTracking) // Read-only queries
           .EnableSensitiveDataLogging(false) // Production
           .EnableDetailedErrors(false); // Production
});

builder.Services.AddSignalR(options =>
{
    options.EnableDetailedErrors = false;
    options.MaximumReceiveMessageSize = 102400; // 100 KB
    options.KeepAliveInterval = TimeSpan.FromSeconds(15);
    options.ClientTimeoutInterval = TimeSpan.FromSeconds(30);
});
```

### Frontend

**Vite build optimization**:

```javascript
// vite.config.ts
export default defineConfig({
  build: {
    minify: 'terser',
    terserOptions: {
      compress: {
        drop_console: true, // Remove console.log in production
      },
    },
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          signalr: ['@microsoft/signalr'],
        },
      },
    },
  },
});
```

### Nginx

```nginx
# Enable gzip compression
gzip on;
gzip_vary on;
gzip_types text/plain text/css application/json application/javascript text/xml application/xml;

# Browser caching
location ~* \.(js|css|png|jpg|jpeg|gif|ico|svg)$ {
    expires 1y;
    add_header Cache-Control "public, immutable";
}
```

---

## Security Checklist

- [ ] API keys stored securely (not in source control)
- [ ] HTTPS enabled with valid certificate
- [ ] CORS configured with specific origins (not *)
- [ ] Rate limiting implemented
- [ ] Authentication/Authorization added (if multi-user)
- [ ] Database backups automated
- [ ] Exchange IP whitelist configured
- [ ] Firewall rules configured
- [ ] Sensitive data encrypted at rest
- [ ] Logs don't contain sensitive information
- [ ] Security headers configured
- [ ] Dependencies up to date
- [ ] Error messages don't expose system details

---

## Maintenance

### Update Application

```bash
# Pull latest code
git pull origin main

# Backend
cd src/CryptoArbitrage.API
dotnet build
sudo systemctl restart arbitrage

# Frontend
cd ../../client
npm install
npm run build
sudo cp -r dist/* /var/www/arbitrage/
```

### Monitor Disk Space

```bash
df -h
du -sh /app/data
```

### Clean Old Logs

```bash
find /app/logs -name "*.log" -mtime +30 -delete
```

### Update Dependencies

```bash
# Backend
dotnet list package --outdated
dotnet add package PackageName

# Frontend
npm outdated
npm update
```

---

This deployment guide should cover most production scenarios. Choose the deployment option that best fits your requirements, budget, and technical expertise.
