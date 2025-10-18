# GitHub CI/CD & DigitalOcean Deployment Setup

Complete automated deployment pipeline for Crypto Arbitrage Platform.

## Table of Contents

- [Quick Start (5 minutes)](#quick-start-5-minutes)
- [Detailed Setup](#detailed-setup)
- [GitHub Secrets Configuration](#github-secrets-configuration)
- [DigitalOcean Droplet Setup](#digitalocean-droplet-setup)
- [Workflow Overview](#workflow-overview)
- [Troubleshooting](#troubleshooting)

## Quick Start (5 minutes)

### Prerequisites

- GitHub repository (with this code)
- DigitalOcean account
- SSH key pair on your local machine

### 1. Create DigitalOcean Droplet

```bash
# Visit: https://cloud.digitalocean.com/droplets/new

# Configuration:
# - Choose: Ubuntu 22.04 LTS
# - Size: $6/month (2GB RAM, 1 vCPU, 50GB SSD)
# - Region: Closest to your location
# - Authentication: Add your SSH key
# - Hostname: crypto-arbitrage-prod

# After creation, note your Droplet IP (e.g., 123.45.67.89)
```

### 2. Initialize Droplet

```bash
# SSH into droplet
ssh root@YOUR_DROPLET_IP

# Run these commands:
apt update && apt upgrade -y
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
apt-get install -y docker-compose-plugin

# Create deploy user
useradd -m -s /bin/bash deploy
usermod -aG docker deploy
usermod -aG sudo deploy
passwd deploy  # Set a password

# Setup SSH for deploy user
sudo -u deploy mkdir -p /home/deploy/.ssh
cat ~/.ssh/id_rsa.pub | sudo tee -a /home/deploy/.ssh/authorized_keys
sudo chmod 700 /home/deploy/.ssh
sudo chmod 600 /home/deploy/.ssh/authorized_keys
sudo chown -R deploy:deploy /home/deploy/.ssh
```

### 3. Configure GitHub Secrets

Go to: **Repository Settings → Secrets and variables → Actions → New repository secret**

Add these secrets:

```
DEPLOY_HOST = YOUR_DROPLET_IP
DEPLOY_USER = deploy
DEPLOY_KEY = <contents of ~/.ssh/id_rsa from your local machine>

PROD_POSTGRES_PASSWORD = <random 32-char password>
PROD_JWT_SECRET = <random 32-char string>
PROD_ENCRYPTION_KEY = <random 32-char string>
PROD_GOOGLE_CLIENT_ID = <your OAuth ID>
PROD_GOOGLE_CLIENT_SECRET = <your OAuth secret>
PROD_DOMAIN = your-domain.com
```

### 4. Push Code

```bash
git add -A
git commit -m "Add CI/CD workflows and production config"
git push origin main
```

**That's it!** Your first deployment will start automatically.

---

## Detailed Setup

### GitHub Secrets Explained

| Secret | Purpose | Example |
|--------|---------|---------|
| `DEPLOY_HOST` | DigitalOcean Droplet IP | `203.0.113.45` |
| `DEPLOY_USER` | SSH user on droplet | `deploy` |
| `DEPLOY_KEY` | SSH private key | `-----BEGIN RSA PRIVATE KEY-----...` |
| `PROD_POSTGRES_PASSWORD` | Database password | `SuperSecure@Pass123!` |
| `PROD_JWT_SECRET` | JWT signing key (32+ chars) | `your_secret_key_minimum_32_characters_long_xyzab` |
| `PROD_ENCRYPTION_KEY` | AES encryption key (32 chars) | `abcdefghijklmnopqrstuvwxyz123456` |
| `PROD_GOOGLE_CLIENT_ID` | OAuth client ID | `123456789.apps.googleusercontent.com` |
| `PROD_GOOGLE_CLIENT_SECRET` | OAuth secret | `GOCSPX-...` |
| `PROD_DOMAIN` | Your domain name | `arbitrage.example.com` |

### Getting SSH Private Key

```bash
# On your local machine:
cat ~/.ssh/id_rsa
```

Copy the entire output including `-----BEGIN RSA PRIVATE KEY-----` and `-----END RSA PRIVATE KEY-----`.

If you don't have an SSH key:

```bash
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""
```

---

## DigitalOcean Droplet Setup

### Complete Setup Script

Run this on your droplet after SSH-ing as root:

```bash
#!/bin/bash

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh

# Install Docker Compose
apt-get install -y docker-compose-plugin

# Add root to docker group (temporary)
usermod -aG docker root

# Create deploy user
useradd -m -s /bin/bash deploy
usermod -aG docker deploy
usermod -aG sudo deploy

# Set deploy user password (run interactively)
passwd deploy

# Setup SSH directory for deploy user
sudo -u deploy mkdir -p /home/deploy/.ssh
chmod 700 /home/deploy/.ssh

# Add your SSH public key
echo "YOUR_SSH_PUBLIC_KEY_HERE" | sudo tee /home/deploy/.ssh/authorized_keys
chmod 600 /home/deploy/.ssh/authorized_keys
chown deploy:deploy /home/deploy/.ssh -R

# Create app directory
mkdir -p /home/deploy/crypto-arbitrage
chown deploy:deploy /home/deploy/crypto-arbitrage

# Enable Docker daemon
systemctl enable docker
systemctl start docker

echo "Setup complete! Droplet is ready for deployment."
```

### Verify Droplet Setup

```bash
# Test Docker
docker --version
docker compose version

# Test deploy user access (from your local machine)
ssh deploy@YOUR_DROPLET_IP "docker ps"
```

---

## Workflow Overview

### CI Workflow (`.github/workflows/ci.yml`)

Runs on every push to `main` or `develop`, and on all pull requests.

**Steps:**
1. Checkout code
2. Setup .NET 8
3. Restore NuGet dependencies
4. Build backend (Release mode)
5. Run backend tests
6. Setup Node.js 18
7. Install npm dependencies
8. Lint frontend code
9. Build frontend

**Failure:** PR cannot be merged if CI fails

### Deploy Workflow (`.github/workflows/deploy.yml`)

Runs automatically after successful push to `main`, or manually via GitHub UI.

**Steps:**
1. Checkout code
2. Build backend Docker image
3. Push to GitHub Container Registry
4. Build frontend Docker image
5. Push to GitHub Container Registry
6. SSH to DigitalOcean droplet
7. Pull latest repository code
8. Update environment variables
9. Pull latest Docker images
10. Stop old containers
11. Start new containers
12. Run database migrations
13. Verify deployment

### Backup Workflow (`.github/workflows/scheduled-backup.yml`)

Runs daily at 2 AM UTC, plus manually via GitHub UI.

**Steps:**
1. SSH to DigitalOcean droplet
2. Create database dump
3. Compress with gzip
4. Save with timestamp
5. Keep only last 7 backups

---

## Manual Deployment

### Trigger via GitHub CLI

```bash
gh workflow run deploy.yml
```

### Trigger via GitHub UI

1. Go to **Actions** tab
2. Click **Build and Deploy**
3. Click **Run workflow** → **Run workflow** again

### Monitor Deployment

```bash
# Watch GitHub Actions logs
# GitHub.com → Repository → Actions → Build and Deploy

# SSH into droplet and check logs
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage
docker compose -f docker-compose.production.yml logs -f
```

---

## Accessing Your Application

After first successful deployment:

```bash
# Frontend
https://your-domain.com

# Backend API
https://your-domain.com/api

# Swagger UI
https://your-domain.com/api/swagger

# Health check
curl https://your-domain.com/api/health
```

---

## Environment Variables

The `.env.production` file is auto-generated from GitHub secrets during deployment. It includes:

```env
# Database
POSTGRES_DB=crypto_arbitrage
POSTGRES_USER=postgres
POSTGRES_PASSWORD=<from PROD_POSTGRES_PASSWORD>

# Ports
BACKEND_PORT=8080
FRONTEND_PORT=3000

# JWT
JWT_SECRET_KEY=<from PROD_JWT_SECRET>
JWT_ISSUER=CryptoArbitrage
JWT_AUDIENCE=CryptoArbitrageClient

# Encryption
ENCRYPTION_KEY=<from PROD_ENCRYPTION_KEY>

# Google OAuth
GOOGLE_CLIENT_ID=<from PROD_GOOGLE_CLIENT_ID>
GOOGLE_CLIENT_SECRET=<from PROD_GOOGLE_CLIENT_SECRET>

# Application
ASPNETCORE_ENVIRONMENT=Production
IMAGE_TAG=<git commit SHA>
```

---

## Database Backups

### Manual Backup

```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage
docker compose -f docker-compose.production.yml exec -T postgres pg_dump -U postgres crypto_arbitrage | gzip > backup-$(date +%Y%m%d_%H%M%S).sql.gz
```

### Automatic Daily Backups

The `scheduled-backup.yml` workflow runs daily and keeps the last 7 backups in `/home/deploy/crypto-arbitrage/backups/`.

### Restore from Backup

```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage

# Stop application
docker compose -f docker-compose.production.yml stop api

# Restore database
docker compose -f docker-compose.production.yml exec -T postgres psql -U postgres crypto_arbitrage < backups/backup-YYYYMMDD_HHMMSS.sql

# Start application
docker compose -f docker-compose.production.yml start api
```

---

## Troubleshooting

### Deployment Fails in GitHub Actions

1. Check **Actions** tab for error logs
2. Most common issues:
   - Missing or incorrect GitHub secrets
   - Droplet IP changed
   - SSH key not added to authorized_keys

### Can't SSH to Droplet

```bash
# Verify SSH key
ls -la ~/.ssh/id_rsa

# Test SSH with verbose output
ssh -vvv deploy@YOUR_DROPLET_IP

# Check if deploy user exists
ssh root@YOUR_DROPLET_IP "id deploy"

# Verify authorized_keys
ssh root@YOUR_DROPLET_IP "cat /home/deploy/.ssh/authorized_keys"
```

### Containers Won't Start

```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage

# Check logs
docker compose -f docker-compose.production.yml logs -f

# Check if ports are available
sudo netstat -tulpn | grep 8080

# Restart Docker
sudo systemctl restart docker
```

### Database Connection Errors

```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage

# Check database container
docker compose -f docker-compose.production.yml ps postgres

# Test database connection
docker compose -f docker-compose.production.yml exec postgres psql -U postgres -d crypto_arbitrage -c "SELECT 1"

# View database logs
docker compose -f docker-compose.production.yml logs postgres
```

### Images Not Found

```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage

# Login to GitHub Container Registry
docker login ghcr.io -u YOUR_GITHUB_USERNAME -p YOUR_GITHUB_TOKEN

# Pull images manually
docker pull ghcr.io/YOUR_USERNAME/crypto-arbitrage/api:latest
docker pull ghcr.io/YOUR_USERNAME/crypto-arbitrage/web:latest
```

### Health Check Fails

```bash
# From your local machine
curl -v https://your-domain.com/api/health

# From droplet
curl http://localhost:8080/api/health

# Check frontend
curl http://localhost:3000
```

---

## Performance Optimization

### Scale Resources

Edit `.github/workflows/deploy.yml` to add resource limits:

```yaml
# Add to docker-compose.production.yml:
services:
  api:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

### Larger Droplet

If experiencing performance issues, upgrade your DigitalOcean Droplet:

```bash
# From DigitalOcean Console:
# Droplets → Select your droplet → Resize → Choose larger plan
# This may require a restart
```

---

## Security Best Practices

1. **Rotate Secrets Regularly**
   - Change JWT_SECRET_KEY, ENCRYPTION_KEY quarterly
   - Update in GitHub secrets
   - Restart deployment to apply

2. **Monitor SSH Access**
   ```bash
   ssh deploy@YOUR_DROPLET_IP
   tail -f /var/log/auth.log
   ```

3. **Use Strong Passwords**
   - PostgreSQL: 32+ random characters
   - SSH: Add passphrase to key for local storage

4. **Enable Firewall**
   ```bash
   ssh root@YOUR_DROPLET_IP
   ufw enable
   ufw allow 22/tcp
   ufw allow 80/tcp
   ufw allow 443/tcp
   ```

5. **SSL/TLS Certificate**
   - Use Let's Encrypt with Certbot (recommended)
   - Add Nginx reverse proxy for HTTPS

---

## Support & Debugging

### Enable Debug Logging

Add to `.env.production`:
```
ASPNETCORE_ENVIRONMENT=Development
```

Then redeploy.

### Contact Support

- GitHub Issues: Report bugs and feature requests
- DigitalOcean Support: Account and infrastructure issues
- Docker Support: Container and image issues

---

## Next Steps

1. ✅ Create DigitalOcean account and droplet
2. ✅ Configure GitHub secrets
3. ✅ Push code to `main` branch
4. ✅ Monitor first deployment via Actions tab
5. ✅ Setup custom domain with DNS
6. ✅ Configure SSL/TLS certificate
7. ✅ Monitor application health
8. ✅ Setup alerts and monitoring

Good luck with your deployment!
