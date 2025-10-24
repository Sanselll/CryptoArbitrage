# Commands Reference Guide

Quick reference for all commands needed to set up and manage the deployment.

## DigitalOcean Droplet Setup

### Create Droplet
```bash
# Via DigitalOcean Console (UI)
# 1. Visit https://cloud.digitalocean.com/droplets/new
# 2. Choose: Ubuntu 22.04 LTS, 2GB RAM ($6/month)
# 3. Add SSH key from your machine
# 4. Name: crypto-arbitrage-prod
# 5. Create
```

### Initialize Droplet (run as root)
```bash
ssh root@YOUR_DROPLET_IP

# Update system
apt update && apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
rm get-docker.sh

# Install Docker Compose
apt-get install -y docker-compose-plugin

# Create deploy user
useradd -m -s /bin/bash deploy
usermod -aG docker deploy
usermod -aG sudo deploy

# Set deploy user password
passwd deploy

# Setup SSH for deploy user
sudo -u deploy mkdir -p /home/deploy/.ssh
chmod 700 /home/deploy/.ssh

# Add your SSH public key
cat ~/.ssh/id_rsa.pub | sudo tee -a /home/deploy/.ssh/authorized_keys

# Fix permissions
chmod 600 /home/deploy/.ssh/authorized_keys
chown deploy:deploy /home/deploy/.ssh -R

# Create app directory
mkdir -p /home/deploy/crypto-arbitrage
chown deploy:deploy /home/deploy/crypto-arbitrage

# Verify setup
docker --version
docker compose version
exit
```

### Verify Connectivity
```bash
# From your local machine
ssh deploy@YOUR_DROPLET_IP "docker ps"
# Should show output (even if no containers yet)
```

## Generate GitHub Secrets

### SSH Private Key
```bash
# Display your private key (copy entire output)
cat ~/.ssh/id_rsa
# Includes: -----BEGIN RSA PRIVATE KEY-----
#           (key content)
#           -----END RSA PRIVATE KEY-----
```

### Generate Passwords & Keys
```bash
# PostgreSQL password (copy one)
openssl rand -base64 32

# JWT secret (copy one)
openssl rand -base64 32

# Encryption key (copy one)
head -c 32 < /dev/urandom | base64

# Or use this Python one-liner for all:
python3 -c "import secrets; print('PROD_POSTGRES_PASSWORD:', secrets.token_urlsafe(32)); print('PROD_JWT_SECRET:', secrets.token_urlsafe(32)); print('PROD_ENCRYPTION_KEY:', secrets.token_urlsafe(24))"
```

## Add GitHub Secrets

```bash
# Via GitHub CLI (if installed)
gh secret set DEPLOY_HOST --body "YOUR_DROPLET_IP"
gh secret set DEPLOY_USER --body "deploy"
gh secret set DEPLOY_KEY --body "$(cat ~/.ssh/id_rsa)"
gh secret set PROD_POSTGRES_PASSWORD --body "YOUR_PASSWORD"
gh secret set PROD_JWT_SECRET --body "YOUR_JWT_SECRET"
gh secret set PROD_ENCRYPTION_KEY --body "YOUR_ENCRYPTION_KEY"
gh secret set PROD_GOOGLE_CLIENT_ID --body "YOUR_GOOGLE_ID"
gh secret set PROD_GOOGLE_CLIENT_SECRET --body "YOUR_GOOGLE_SECRET"
gh secret set PROD_DOMAIN --body "your-domain.com"

# Via GitHub UI
# Settings → Secrets and variables → Actions → New repository secret
# (Add each secret one by one)
```

## Deploy

### Via Git Push (Recommended)
```bash
# Commit code
git add -A
git commit -m "Add CI/CD and production deployment"

# Push to main (triggers deployment automatically)
git push origin main

# Watch deployment in GitHub Actions tab
# Should complete in 10-15 minutes
```

### Via GitHub CLI
```bash
# List workflows
gh workflow list

# Run deploy workflow manually
gh workflow run deploy.yml

# Watch workflow progress
gh run watch
```

### Via GitHub UI
```
1. Go to repository → Actions
2. Click "Build and Deploy"
3. Click "Run workflow" → "Run workflow" button
4. Watch logs as deployment progresses
```

## Monitoring & Maintenance

### Check Deployment Status
```bash
# Via GitHub Actions
gh run list --workflow=deploy.yml
gh run view <run_id> --log

# Via GitHub UI
# Repository → Actions → Build and Deploy → Latest run
```

### SSH to Droplet & View Logs
```bash
# SSH to droplet
ssh deploy@YOUR_DROPLET_IP

# Go to app directory
cd /home/deploy/crypto-arbitrage

# View container status
docker compose -f docker-compose.production.yml ps

# View application logs
docker compose -f docker-compose.production.yml logs -f api

# View database logs
docker compose -f docker-compose.production.yml logs -f postgres

# View frontend logs
docker compose -f docker-compose.production.yml logs -f web

# Exit logs (Ctrl+C)
```

### Health Checks
```bash
# Test API health
curl http://YOUR_DROPLET_IP:8080/api/health

# Test frontend
curl http://YOUR_DROPLET_IP:3000

# With domain (after setup)
curl https://your-domain.com/api/health
```

### Check System Resources
```bash
ssh deploy@YOUR_DROPLET_IP

# CPU and memory usage
free -h
top -b -n 1 | head -20

# Disk usage
df -h

# Container resource usage
docker stats

# Exit
exit
```

## Database Backup & Restore

### Create Manual Backup
```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage

# Create backup
docker compose -f docker-compose.production.yml exec -T postgres \
  pg_dump -U postgres crypto_arbitrage | gzip > backup-$(date +%Y%m%d_%H%M%S).sql.gz

# List backups
ls -lah backup-*.sql.gz
```

### Trigger Backup via GitHub Actions
```bash
gh workflow run scheduled-backup.yml
```

### Restore from Backup
```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage

# List available backups
ls -la backups/

# Stop application
docker compose -f docker-compose.production.yml stop api

# Restore backup
docker compose -f docker-compose.production.yml exec -T postgres \
  psql -U postgres crypto_arbitrage < backups/backup-YYYYMMDD_HHMMSS.sql

# Start application
docker compose -f docker-compose.production.yml start api

# Verify
docker compose -f docker-compose.production.yml ps
```

## Restart Services

### Restart Everything
```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage

# Graceful restart (preserves data)
docker compose -f docker-compose.production.yml restart

# View status
docker compose -f docker-compose.production.yml ps
```

### Restart Specific Service
```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage

# Restart backend
docker compose -f docker-compose.production.yml restart api

# Restart frontend
docker compose -f docker-compose.production.yml restart web

# Restart database
docker compose -f docker-compose.production.yml restart postgres
```

### Full Redeploy
```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage

# Stop all services
docker compose -f docker-compose.production.yml down

# Pull latest images
docker pull ghcr.io/YOUR_USERNAME/crypto-arbitrage/api:latest
docker pull ghcr.io/YOUR_USERNAME/crypto-arbitrage/web:latest

# Start services
docker compose -f docker-compose.production.yml --env-file .env.production up -d

# Run migrations
docker compose -f docker-compose.production.yml exec -T api dotnet ef database update

# Verify
docker compose -f docker-compose.production.yml ps
```

## Update Secrets

### Rotate JWT Secret
```bash
# 1. Generate new secret
openssl rand -base64 32

# 2. Update GitHub secret
gh secret set PROD_JWT_SECRET --body "NEW_SECRET_HERE"

# 3. Redeploy (automatic via git push or manual)
git push origin main
# or
gh workflow run deploy.yml

# 4. Check logs to verify new secret is applied
```

### Rotate Encryption Key
```bash
# 1. Generate new key
head -c 32 < /dev/urandom | base64

# 2. Update GitHub secret
gh secret set PROD_ENCRYPTION_KEY --body "NEW_KEY_HERE"

# 3. Redeploy
git push origin main
```

## Troubleshooting Commands

### Cannot SSH to Droplet
```bash
# Test SSH with verbose output
ssh -vvv deploy@YOUR_DROPLET_IP

# Check if host is reachable
ping YOUR_DROPLET_IP

# Verify SSH key permissions
ls -la ~/.ssh/id_rsa
# Should be: -rw------- (600)
```

### Docker Issues on Droplet
```bash
ssh deploy@YOUR_DROPLET_IP

# Check Docker daemon status
sudo systemctl status docker

# Restart Docker
sudo systemctl restart docker

# View Docker logs
sudo journalctl -u docker -n 50

# Check disk space (containers may fail if full)
df -h
```

### Container Won't Start
```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage

# Check why container failed
docker compose -f docker-compose.production.yml logs api

# Check if ports are available
sudo netstat -tulpn | grep 8080

# Try pulling images again
docker pull ghcr.io/YOUR_USERNAME/crypto-arbitrage/api:latest
```

### Database Connection Error
```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage

# Check database container
docker compose -f docker-compose.production.yml ps postgres

# Test database connection
docker compose -f docker-compose.production.yml exec postgres \
  psql -U postgres -d crypto_arbitrage -c "SELECT 1"

# View database logs
docker compose -f docker-compose.production.yml logs postgres
```

### Check Environment Variables
```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage

# View .env.production file
cat .env.production

# Check backend environment
docker compose -f docker-compose.production.yml exec api env | grep ASPNETCORE
```

## Cleanup Commands

### Remove Old Images (to free space)
```bash
ssh deploy@YOUR_DROPLET_IP

# Remove unused images
docker image prune -a -f

# Remove unused volumes
docker volume prune -f

# Show disk space saved
df -h
```

### Remove Old Backups (keep last 7)
```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage

# List backups
ls -la backups/

# Remove old ones (keep last 7)
ls -t backups/backup-*.sql.gz | tail -n +8 | xargs -r rm
```

## GitHub Actions Commands

### View All Workflows
```bash
gh workflow list
gh workflow list --json name,state,path
```

### View Recent Runs
```bash
gh run list
gh run list --workflow=deploy.yml
gh run list --limit 20
```

### View Specific Run
```bash
gh run view <run_id>
gh run view <run_id> --log
gh run view <run_id> --log --jq '.jobs[] | select(.name=="deploy")'
```

### Cancel Running Workflow
```bash
gh run cancel <run_id>
```

### Trigger Workflow
```bash
# CI workflow
gh workflow run ci.yml

# Deploy workflow
gh workflow run deploy.yml

# Backup workflow
gh workflow run scheduled-backup.yml
```

## Quick Copy-Paste Setup

```bash
#!/bin/bash
# Run this on droplet as root

apt update && apt upgrade -y
curl -fsSL https://get.docker.com | sh
apt-get install -y docker-compose-plugin
useradd -m -s /bin/bash deploy
usermod -aG docker deploy
usermod -aG sudo deploy
mkdir -p /home/deploy/.ssh /home/deploy/crypto-arbitrage
chown deploy:deploy /home/deploy -R

echo "Setup complete! Add your SSH key to /home/deploy/.ssh/authorized_keys"
```

