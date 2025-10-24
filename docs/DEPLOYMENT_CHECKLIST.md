# Deployment Checklist

Use this checklist to set up automated deployment with GitHub CI/CD and DigitalOcean.

## Prerequisites
- [ ] GitHub repository access
- [ ] DigitalOcean account (free $200 credit available)
- [ ] SSH key pair generated (`ssh-keygen` or existing key)
- [ ] Domain name (optional, for custom domain)

## Step 1: Create DigitalOcean Droplet
- [ ] Go to DigitalOcean Console → Droplets → Create Droplet
- [ ] Select: Ubuntu 22.04 LTS
- [ ] Choose size: $6/month (2GB RAM, 50GB SSD)
- [ ] Select closest region
- [ ] Add your SSH public key
- [ ] Name: `crypto-arbitrage-prod`
- [ ] Click Create
- [ ] **Save the Droplet IP address** (e.g., `123.45.67.89`)

## Step 2: Initialize Droplet
- [ ] SSH into droplet: `ssh root@YOUR_DROPLET_IP`
- [ ] Update system: `apt update && apt upgrade -y`
- [ ] Install Docker: `curl -fsSL https://get.docker.com | sh`
- [ ] Install Docker Compose: `apt-get install -y docker-compose-plugin`
- [ ] Create deploy user: `useradd -m -s /bin/bash deploy`
- [ ] Add deploy to docker: `usermod -aG docker deploy && usermod -aG sudo deploy`
- [ ] Set deploy password: `passwd deploy`
- [ ] Setup SSH for deploy user (copy your public key to `.ssh/authorized_keys`)
- [ ] Create app directory: `mkdir -p /home/deploy/crypto-arbitrage && chown deploy:deploy /home/deploy/crypto-arbitrage`

## Step 3: Prepare GitHub Secrets

Get values for these secrets:

### SSH Connection
- [ ] **DEPLOY_HOST**: Your DigitalOcean Droplet IP
  - Example: `203.0.113.45`
  
- [ ] **DEPLOY_USER**: `deploy`
  
- [ ] **DEPLOY_KEY**: Contents of `~/.ssh/id_rsa`
  - Run: `cat ~/.ssh/id_rsa`
  - Copy entire content including `-----BEGIN` and `-----END`

### Database
- [ ] **PROD_POSTGRES_PASSWORD**: Generate strong 32+ character password
  - Use: `openssl rand -base64 32`

### Security Keys
- [ ] **PROD_JWT_SECRET**: Generate 32+ character random string
  - Use: `openssl rand -base64 32`
  
- [ ] **PROD_ENCRYPTION_KEY**: Generate 32 character random string
  - Use: `head -c 32 < /dev/urandom | base64`

### Google OAuth
- [ ] **PROD_GOOGLE_CLIENT_ID**: From Google Console
  - Go to: https://console.cloud.google.com/apis/credentials
  
- [ ] **PROD_GOOGLE_CLIENT_SECRET**: From Google Console

### Domain
- [ ] **PROD_DOMAIN**: Your domain name
  - Example: `arbitrage.example.com`
  - Can be left as `localhost` initially for testing

## Step 4: Add GitHub Secrets

- [ ] Go to GitHub repository → Settings → Secrets and variables → Actions
- [ ] Click "New repository secret" for each:
  - [ ] DEPLOY_HOST
  - [ ] DEPLOY_USER
  - [ ] DEPLOY_KEY
  - [ ] PROD_POSTGRES_PASSWORD
  - [ ] PROD_JWT_SECRET
  - [ ] PROD_ENCRYPTION_KEY
  - [ ] PROD_GOOGLE_CLIENT_ID
  - [ ] PROD_GOOGLE_CLIENT_SECRET
  - [ ] PROD_DOMAIN

## Step 5: Repository Files

These files should be in the repository:
- [ ] `.github/workflows/ci.yml` - Continuous Integration
- [ ] `.github/workflows/deploy.yml` - Automated Deployment
- [ ] `.github/workflows/scheduled-backup.yml` - Daily Backups
- [ ] `.env.production.example` - Production template
- [ ] `docker-compose.production.yml` - Production containers
- [ ] `CI_CD_SETUP.md` - Detailed setup guide
- [ ] `DEPLOYMENT.md` - General deployment guide

## Step 6: Verify Droplet Connectivity

```bash
# Test SSH connection from your local machine
ssh -i ~/.ssh/id_rsa deploy@YOUR_DROPLET_IP "docker ps"

# If successful, you should see output (likely empty container list)
```

- [ ] Can SSH to droplet as deploy user
- [ ] Docker is running on droplet
- [ ] Deploy user can run docker commands

## Step 7: Trigger First Deployment

Choose one method:

### Option A: Via GitHub CLI
```bash
gh workflow run deploy.yml
```
- [ ] GitHub CLI installed
- [ ] Command executed
- [ ] Workflow started

### Option B: Via GitHub UI
- [ ] Go to GitHub → Actions → Build and Deploy
- [ ] Click "Run workflow" → "Run workflow"

### Option C: Via Git Push
```bash
git add -A
git commit -m "Configure CI/CD and production deployment"
git push origin main
```
- [ ] Pushed to main branch
- [ ] CI workflow runs first (check Actions tab)
- [ ] After CI passes, Deploy workflow starts

## Step 8: Monitor Deployment

- [ ] Go to GitHub Actions tab
- [ ] Watch "Build and Deploy" workflow progress
- [ ] Wait for "deploy-and-push" job to complete (5-10 minutes)
- [ ] Wait for "deploy" job to complete (2-5 minutes)
- [ ] Check for success (green checkmark)

## Step 9: Verify Deployment

### Check Container Status
```bash
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage
docker compose -f docker-compose.production.yml ps
```
- [ ] All containers are running (STATUS: healthy or Up)
- [ ] postgres, api, web containers present

### Check Application Logs
```bash
docker compose -f docker-compose.production.yml logs -f api
```
- [ ] No critical errors
- [ ] Application started successfully
- [ ] Database migrations completed

### Test API Health
```bash
curl http://YOUR_DROPLET_IP:8080/api/health
```
- [ ] Returns 200 OK
- [ ] Shows health status JSON

### Test Frontend
```bash
curl http://YOUR_DROPLET_IP:3000
```
- [ ] Returns HTML (frontend page)
- [ ] No errors

## Step 10: Setup Domain (Optional)

- [ ] Register domain or use existing
- [ ] Point DNS to DigitalOcean Droplet IP:
  - [ ] Add A record: `arbitrage` → `YOUR_DROPLET_IP`
  - [ ] Or CNAME: `www` → main domain

- [ ] Setup SSL/TLS (Let's Encrypt):
  ```bash
  ssh root@YOUR_DROPLET_IP
  apt-get install certbot
  certbot certonly --standalone -d your-domain.com
  ```
  - [ ] Certificate generated
  - [ ] Configure reverse proxy (Nginx/Caddy)

- [ ] Update CORS in backend if using custom domain

## Step 11: Monitor & Maintain

### Daily
- [ ] Check application health: `curl https://your-domain.com/api/health`
- [ ] Review DigitalOcean Droplet monitoring dashboard
- [ ] Check for any alerts

### Weekly
- [ ] Verify database backups exist: `ssh deploy@YOUR_DROPLET_IP ls -la /home/deploy/crypto-arbitrage/backups/`
- [ ] Review application logs for errors
- [ ] Test disaster recovery (restore a backup)

### Monthly
- [ ] Rotate JWT_SECRET_KEY and ENCRYPTION_KEY
- [ ] Update GitHub secrets with new values
- [ ] Redeploy to apply new secrets

## Troubleshooting Reference

If something fails, check:

1. **GitHub Actions logs** - Most detailed error messages
2. **Droplet container logs**:
   ```bash
   ssh deploy@YOUR_DROPLET_IP
   cd /home/deploy/crypto-arbitrage
   docker compose -f docker-compose.production.yml logs
   ```
3. **SSH connectivity** - Can you reach the droplet?
4. **GitHub secrets** - Are they all set correctly?
5. **Disk space** - `df -h` on droplet
6. **Docker images** - `docker images` on droplet

## Success Indicators

Deployment is successful when:
- ✅ GitHub Actions workflow shows green checkmark
- ✅ All containers running without restart loops
- ✅ API health check returns 200 OK
- ✅ Frontend loads in browser
- ✅ No critical errors in logs
- ✅ Database is accessible
- ✅ Daily backups are being created

## Next Steps After Deployment

1. Test user registration and authentication
2. Configure exchange API keys in the application
3. Set up monitoring and alerting
4. Load test the application
5. Create backup restore procedure documentation
6. Setup error tracking (Sentry)
7. Configure rate limiting

## Support Resources

- **GitHub Issues**: Report bugs and feature requests
- **DigitalOcean Docs**: https://docs.digitalocean.com
- **Docker Docs**: https://docs.docker.com
- **Docker Compose Docs**: https://docs.docker.com/compose/
- **GitHub Actions Docs**: https://docs.github.com/actions

---

Once you complete all steps, your application will automatically:
- ✅ Build on every code push
- ✅ Test before deployment
- ✅ Deploy to production automatically
- ✅ Backup database daily
- ✅ Keep running even after restarts
