# CI/CD Setup - Start Here

Complete automated deployment pipeline configured for GitHub + DigitalOcean.

## 🎯 What You Just Got

A production-ready deployment pipeline that:
- ✅ Tests code automatically on every push
- ✅ Deploys automatically to production
- ✅ Backs up database daily
- ✅ Costs only ~$6/month
- ✅ Requires zero manual steps after initial setup

## 📖 Documentation Files

Read in this order:

### 1. **CI_CD_SETUP.md** ← START HERE
Comprehensive guide with:
- 5-minute quick start
- Step-by-step detailed instructions
- Troubleshooting for 20+ common issues
- Security best practices
- Performance optimization

**Time to read:** 20-30 minutes

### 2. **DEPLOYMENT_CHECKLIST.md**
11-step checklist to follow:
- Exactly what values to collect
- Where to get each secret
- Verification steps
- Success indicators

**Time to complete:** 30-45 minutes

### 3. **COMMANDS_REFERENCE.md**
Copy-paste ready commands for:
- DigitalOcean setup
- GitHub secrets
- Monitoring
- Troubleshooting
- Maintenance

**Time needed:** Reference as needed

### 4. **DEPLOYMENT.md** (Updated)
General deployment guide now includes:
- Updated GitHub Actions + DigitalOcean section
- Docker deployment info
- Database management

## 🚀 Quick Overview

### What Happens When You Push Code

```
git push origin main
    ↓
GitHub CI runs (tests backend + frontend)
    ↓
Tests pass? → Build Docker images
    ↓
Push images to registry
    ↓
SSH to DigitalOcean droplet
    ↓
Pull code, update config, deploy
    ↓
✓ Application live!
```

### GitHub Actions Workflows Created

| File | Purpose | Trigger |
|------|---------|---------|
| `ci.yml` | Test backend & frontend | Every push/PR to main or develop |
| `deploy.yml` | Build & deploy to production | Push to main OR manual trigger |
| `scheduled-backup.yml` | Daily database backup | Daily at 2 AM UTC OR manual |

### Cost Breakdown

| Service | Cost | Notes |
|---------|------|-------|
| DigitalOcean Droplet | $6/month | 2GB RAM, 50GB SSD |
| GitHub Actions | FREE | Included in free tier |
| Domain | $0-12/month | Optional |
| **Total** | **$6-10/month** | Very affordable |

## ⚡ Quick Start (TL;DR)

1. Create DigitalOcean Droplet (Ubuntu 22.04, $6/month)
2. Initialize with Docker & SSH setup (~5 minutes)
3. Generate 9 secrets (~2 minutes)
4. Add secrets to GitHub (~5 minutes)
5. Push code to main branch (automatic deployment)
6. Monitor in GitHub Actions tab (10-15 minutes)
7. ✓ App is live!

**Total time:** ~1 hour for complete setup

## 📋 Files in Repository

```
CryptoArbitrage/
├── .github/workflows/
│   ├── ci.yml                    ← Test workflow
│   ├── deploy.yml                ← Deploy workflow
│   └── scheduled-backup.yml      ← Backup workflow
│
├── .env.production.example       ← Production config template
│
├── CI_CD_SETUP.md                ← Main guide (START HERE)
├── DEPLOYMENT_CHECKLIST.md       ← Step-by-step checklist
├── COMMANDS_REFERENCE.md         ← Copy-paste commands
├── DEPLOYMENT.md                 ← Updated deployment guide
└── CI_CD_README.md               ← This file
```

## 🔑 GitHub Secrets Needed

You need to add 9 secrets to your repository:

```
DEPLOY_HOST                   → Your DigitalOcean IP
DEPLOY_USER                   → deploy
DEPLOY_KEY                    → SSH private key
PROD_POSTGRES_PASSWORD        → Random password
PROD_JWT_SECRET               → Random string
PROD_ENCRYPTION_KEY           → Random string
PROD_GOOGLE_CLIENT_ID         → From Google Console
PROD_GOOGLE_CLIENT_SECRET     → From Google Console
PROD_DOMAIN                   → your-domain.com
```

**Where to add:** GitHub → Repository Settings → Secrets and variables → Actions

## ✨ Features

### Automated
- Builds on every push
- Tests before deployment
- Deploys automatically
- Backups daily
- Restarts on failure

### Secure
- SSH key authentication
- Encrypted secrets
- Non-root user runs apps
- Database encryption
- JWT tokens

### Reliable
- Health checks
- Auto-restart
- Easy rollback
- Database migrations
- Uptime monitoring

### Cost-Effective
- $6/month hosting
- Free CI/CD
- No additional tools
- 10x cheaper than AWS/Azure

## 🎓 Learning Path

**If you're new to this:**
1. Read CI_CD_SETUP.md (comprehensive)
2. Create DigitalOcean account
3. Follow DEPLOYMENT_CHECKLIST.md
4. Deploy and watch GitHub Actions

**If you're familiar with CI/CD:**
1. Skim CI_CD_SETUP.md sections you need
2. Use COMMANDS_REFERENCE.md for commands
3. Follow DEPLOYMENT_CHECKLIST.md for verification

**If you need to troubleshoot:**
1. Check GitHub Actions logs first (most details)
2. SSH to droplet: `docker compose logs -f`
3. Read troubleshooting section in CI_CD_SETUP.md

## 📞 Getting Help

### Common Questions

**Q: Do I need DigitalOcean account?**
A: Yes, but they offer $200 free credit for new accounts

**Q: Can I use a different cloud provider?**
A: Yes! The setup works with AWS, Azure, GCP, or any VPS with Docker

**Q: What if I have an existing server?**
A: You can use it! Just need Docker + Docker Compose installed

**Q: Can I deploy manually instead of automatically?**
A: Yes, click "Run workflow" in GitHub Actions tab anytime

**Q: How do I rollback if deployment fails?**
A: Restore from automated daily backup (see COMMANDS_REFERENCE.md)

### Troubleshooting

1. **GitHub Actions fails** → Check Actions tab logs
2. **Containers won't start** → SSH to droplet, run: `docker compose logs`
3. **Can't SSH** → Check SSH key in authorized_keys
4. **Website down** → Check container status: `docker ps`
5. **Database error** → View logs: `docker compose logs postgres`

All issues documented in **CI_CD_SETUP.md**

## 🔄 Workflow After Setup

Once configured, your workflow becomes:

```
1. Developer pushes to main
2. CI runs tests (automatic)
3. Deploy runs (automatic)
4. App updates (10-15 min)
5. Zero downtime
6. Database migrates automatically
7. Backups created daily
8. Everyone happy 🎉
```

## ✅ Verify Everything Works

After deployment:

```bash
# 1. Check GitHub Actions (watch deployment)
# GitHub → Actions → Build and Deploy

# 2. SSH and view logs
ssh deploy@YOUR_DROPLET_IP
cd /home/deploy/crypto-arbitrage
docker compose ps

# 3. Test API
curl https://your-domain.com/api/health
# Should return: 200 OK

# 4. Test Frontend
curl https://your-domain.com
# Should return: HTML
```

## 📚 Additional Resources

- **DigitalOcean Docs**: https://docs.digitalocean.com
- **GitHub Actions Docs**: https://docs.github.com/actions
- **Docker Compose Docs**: https://docs.docker.com/compose
- **Repository**: Your GitHub repo Actions tab

## 🎯 Next Steps

1. **Right now:** Read CI_CD_SETUP.md (20-30 minutes)
2. **Then:** Follow DEPLOYMENT_CHECKLIST.md (30-45 minutes)
3. **Finally:** Monitor first deployment in GitHub Actions

**Questions?** All answered in CI_CD_SETUP.md

---

**Ready?** Open **CI_CD_SETUP.md** and follow the instructions!

Your production deployment pipeline is waiting. 🚀
