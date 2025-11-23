# Agent Fix Deployment Guide

## Problem Identified

The agent was returning "HOLD" even with 0 open positions because:

**Root Cause:** Backend API couldn't connect to ML API in Docker environment
- Backend was trying to connect to `localhost:5250` instead of `ml-api:5250`
- Configuration mismatch: `docker-compose.yml` sets `ML_API_URL` but code reads `MLApi:Host`
- When ML API connection fails, prediction service returns null → agent defaults to HOLD

**Log Evidence:**
```
fail: CryptoArbitrage.API.Services.ML.RLPredictionService[0]
      Failed to connect to ML API
      System.Net.Http.HttpRequestException: Connection refused (localhost:5250)
```

## Fix Applied

Added proper ASP.NET Core configuration to both docker-compose files:
- `MLApi__Host=ml-api` (Docker service name)
- `MLApi__Port=5250`

**Files Modified:**
1. `docker-compose.yml` (local/development)
2. `docker-compose.production.yml` (production)

## Deployment Steps for Production Server

### Step 1: Connect to Server
```bash
ssh your-server
cd /path/to/CryptoArbitrage
```

### Step 2: Pull Latest Changes
```bash
git pull origin feature/RL-v5
```

### Step 3: Restart Affected Services
```bash
# Option A: Restart only backend services (faster, ~10 seconds downtime)
docker-compose -f docker-compose.production.yml restart backend-real backend-demo

# Option B: Full restart with rebuild (if needed, ~2-3 minutes)
docker-compose -f docker-compose.production.yml down backend-real backend-demo
docker-compose -f docker-compose.production.yml up -d backend-real backend-demo
```

### Step 4: Verify ML API Connection
```bash
# Check backend logs for successful ML API connection
docker logs crypto-arbitrage-api-real --tail 50 | grep -i "ml api"

# You should see:
# "AgentBackgroundService initialized. ML API: http://ml-api:5250"

# Check for successful predictions (no more "Connection refused" errors)
docker logs crypto-arbitrage-api-real --tail 100 | grep -i "failed to connect"
```

### Step 5: Verify Agent is Making Predictions
```bash
# Wait ~30 seconds for next prediction cycle, then check:
docker logs crypto-arbitrage-api-real --tail 20 | grep -i "action:"

# You should see actual ENTER/EXIT actions, not just HOLD
# Example: "Action: ENTER, Symbol: BTCUSDT"
```

### Step 6: Monitor Agent Activity
```bash
# Watch agent decisions in real-time
docker logs -f crypto-arbitrage-api-real | grep -i "agent"
```

## Expected Behavior After Fix

**Before Fix:**
```
info: CryptoArbitrage.API.Services.Agent.AgentBackgroundService[0]
      User: xxx, Executions: 0 (0 individual positions)
fail: CryptoArbitrage.API.Services.ML.RLPredictionService[0]
      Failed to connect to ML API
      System.Net.Http.HttpRequestException: Connection refused (localhost:5250)
info: CryptoArbitrage.API.Services.Agent.AgentBackgroundService[0]
      Action: HOLD, Symbol: (null)  ← Always HOLD because ML API unreachable
```

**After Fix:**
```
info: CryptoArbitrage.API.Services.Agent.AgentBackgroundService[0]
      User: xxx, Executions: 0 (0 individual positions)
info: CryptoArbitrage.API.Services.ML.RLPredictionService[0]
      Sending prediction request (UnifiedFeatureBuilder): 10 opportunities, 0 positions
info: CryptoArbitrage.API.Services.Agent.AgentBackgroundService[0]
      ========== ML API RETURNED ==========
info: CryptoArbitrage.API.Services.Agent.AgentBackgroundService[0]
      Action: ENTER, Symbol: BTCUSDT  ← Actual ML predictions!
```

## Quick Verification Command

Run this single command to verify everything is working:

```bash
echo "=== 1. ML API Container Status ===" && \
docker ps | grep ml-api && \
echo "" && \
echo "=== 2. Backend ML API Configuration ===" && \
docker exec crypto-arbitrage-api-real printenv | grep MLApi && \
echo "" && \
echo "=== 3. Recent Agent Actions ===" && \
docker logs crypto-arbitrage-api-real --tail 30 | grep "Action:" && \
echo "" && \
echo "=== 4. ML API Connection Status ===" && \
docker logs crypto-arbitrage-api-real --tail 100 | grep -i "failed to connect" || echo "✅ No connection errors found!"
```

## Rollback (If Needed)

If something goes wrong:
```bash
# 1. Revert docker-compose changes
git checkout HEAD~1 docker-compose.production.yml

# 2. Restart services
docker-compose -f docker-compose.production.yml restart backend-real backend-demo

# 3. Check logs
docker logs crypto-arbitrage-api-real --tail 50
```

## Additional Notes

- **No database changes required** - this is purely a configuration fix
- **No code recompilation needed** - just restart containers with new env vars
- **Downtime:** ~10 seconds if using `restart`, ~30 seconds if using `down/up`
- **Agent will resume automatically** once backend reconnects to ML API

## Troubleshooting

If agent still returns HOLD after fix:

1. **Verify ML API is running:**
   ```bash
   docker logs crypto-arbitrage-ml-api-prod --tail 50
   ```

2. **Check ML model files are loaded:**
   ```bash
   docker exec crypto-arbitrage-ml-api-prod ls -lh /app/checkpoints/v3-fixed-entropy/checkpoint_ep350.pt
   # Should show ~9.1M, not 133 bytes (LFS pointer)
   ```

3. **Test ML API health endpoint:**
   ```bash
   docker exec crypto-arbitrage-api-real curl -s http://ml-api:5250/health
   ```

4. **Check for opportunities in database:**
   ```bash
   docker exec crypto-arbitrage-db-prod psql -U postgres -d crypto_arbitrage_real \
     -c "SELECT COUNT(*) FROM \"ArbitrageOpportunities\" WHERE \"Status\" = 0;"
   ```

If no opportunities are detected, agent will correctly return HOLD (this is expected behavior).
