#!/bin/bash
# Debugging script for ML Agent production issues
# Run this on your production server via SSH

echo "=========================================="
echo "ML AGENT PRODUCTION DIAGNOSTICS"
echo "=========================================="
echo ""

echo "1. CHECKING DOCKER CONTAINER STATUS"
echo "-----------------------------------"
docker ps -a | grep -E "crypto-arbitrage-(ml-api|api-real)"
echo ""

echo "2. ML API HEALTH CHECK"
echo "----------------------"
curl -s http://localhost:8080/health 2>/dev/null || echo "Backend health check failed"
docker exec crypto-arbitrage-ml-api-prod curl -s http://localhost:5250/health 2>/dev/null || echo "ML API health check failed"
echo ""

echo "3. CHECKING ML MODEL FILES IN CONTAINER"
echo "---------------------------------------"
docker exec crypto-arbitrage-ml-api-prod ls -lh /app/checkpoints/v3-fixed-entropy/checkpoint_ep350.pt 2>/dev/null
docker exec crypto-arbitrage-ml-api-prod ls -lh /app/trained_models/rl/feature_scaler_v2.pkl 2>/dev/null
echo ""
echo "NOTE: checkpoint_ep350.pt should be ~9.1M, not 133 bytes (LFS pointer)"
echo ""

echo "4. ML API CONTAINER LOGS (Last 50 lines)"
echo "----------------------------------------"
docker logs crypto-arbitrage-ml-api-prod --tail 50 2>&1
echo ""

echo "5. BACKEND AGENT SERVICE LOGS (Last 50 lines)"
echo "---------------------------------------------"
docker logs crypto-arbitrage-api-real --tail 50 2>&1 | grep -i "agent\|ml\|prediction\|action"
echo ""

echo "6. CHECKING ML API PREDICTION REQUESTS"
echo "--------------------------------------"
docker logs crypto-arbitrage-ml-api-prod 2>&1 | grep -E "POST /predict|prediction|action" | tail -20
echo ""

echo "7. CHECKING FOR ML API ERRORS"
echo "-----------------------------"
docker logs crypto-arbitrage-ml-api-prod 2>&1 | grep -i "error\|exception\|traceback" | tail -20
echo ""

echo "8. TESTING ML API MANUALLY"
echo "-------------------------"
echo "Sending test prediction request..."
curl -X POST http://localhost:8080/api/ml/test-prediction 2>&1
echo ""

echo "9. ENVIRONMENT VARIABLES CHECK"
echo "------------------------------"
docker exec crypto-arbitrage-api-real printenv | grep -E "ML_API_URL|AGENT"
echo ""

echo "10. CHECKING DATABASE FOR OPPORTUNITIES"
echo "---------------------------------------"
docker exec crypto-arbitrage-db-prod psql -U ${POSTGRES_USER:-postgres} -d crypto_arbitrage_real -c "SELECT COUNT(*) as detected_opportunities FROM \"ArbitrageOpportunities\" WHERE \"Status\" = 0;" 2>/dev/null
docker exec crypto-arbitrage-db-prod psql -U ${POSTGRES_USER:-postgres} -d crypto_arbitrage_real -c "SELECT COUNT(*) as open_positions FROM \"Positions\" WHERE \"Status\" = 0;" 2>/dev/null
echo ""

echo "=========================================="
echo "DIAGNOSTICS COMPLETE"
echo "=========================================="
