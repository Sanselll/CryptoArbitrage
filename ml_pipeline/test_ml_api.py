"""
Test ML API RL Endpoints

Simple test script to verify the RL prediction endpoints work correctly.
"""

import requests
import json

API_URL = "http://localhost:5250"

def test_health():
    """Test health endpoint."""
    print("\n" + "="*80)
    print("Testing /health endpoint")
    print("="*80)

    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    assert response.status_code == 200
    print("✅ Health check passed")


def test_opportunities():
    """Test RL opportunities endpoint."""
    print("\n" + "="*80)
    print("Testing /rl/predict/opportunities endpoint")
    print("="*80)

    # Sample data
    payload = {
        "opportunities": [
            {
                "symbol": "BTCUSDT",
                "long_funding_rate": 0.01,
                "short_funding_rate": -0.02,
                "long_funding_interval_hours": 8,
                "short_funding_interval_hours": 8,
                "fund_profit_8h": 0.03,
                "fundProfit8h24hProj": 0.09,
                "fundProfit8h3dProj": 0.27,
                "fund_apr": 45.0,
                "fundApr24hProj": 42.0,
                "fundApr3dProj": 40.0,
                "spread30SampleAvg": 0.02,
                "priceSpread24hAvg": 0.015,
                "priceSpread3dAvg": 0.01,
                "spread_volatility_stddev": 0.005,
                "volume_24h": 1000000000,
                "bidAskSpreadPercent": 0.001,
                "orderbookDepthUsd": 5000000,
                "estimatedProfitPercentage": 0.025,
                "positionCostPercent": 0.002
            },
            {
                "symbol": "ETHUSDT",
                "long_funding_rate": 0.005,
                "short_funding_rate": -0.01,
                "long_funding_interval_hours": 8,
                "short_funding_interval_hours": 8,
                "fund_profit_8h": 0.015,
                "fundProfit8h24hProj": 0.045,
                "fundProfit8h3dProj": 0.135,
                "fund_apr": 22.5,
                "fundApr24hProj": 21.0,
                "fundApr3dProj": 20.0,
                "spread30SampleAvg": 0.01,
                "priceSpread24hAvg": 0.008,
                "priceSpread3dAvg": 0.006,
                "spread_volatility_stddev": 0.003,
                "volume_24h": 500000000,
                "bidAskSpreadPercent": 0.0008,
                "orderbookDepthUsd": 3000000,
                "estimatedProfitPercentage": 0.012,
                "positionCostPercent": 0.0015
            }
        ],
        "portfolio": {
            "capital": 10000.0,
            "initial_capital": 10000.0,
            "num_positions": 0,
            "utilization": 0.0,
            "total_pnl_pct": 0.0,
            "drawdown": 0.0,
            "positions": []
        }
    }

    response = requests.post(
        f"{API_URL}/rl/predict/opportunities",
        json=payload
    )

    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    assert response.status_code == 200
    assert 'predictions' in result
    assert 'model_version' in result
    assert len(result['predictions']) == 2

    for i, pred in enumerate(result['predictions']):
        print(f"\n  Opportunity {i}: {pred['symbol']}")
        print(f"    Enter Probability: {pred['enter_probability']:.2%}")
        print(f"    Confidence: {pred['confidence']}")
        print(f"    Hold Probability: {pred['hold_probability']:.2%}")

    print("\n✅ Opportunities prediction passed")


def test_positions():
    """Test RL positions endpoint."""
    print("\n" + "="*80)
    print("Testing /rl/predict/positions endpoint")
    print("="*80)

    # Sample data
    payload = {
        "positions": [
            {
                "symbol": "BTCUSDT",
                "pnl_pct": 1.5,
                "hours_held": 12.0,
                "funding_rate": 0.01
            }
        ],
        "portfolio": {
            "capital": 10150.0,
            "initial_capital": 10000.0,
            "num_positions": 1,
            "utilization": 0.33,
            "total_pnl_pct": 1.5,
            "drawdown": 0.0,
            "positions": [
                {
                    "pnl_pct": 1.5,
                    "hours_held": 12.0,
                    "funding_rate": 0.01
                }
            ]
        },
        "opportunities": []  # Optional
    }

    response = requests.post(
        f"{API_URL}/rl/predict/positions",
        json=payload
    )

    print(f"Status: {response.status_code}")
    result = response.json()
    print(f"Response: {json.dumps(result, indent=2)}")

    assert response.status_code == 200
    assert 'predictions' in result
    assert 'model_version' in result
    assert len(result['predictions']) == 1

    for i, pred in enumerate(result['predictions']):
        print(f"\n  Position {i}: {pred['symbol']}")
        print(f"    Exit Probability: {pred['exit_probability']:.2%}")
        print(f"    Confidence: {pred['confidence']}")
        print(f"    Hold Probability: {pred['hold_probability']:.2%}")

    print("\n✅ Positions prediction passed")


if __name__ == '__main__':
    print("\n" + "="*80)
    print("ML API RL Endpoints Test Suite")
    print("="*80)
    print("\nTesting ML API server on port 5250...")

    try:
        test_health()
        test_opportunities()
        test_positions()

        print("\n" + "="*80)
        print("✅ All tests passed!")
        print("="*80 + "\n")

    except requests.exceptions.ConnectionError:
        print("\n❌ Error: Could not connect to ML API server")
        print("   Make sure the server is running: python ml_api_server.py")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
