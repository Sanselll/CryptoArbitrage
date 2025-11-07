"""
Test Backend Prediction API - Verify backend sends correct position features to ML server
"""

import requests
import json
from typing import Dict, List

# Test data mimicking what backend would send
test_request = {
    "opportunities": [
        {
            "symbol": "BTCUSDT",
            "long_exchange": "binance",
            "short_exchange": "bybit",
            "long_funding_rate": 0.0001,
            "short_funding_rate": -0.0002,
            "fund_profit_8h": 0.03,
            "fundProfit8h24hProj": 0.09,
            "fundProfit8h3dProj": 0.27,
            "fund_apr": 10.95,
            "fundApr24hProj": 10.95,
            "fundApr3dProj": 10.95,
            "spread30SampleAvg": 0.02,
            "priceSpread24hAvg": 0.02,
            "priceSpread3dAvg": 0.02,
            "spread_volatility_stddev": 0.005,
            "volume_24h": 50000000000,
            "bidAskSpreadPercent": 0.01,
            "orderbookDepthUsd": 10000000,
            "estimatedProfitPercentage": 0.15,
            "positionCostPercent": 0.02,
        }
    ],
    "portfolio": {
        "total_capital": 10000.0,
        "capital": 10000.0,
        "initial_capital": 10000.0,
        "available_margin": 8000.0,
        "margin_utilization": 20.0,
        "utilization": 20.0,
        "num_positions": 1,
        "total_pnl_pct": 2.5,
        "max_drawdown": 0.5,
        "positions": [
            {
                # Direct ML predictor fields
                "unrealized_pnl_pct": 2.5,
                "long_pnl_pct": 1.2,
                "short_pnl_pct": 1.3,
                "liquidation_distance": 0.85,

                # Raw data fields for ML to calculate features
                "position_age_hours": 24.0,
                "long_net_funding_usd": 5.0,
                "short_net_funding_usd": 3.0,
                "short_funding_rate": -0.0002,
                "long_funding_rate": 0.0001,
                "current_long_price": 45000.0,
                "current_short_price": 45010.0,
                "entry_long_price": 44000.0,
                "entry_short_price": 44015.0,
                "position_size_usd": 1000.0,
                "entry_fees_paid_usd": 2.0,
            }
        ]
    },
    "trading_config": {
        "max_leverage": 1.0,
        "target_utilization": 0.5,
        "max_positions": 3,
        "stop_loss_threshold": -0.02,
        "liquidation_buffer": 0.15,
    }
}


def test_ml_server_prediction():
    """Test ML server directly"""
    print("=" * 80)
    print("TESTING ML SERVER PREDICTION ENDPOINT")
    print("=" * 80)

    ml_server_url = "http://localhost:5250/rl/predict/opportunities"

    print(f"\n📡 Sending request to: {ml_server_url}")
    print(f"\n📊 Request payload:")
    print(f"  Opportunities: {len(test_request['opportunities'])}")
    print(f"  Portfolio positions: {len(test_request['portfolio']['positions'])}")
    print(f"  Total capital: ${test_request['portfolio']['total_capital']}")

    # Show position details
    pos = test_request['portfolio']['positions'][0]
    print(f"\n  Position 1 details:")
    print(f"    Unrealized P&L: {pos['unrealized_pnl_pct']:.2f}%")
    print(f"    Position age: {pos['position_age_hours']:.1f}h")
    print(f"    Position size: ${pos['position_size_usd']:.2f}")
    print(f"    Long funding: ${pos['long_net_funding_usd']:.2f}")
    print(f"    Short funding: ${pos['short_net_funding_usd']:.2f}")
    print(f"    Current long price: ${pos['current_long_price']:.2f}")
    print(f"    Current short price: ${pos['current_short_price']:.2f}")

    try:
        response = requests.post(ml_server_url, json=test_request, timeout=10)

        if response.status_code == 200:
            result = response.json()
            print(f"\n✅ Prediction successful!")
            print(f"\n📈 Prediction result:")
            print(f"  Action: {result.get('action', 'N/A')}")
            print(f"  Confidence: {result.get('confidence', 0):.1%}")
            print(f"  State value: {result.get('state_value', 0):.4f}")

            if result.get('action') == 'ENTER':
                print(f"  Symbol: {result.get('opportunity_symbol', 'N/A')}")
                print(f"  Size: {result.get('position_size', 'N/A')}")
            elif result.get('action') == 'EXIT':
                print(f"  Position index: {result.get('position_index', 'N/A')}")

            return True
        else:
            print(f"\n❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False

    except requests.exceptions.ConnectionError:
        print(f"\n❌ Could not connect to ML server at {ml_server_url}")
        print("   Make sure the ML server is running: cd ml_pipeline && python server/app.py")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = test_ml_server_prediction()

    if success:
        print("\n" + "=" * 80)
        print("✅ ML SERVER TEST PASSED")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("❌ ML SERVER TEST FAILED")
        print("=" * 80)
        exit(1)
