"""
ML API Server

Simple Flask API server for ML predictions.
Runs on port 5053 and provides prediction endpoints.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.inference.rl_predictor import RLPredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for C# API calls

# Global predictor instance
rl_predictor = None


def initialize_predictor():
    """Initialize the RL predictor on startup."""
    global rl_predictor
    if rl_predictor is None:
        print("Initializing RL predictor (Simple Mode)...")
        try:
            rl_predictor = RLPredictor(
                model_path='trained_models/rl/deployed/best_model.zip',
                feature_scaler_path='trained_models/rl/feature_scaler.pkl'
            )
            print("✅ RL predictor initialized successfully")
            print("   Model: Latest PBT Agent #1 (Simple Mode: 36 dims = 14 portfolio+execution + 22 opportunity)")
            print("   Path: trained_models/rl/deployed/best_model.zip")
        except Exception as e:
            print(f"⚠️  Warning: Could not initialize RL predictor: {e}")
            print("   RL endpoints will be unavailable")
            rl_predictor = None


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'service': 'ml-api',
        'version': '1.0.0'
    })


@app.route('/rl/predict/opportunities', methods=['POST'])
def predict_opportunities():
    """
    Evaluate ENTER probabilities for opportunities using RL model (Simple Mode).

    Simple Mode: Each opportunity is evaluated independently (1 at a time).

    Expects JSON body:
    {
        "opportunities": [...],  # List of opportunity dicts (any number)
        "portfolio": {...}       # Current portfolio state
    }

    Returns:
    {
        "predictions": [
            {
                "opportunity_index": 0,
                "symbol": "BTCUSDT",
                "enter_probability": 0.75,
                "confidence": "HIGH",
                "hold_probability": 0.10,
                "state_value": 150.5
            },
            ...
        ],
        "model_version": "pbt_20251103_092148"
    }
    """
    try:
        if rl_predictor is None:
            return jsonify({'error': 'RL predictor not initialized'}), 503

        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        opportunities = data.get('opportunities', [])
        portfolio = data.get('portfolio', {})

        if not opportunities:
            return jsonify({'error': 'No opportunities provided'}), 400

        if not portfolio:
            return jsonify({'error': 'No portfolio state provided'}), 400

        # Evaluate opportunities
        predictions = rl_predictor.evaluate_opportunities(opportunities, portfolio)

        # Get model info
        model_info = rl_predictor.get_model_info()

        return jsonify({
            'predictions': predictions,
            'model_version': model_info['model_version']
        })

    except Exception as e:
        print(f"Error in /rl/predict/opportunities: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/rl/predict/positions', methods=['POST'])
def predict_positions():
    """
    Evaluate EXIT probability for open position using RL model (Simple Mode).

    Simple Mode: Only evaluates first position (1 position only).

    Expects JSON body:
    {
        "positions": [...],       # List of position dicts (only first is evaluated)
        "portfolio": {...},       # Current portfolio state
        "opportunity": {...}      # Optional: current opportunity for full observation
    }

    Returns:
    {
        "predictions": [
            {
                "position_index": 0,
                "symbol": "BTCUSDT",
                "exit_probability": 0.65,
                "confidence": "MEDIUM",
                "hold_probability": 0.20,
                "state_value": 150.5
            }
        ],
        "model_version": "pbt_20251103_092148"
    }
    """
    try:
        if rl_predictor is None:
            return jsonify({'error': 'RL predictor not initialized'}), 503

        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        positions = data.get('positions', [])
        portfolio = data.get('portfolio', {})
        opportunity = data.get('opportunity', None)  # Single opportunity (simple mode)

        if not positions:
            return jsonify({'error': 'No positions provided'}), 400

        if not portfolio:
            return jsonify({'error': 'No portfolio state provided'}), 400

        # Evaluate positions (simple mode: only first position)
        predictions = rl_predictor.evaluate_positions(positions, portfolio, opportunity)

        # Get model info
        model_info = rl_predictor.get_model_info()

        return jsonify({
            'predictions': predictions,
            'model_version': model_info['model_version']
        })

    except Exception as e:
        print(f"Error in /rl/predict/positions: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize predictor before starting server
    initialize_predictor()

    # Run Flask server
    print("\n" + "="*80)
    print("ML API Server - RL Only")
    print("="*80)
    print(f"Starting server on http://localhost:5250")
    print(f"Endpoints:")
    print(f"  GET  /health                       - Health check")
    print(f"  POST /rl/predict/opportunities     - RL opportunity evaluation (Simple Mode)")
    print(f"  POST /rl/predict/positions         - RL position evaluation (Simple Mode)")
    print("="*80 + "\n")

    app.run(host='0.0.0.0', port=5250, debug=False)
