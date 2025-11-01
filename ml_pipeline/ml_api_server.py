"""
ML API Server

Simple Flask API server for ML predictions.
Runs on port 5053 and provides prediction endpoints.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from src.csharp_bridge import MLPredictor
from src.rl_predictor import RLPredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for C# API calls

# Global predictor instances
predictor = None
rl_predictor = None


def initialize_predictor():
    """Initialize the ML predictor on startup."""
    global predictor, rl_predictor
    if predictor is None:
        print("Initializing XGBoost predictor...")
        predictor = MLPredictor(model_dir='models/xgboost')
        print("✅ XGBoost predictor initialized successfully")

    if rl_predictor is None:
        print("Initializing RL predictor...")
        try:
            rl_predictor = RLPredictor(
                model_path='models/pbt_20251101_091256/agent_7_model.zip',
                feature_scaler_path='models/rl/feature_scaler.pkl'
            )
            print("✅ RL predictor initialized successfully")
            print("   Model: agent_7 (P&L: +4.49%, Win Rate: 45.8%)")
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


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict for a single opportunity.

    Expects JSON body with opportunity data.
    Returns prediction with profit, success probability, duration, and composite score.
    """
    try:
        opportunity = request.get_json()

        if not opportunity:
            return jsonify({'error': 'No opportunity data provided'}), 400

        # Make prediction
        result = predictor.predict_single(opportunity)

        return jsonify(result)

    except Exception as e:
        print(f"Error in /predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict for multiple opportunities.

    Expects JSON body with array of opportunity data.
    Returns array of predictions.
    """
    try:
        opportunities = request.get_json()

        if not opportunities:
            return jsonify({'error': 'No opportunities data provided'}), 400

        if not isinstance(opportunities, list):
            return jsonify({'error': 'Expected array of opportunities'}), 400

        # Make batch prediction
        results = predictor.predict_batch(opportunities)

        return jsonify(results)

    except Exception as e:
        print(f"Error in /predict/batch: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/rl/predict/opportunities', methods=['POST'])
def predict_opportunities():
    """
    Evaluate ENTER probabilities for opportunities using RL model.

    Expects JSON body:
    {
        "opportunities": [...],  # List of opportunity dicts (max 5)
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
        "model_version": "pbt_20251101_083701"
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
    Evaluate EXIT probabilities for open positions using RL model.

    Expects JSON body:
    {
        "positions": [...],       # List of position dicts (max 3)
        "portfolio": {...},       # Current portfolio state
        "opportunities": [...]    # Optional: current opportunities for full observation
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
            },
            ...
        ],
        "model_version": "pbt_20251101_083701"
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
        opportunities = data.get('opportunities', [])

        if not positions:
            return jsonify({'error': 'No positions provided'}), 400

        if not portfolio:
            return jsonify({'error': 'No portfolio state provided'}), 400

        # Evaluate positions
        predictions = rl_predictor.evaluate_positions(positions, portfolio, opportunities)

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
    print("ML API Server")
    print("="*80)
    print(f"Starting server on http://localhost:5250")
    print(f"Endpoints:")
    print(f"  GET  /health                       - Health check")
    print(f"  POST /predict                      - Single prediction (XGBoost)")
    print(f"  POST /predict/batch                - Batch prediction (XGBoost)")
    print(f"  POST /rl/predict/opportunities     - RL opportunity evaluation")
    print(f"  POST /rl/predict/positions         - RL position evaluation")
    print("="*80 + "\n")

    app.run(host='0.0.0.0', port=5250, debug=False)
