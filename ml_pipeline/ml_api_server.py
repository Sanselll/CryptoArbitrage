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

app = Flask(__name__)
CORS(app)  # Enable CORS for C# API calls

# Global predictor instance
predictor = None


def initialize_predictor():
    """Initialize the ML predictor on startup."""
    global predictor
    if predictor is None:
        print("Initializing ML predictor...")
        predictor = MLPredictor(model_dir='models/xgboost')
        print("✅ ML predictor initialized successfully")


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


if __name__ == '__main__':
    # Initialize predictor before starting server
    initialize_predictor()

    # Run Flask server
    print("\n" + "="*80)
    print("ML API Server")
    print("="*80)
    print(f"Starting server on http://localhost:5250")
    print(f"Endpoints:")
    print(f"  GET  /health           - Health check")
    print(f"  POST /predict          - Single prediction")
    print(f"  POST /predict/batch    - Batch prediction")
    print("="*80 + "\n")

    app.run(host='0.0.0.0', port=5250, debug=False)
