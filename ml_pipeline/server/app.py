"""
ML API Server

Simple Flask API server for ML predictions.
Runs on port 5053 and provides prediction endpoints.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os
from pathlib import Path
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from server.agent_manager import AgentManager, AgentConfig, AgentStatus
from server.decision_logger import DecisionLogger
from common.features.schemas import RLRawDataRequest, RLPredictionResponse
from server.inference.rl_predictor import ModularRLPredictor

app = Flask(__name__)
CORS(app)  # Enable CORS for C# API calls

# Global instances
rl_predictor = None
agent_manager = AgentManager()
decision_logger = DecisionLogger(max_decisions_per_user=1000)

# Confidence thresholds for action filtering
# Actions below these thresholds are converted to HOLD
ENTER_CONFIDENCE_THRESHOLD = 0.0   # Disabled - allow all ENTER actions
EXIT_CONFIDENCE_THRESHOLD = 0.0    # Disabled - allow all EXIT actions

# Decision log file for raw input/output analysis
# Stored in server directory for easy access
DECISION_LOG_FILE = os.path.join(os.path.dirname(__file__), 'ml_decisions.log')


def log_raw_decision(request_data: dict, response_data: dict, user_id: str = None):
    """
    Log raw ML prediction input and output to separate file for analysis.

    Args:
        request_data: Complete request data (opportunities, portfolio, trading_config)
        response_data: Complete response data (action, confidence, etc.)
        user_id: Optional user identifier
    """
    try:
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'user_id': user_id,
            'request': request_data,
            'response': response_data
        }

        # Write as JSON lines format (one JSON object per line)
        with open(DECISION_LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    except Exception as e:
        print(f"Warning: Failed to write decision log: {e}")


def initialize_predictor():
    """Initialize the RL predictor on startup."""
    import numpy as np
    import torch

    # Set random seed for reproducible predictions
    np.random.seed(42)
    torch.manual_seed(42)

    # Enable deterministic algorithms for 100% reproducibility
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print("🎲 Random seed: 42 (deterministic mode enabled)")
    print("   torch.use_deterministic_algorithms(True)")
    print("   cudnn.deterministic=True, cudnn.benchmark=False")

    global rl_predictor
    if rl_predictor is None:
        print("Initializing Modular RL predictor...")
        try:
            rl_predictor = ModularRLPredictor(
                model_path='trained_models/rl/v7_5000_ep4400.pt',
                device='cpu'
            )
            print("✅ RL predictor initialized successfully")
            print("   Architecture: Unified Feature Builder (229 dims)")
            print("   Model: trained_models/rl/v7_5000_ep4400.pt")
            print("   Action space: 36 actions (1 HOLD + 30 ENTER + 5 EXIT)")
            print("   Features: 5 config + 3 portfolio + 85 executions + 120 opportunities")
            print(f"   Confidence thresholds: ENTER >= {ENTER_CONFIDENCE_THRESHOLD:.0%}, EXIT >= {EXIT_CONFIDENCE_THRESHOLD:.0%}")
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


# ============================================================================
# AGENT MANAGEMENT ENDPOINTS
# ============================================================================

@app.route('/agent/start', methods=['POST'])
def start_agent():
    """
    Start autonomous trading agent for user.

    Expects JSON body:
    {
        "user_id": "user123",
        "config": {
            "max_leverage": 1.0,
            "target_utilization": 0.9,
            "max_positions": 3,
            "prediction_interval_sec": 60
        }
    }

    Returns:
    {
        "success": true,
        "message": "Agent started successfully",
        "session": {...}
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        user_id = data.get('user_id')
        config_data = data.get('config', {})

        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400

        # Parse config
        config = AgentConfig.from_dict(config_data)

        # Start agent
        success, error_msg = agent_manager.start_agent(user_id, config)

        if not success:
            return jsonify({'error': error_msg}), 400

        # Get session
        session = agent_manager.get_session(user_id)

        return jsonify({
            'success': True,
            'message': 'Agent started successfully',
            'session': session.to_dict() if session else None
        })

    except Exception as e:
        print(f"Error in /agent/start: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/agent/stop', methods=['POST'])
def stop_agent():
    """
    Stop autonomous trading agent for user.

    Expects JSON body:
    {
        "user_id": "user123"
    }

    Returns:
    {
        "success": true,
        "message": "Agent stopped successfully"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        user_id = data.get('user_id')

        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400

        # Stop agent
        success, error_msg = agent_manager.stop_agent(user_id)

        if not success:
            return jsonify({'error': error_msg}), 400

        return jsonify({
            'success': True,
            'message': 'Agent stopped successfully'
        })

    except Exception as e:
        print(f"Error in /agent/stop: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/agent/pause', methods=['POST'])
def pause_agent():
    """
    Pause autonomous trading agent for user.

    Expects JSON body:
    {
        "user_id": "user123"
    }

    Returns:
    {
        "success": true,
        "message": "Agent paused successfully"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        user_id = data.get('user_id')

        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400

        # Pause agent
        success, error_msg = agent_manager.pause_agent(user_id)

        if not success:
            return jsonify({'error': error_msg}), 400

        return jsonify({
            'success': True,
            'message': 'Agent paused successfully'
        })

    except Exception as e:
        print(f"Error in /agent/pause: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/agent/resume', methods=['POST'])
def resume_agent():
    """
    Resume paused trading agent for user.

    Expects JSON body:
    {
        "user_id": "user123"
    }

    Returns:
    {
        "success": true,
        "message": "Agent resumed successfully"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        user_id = data.get('user_id')

        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400

        # Resume agent
        success, error_msg = agent_manager.resume_agent(user_id)

        if not success:
            return jsonify({'error': error_msg}), 400

        return jsonify({
            'success': True,
            'message': 'Agent resumed successfully'
        })

    except Exception as e:
        print(f"Error in /agent/resume: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/agent/status', methods=['GET'])
def get_agent_status():
    """
    Get agent status for user.

    Query params:
        user_id: User identifier

    Returns:
    {
        "session": {...},
        "decision_count": 123,
        "decision_summary": {...}
    }
    """
    try:
        user_id = request.args.get('user_id')

        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400

        # Get session
        session = agent_manager.get_session(user_id)

        if not session:
            return jsonify({
                'session': None,
                'decision_count': 0,
                'decision_summary': {
                    'total_decisions': 0,
                    'hold_count': 0,
                    'enter_count': 0,
                    'exit_count': 0
                }
            })

        # Get decision stats
        decision_count = decision_logger.get_decision_count(user_id)
        decision_summary = decision_logger.get_decision_summary(user_id)

        return jsonify({
            'session': session.to_dict(),
            'decision_count': decision_count,
            'decision_summary': decision_summary
        })

    except Exception as e:
        print(f"Error in /agent/status: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/agent/config', methods=['PUT'])
def update_agent_config():
    """
    Update agent configuration (only allowed when stopped).

    Expects JSON body:
    {
        "user_id": "user123",
        "config": {
            "max_leverage": 2.0,
            "target_utilization": 0.8,
            "max_positions": 3
        }
    }

    Returns:
    {
        "success": true,
        "message": "Configuration updated successfully"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        user_id = data.get('user_id')
        config_data = data.get('config', {})

        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400

        # Parse config
        config = AgentConfig.from_dict(config_data)

        # Update config
        success, error_msg = agent_manager.update_config(user_id, config)

        if not success:
            return jsonify({'error': error_msg}), 400

        return jsonify({
            'success': True,
            'message': 'Configuration updated successfully'
        })

    except Exception as e:
        print(f"Error in /agent/config: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/agent/decisions', methods=['GET'])
def get_agent_decisions():
    """
    Get recent decisions for user.

    Query params:
        user_id: User identifier
        limit: Maximum number of decisions to return (default: 100)

    Returns:
    {
        "decisions": [
            {
                "timestamp": "2025-11-04T12:34:56",
                "action": "ENTER",
                "opportunity_symbol": "BTCUSDT",
                "confidence": "HIGH",
                ...
            },
            ...
        ]
    }
    """
    try:
        user_id = request.args.get('user_id')
        limit = request.args.get('limit', 100, type=int)

        if not user_id:
            return jsonify({'error': 'user_id is required'}), 400

        # Get decisions
        decisions = decision_logger.get_recent_decisions(user_id, limit)

        return jsonify({
            'decisions': decisions
        })

    except Exception as e:
        print(f"Error in /agent/decisions: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/agent/decisions/update', methods=['POST'])
def update_agent_decision():
    """
    Update decision with execution results.

    Called by backend after execution completes.

    Expects JSON body:
    {
        "user_id": "user123",
        "timestamp": "2025-11-04T12:34:56",
        "execution_status": "filled" | "failed",
        "execution_id": 123,
        "filled_price": 50000.0,
        "filled_amount_usd": 1000.0,
        "error_message": "...",  # if failed
        "profit_usd": 50.0,  # for EXIT
        "profit_pct": 5.0,  # for EXIT
        "duration_hours": 24.5  # for EXIT
    }

    Returns:
    {
        "success": true,
        "message": "Decision updated successfully"
    }
    """
    try:
        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        user_id = data.get('user_id')
        timestamp_str = data.get('timestamp')

        if not user_id or not timestamp_str:
            return jsonify({'error': 'user_id and timestamp are required'}), 400

        # Parse timestamp
        from datetime import datetime
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))

        # Update decision
        success = decision_logger.update_decision_execution(
            user_id=user_id,
            timestamp=timestamp,
            execution_status=data.get('execution_status', 'pending'),
            execution_id=data.get('execution_id'),
            filled_price=data.get('filled_price'),
            filled_amount_usd=data.get('filled_amount_usd'),
            error_message=data.get('error_message'),
            profit_usd=data.get('profit_usd'),
            profit_pct=data.get('profit_pct'),
            duration_hours=data.get('duration_hours')
        )

        if success:
            return jsonify({
                'success': True,
                'message': 'Decision updated successfully'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Decision not found or could not be updated'
            }), 404

    except Exception as e:
        print(f"Error in /agent/decisions/update: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ============================================================================
# RL PREDICTION ENDPOINT
# ============================================================================

@app.route('/rl/predict', methods=['POST'])
def predict():
    """
    RL Prediction API - Uses UnifiedFeatureBuilder architecture.

    This endpoint uses the refactored architecture with:
    - Pydantic schema validation for raw data
    - UnifiedFeatureBuilder for feature preparation
    - Single source of truth for all feature engineering

    Expects JSON body:
    {
        "trading_config": {
            "max_leverage": 2.0,
            "target_utilization": 0.8,
            "max_positions": 3,
            "stop_loss_threshold": -0.02,
            "liquidation_buffer": 0.15
        },
        "portfolio": {
            "total_capital": 10000.0,
            "capital_utilization": 0.5,
            "positions": [...]
        },
        "opportunities": [...]
    }

    Returns:
    {
        "action": "ENTER" | "HOLD" | "EXIT",
        "action_id": 5,
        "confidence": 0.75,
        "state_value": 150.5,
        "opportunity_symbol": "BTCUSDT",  # If ENTER action
        "opportunity_index": 2,           # If ENTER action
        "position_size": "MEDIUM",        # If ENTER action
        "size_multiplier": 0.20,          # If ENTER action
        "position_index": 0,              # If EXIT action
        "model_info": {...}
    }
    """
    try:
        if rl_predictor is None:
            return jsonify({'error': 'RL predictor not initialized'}), 503

        data = request.get_json()

        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # LOG RAW REQUEST DATA BEFORE VALIDATION FOR DEBUGGING
        try:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'data': data
            }
            with open('/tmp/ml_api_requests_raw.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Failed to log raw request data: {e}")

        # Validate input with Pydantic
        try:
            validated_request = RLRawDataRequest(**data)
        except Exception as validation_error:
            return jsonify({
                'error': 'Validation failed',
                'details': str(validation_error)
            }), 400

        # Convert to dict for predictor
        raw_data_dict = validated_request.dict()

        # LOG VALIDATED REQUEST DATA FOR DEBUGGING
        try:
            log_entry = {
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'data': raw_data_dict
            }
            with open('/tmp/ml_api_requests_validated.jsonl', 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
        except Exception as e:
            print(f"Warning: Failed to log validated request data: {e}")

        # LOG REQUEST DATA
        print(f"\n========== ML API REQUEST ==========")
        print(f"Num Opportunities: {len(raw_data_dict['opportunities'])}")
        print(f"Portfolio positions: {len(raw_data_dict['portfolio'].get('positions', []))}")
        print(f"Trading config: max_leverage={raw_data_dict['trading_config'].get('max_leverage')}, "
              f"max_positions={raw_data_dict['trading_config'].get('max_positions')}")
        print(f"====================================\n")

        # Make prediction
        prediction = rl_predictor.predict_opportunities(
            opportunities=raw_data_dict['opportunities'],
            portfolio=raw_data_dict['portfolio'],
            trading_config=raw_data_dict['trading_config']
        )

        # Apply confidence thresholds
        # Low-confidence ENTER/EXIT actions are converted to HOLD
        original_action = prediction.get('action')
        original_action_id = prediction.get('action_id')
        confidence = prediction.get('confidence', 0.0)

        threshold_applied = False
        if original_action == 'ENTER' and confidence < ENTER_CONFIDENCE_THRESHOLD:
            prediction['action'] = 'HOLD'
            prediction['action_id'] = 0
            prediction['threshold_blocked'] = True
            prediction['original_action'] = original_action
            prediction['original_action_id'] = original_action_id
            threshold_applied = True
        elif original_action == 'EXIT' and confidence < EXIT_CONFIDENCE_THRESHOLD:
            prediction['action'] = 'HOLD'
            prediction['action_id'] = 0
            prediction['threshold_blocked'] = True
            prediction['original_action'] = original_action
            prediction['original_action_id'] = original_action_id
            threshold_applied = True

        # LOG RESPONSE DATA
        print(f"\n========== ML API RESPONSE ==========")
        print(f"Action: {prediction.get('action', 'N/A')}")
        if threshold_applied:
            print(f"  (Original: {original_action} blocked by threshold, conf={confidence:.1%})")
        print(f"Opportunity Symbol: {prediction.get('opportunity_symbol', 'N/A')}")
        print(f"Opportunity Index: {prediction.get('opportunity_index', 'N/A')}")
        print(f"Confidence: {prediction.get('confidence', 'N/A')}")
        print(f"State Value: {prediction.get('state_value', 'N/A')}")
        print(f"====================================\n")

        # Get model info
        model_info = rl_predictor.get_model_info()
        prediction['model_info'] = model_info

        # Log raw input/output for analysis
        log_raw_decision(
            request_data=raw_data_dict,
            response_data=prediction,
            user_id=raw_data_dict['portfolio'].get('user_id')
        )

        return jsonify(prediction)

    except Exception as e:
        print(f"Error in /rl/predict: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # Initialize predictor before starting server
    initialize_predictor()

    # Run Flask server
    print("\n" + "="*80)
    print("ML API Server - RL with Unified Feature Builder")
    print("="*80)
    print(f"Starting server on http://localhost:5250")
    print(f"\nEndpoints:")
    print(f"  GET  /health                       - Health check")
    print(f"\nAgent Management:")
    print(f"  POST /agent/start                  - Start autonomous trading agent")
    print(f"  POST /agent/stop                   - Stop trading agent")
    print(f"  POST /agent/pause                  - Pause trading agent")
    print(f"  POST /agent/resume                 - Resume paused agent")
    print(f"  GET  /agent/status?user_id=X       - Get agent status and stats")
    print(f"  PUT  /agent/config                 - Update agent configuration")
    print(f"  GET  /agent/decisions?user_id=X    - Get recent decisions")
    print(f"\nRL Predictions:")
    print(f"  POST /rl/predict                   - RL prediction (Unified Feature Builder)")
    print("="*80 + "\n")

    app.run(host='0.0.0.0', port=5250, debug=False)
