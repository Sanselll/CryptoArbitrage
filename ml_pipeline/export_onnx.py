"""
Export Trained Models to ONNX Format

Exports XGBoost models from pickle format to ONNX for C# integration.
Uses onnxmltools with proper configuration for XGBoost models.
"""

import argparse
import joblib
import json
from pathlib import Path
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def export_model_to_onnx(model_dict, feature_names, output_path):
    """Export a model dictionary to ONNX format."""
    try:
        import onnxmltools
        from onnxmltools.convert.common.data_types import FloatTensorType
        import xgboost as xgb

        model = model_dict['model']

        # Get the booster (underlying XGBoost model)
        booster = model.get_booster()

        # IMPORTANT: For onnxmltools, we need to use feature indices, not names
        # So we'll convert the model to use numeric feature names
        # Save current feature names for later
        current_feature_names = booster.feature_names

        # Set feature names to f0, f1, f2, ... (required by onnxmltools)
        numeric_feature_names = [f'f{i}' for i in range(len(feature_names))]
        booster.feature_names = numeric_feature_names

        # Define initial types for ONNX conversion
        # Input shape: [batch_size, num_features]
        initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]

        # Convert to ONNX
        onnx_model = onnxmltools.convert_xgboost(
            booster,
            initial_types=initial_type,
            target_opset=12  # Use opset 12 for better compatibility
        )

        # Save ONNX model
        onnx_path = output_path.with_suffix('.onnx')
        onnxmltools.utils.save_model(onnx_model, str(onnx_path))

        # Restore original feature names
        booster.feature_names = current_feature_names

        print(f"✅ Exported: {onnx_path.name}")

        return True, onnx_path

    except Exception as e:
        print(f"❌ Export failed for {output_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    parser = argparse.ArgumentParser(description='Export XGBoost models to ONNX')
    parser.add_argument('--model-dir', type=str, default='models/xgboost',
                        help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default='../src/CryptoArbitrage.API/models/ml',
                        help='Output directory for ONNX models')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EXPORTING MODELS TO ONNX FORMAT")
    print("="*80)
    print(f"\nModel directory: {model_dir}")
    print(f"Output directory: {output_dir}")

    # Check if onnxmltools is installed
    try:
        import onnxmltools
    except ImportError:
        print("\n❌ onnxmltools not installed. Installing required packages...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxmltools", "onnx"])
        import onnxmltools

    # Load preprocessor to get feature names
    print("\nLoading preprocessor...")
    preprocessor_data = joblib.load(model_dir / 'preprocessor.pkl')
    feature_names = preprocessor_data['feature_names']
    scaler = preprocessor_data['scaler']
    config = preprocessor_data['config']

    print(f"✅ Found {len(feature_names)} features")

    # Save preprocessor parameters for C#
    print("\nSaving preprocessor parameters...")
    preprocessor_params = {
        'feature_names': feature_names,
        'scaler_type': config.get('scaler_type', 'standard'),
        'scaler_mean': scaler.mean_.tolist() if hasattr(scaler, 'mean_') else None,
        'scaler_scale': scaler.scale_.tolist() if hasattr(scaler, 'scale_') else None,
        'scaler_var': scaler.var_.tolist() if hasattr(scaler, 'var_') else None
    }

    with open(output_dir / 'preprocessor_params.json', 'w') as f:
        json.dump(preprocessor_params, f, indent=2)

    print(f"✅ Saved preprocessor parameters")

    # Export each model
    models = {
        'profit_model.pkl': 'profit_model.onnx',
        'success_model.pkl': 'success_model.onnx',
        'duration_model.pkl': 'duration_model.onnx'
    }

    print("\nExporting models...")
    success_count = 0
    exported_files = []

    for pkl_name, onnx_name in models.items():
        print(f"\n📦 Processing {pkl_name}...")

        # Load model
        model_dict = joblib.load(model_dir / pkl_name)

        # Export to ONNX
        output_path = output_dir / onnx_name
        success, exported_path = export_model_to_onnx(model_dict, feature_names, output_path)
        if success:
            success_count += 1
            exported_files.append(exported_path)

    # Summary
    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print(f"\n✅ Successfully exported {success_count}/{len(models)} models")
    print(f"\nOutput files:")
    for file_path in exported_files:
        print(f"  • {file_path}")
    print(f"  • {output_dir / 'preprocessor_params.json'}")

    print(f"\n📝 Note: Models exported in ONNX format (opset 12)")
    print(f"   Use Microsoft.ML.OnnxRuntime NuGet package in C#")

    if success_count == len(models):
        print("\n🎉 All models exported successfully!")
        return 0
    else:
        print(f"\n⚠️  {len(models) - success_count} models failed to export")
        return 1


if __name__ == '__main__':
    sys.exit(main())
