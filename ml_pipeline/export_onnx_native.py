"""
Export Trained Models to ONNX Format using XGBoost Native Export

Uses XGBoost's built-in save_model with ONNX format (available in XGBoost >= 2.0)
"""

import argparse
import joblib
import json
from pathlib import Path
import sys
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def export_model_to_onnx_native(model_dict, feature_names, output_path):
    """Export a model using XGBoost's native ONNX export."""
    try:
        import xgboost as xgb

        model = model_dict['model']

        # Get the booster (underlying XGBoost model)
        booster = model.get_booster()

        # XGBoost 2.0+ supports native ONNX export
        # But we need to save via JSON first, then convert
        onnx_path = output_path.with_suffix('.onnx')

        # Try direct ONNX export (if supported)
        try:
            booster.save_model(str(onnx_path), format='onnx')
            print(f"✅ Exported: {onnx_path.name}")
            return True, onnx_path
        except Exception as e:
            print(f"   Native ONNX export not available: {e}")
            print(f"   Falling back to ubj format...")

            # Fall back to UBJ (Universal Binary JSON) format
            # This is XGBoost's binary format that can be loaded cross-platform
            ubj_path = output_path.with_suffix('.ubj')
            booster.save_model(str(ubj_path), format='ubj')
            print(f"✅ Exported: {ubj_path.name} (UBJ format)")
            return True, ubj_path

    except Exception as e:
        print(f"❌ Export failed for {output_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def main():
    parser = argparse.ArgumentParser(description='Export XGBoost models to ONNX/UBJ')
    parser.add_argument('--model-dir', type=str, default='models/xgboost',
                        help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default='../src/CryptoArbitrage.API/models/ml',
                        help='Output directory for exported models')

    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("EXPORTING MODELS TO ONNX/UBJ FORMAT")
    print("="*80)
    print(f"\nModel directory: {model_dir}")
    print(f"Output directory: {output_dir}")

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
        'profit_model.pkl': 'profit_model',
        'success_model.pkl': 'success_model',
        'duration_model.pkl': 'duration_model'
    }

    print("\nExporting models...")
    success_count = 0
    exported_files = []

    for pkl_name, model_name in models.items():
        print(f"\n📦 Processing {pkl_name}...")

        # Load model
        model_dict = joblib.load(model_dir / pkl_name)

        # Export (will use ONNX if available, otherwise UBJ)
        output_path = output_dir / model_name
        success, exported_path = export_model_to_onnx_native(model_dict, feature_names, output_path)
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

    export_format = "UBJ" if any('.ubj' in str(f) for f in exported_files) else "ONNX"
    print(f"\n📝 Note: Models exported in {export_format} format")
    if export_format == "UBJ":
        print(f"   UBJ is XGBoost's cross-platform binary format")
        print(f"   Can be loaded with XGBoost in any language")

    if success_count == len(models):
        print("\n🎉 All models exported successfully!")
        return 0
    else:
        print(f"\n⚠️  {len(models) - success_count} models failed to export")
        return 1


if __name__ == '__main__':
    sys.exit(main())
