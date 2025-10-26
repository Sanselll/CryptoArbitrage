"""
Export Trained Models to ONNX Format

Exports XGBoost models from pickle format to ONNX for C# integration.
"""

import argparse
import joblib
import json
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from export.onnx_exporter import validate_onnx_model, export_feature_mapping, create_csharp_integration_guide


def export_model_to_json(model_dict, feature_names, output_path, model_type):
    """Export a model dictionary to XGBoost JSON format with metadata."""
    try:
        import xgboost as xgb

        model = model_dict['model']
        config = model_dict.get('config', {})

        # Get the booster (underlying XGBoost model)
        booster = model.get_booster()

        # Save in XGBoost's native JSON format
        json_path = output_path.with_suffix('.json')
        booster.save_model(str(json_path))

        print(f"✅ Exported: {json_path.name}")

        # Extract objective from model config
        objective = config.get('objective', 'reg:squarederror')
        is_classifier = 'logistic' in objective or 'binary' in objective
        requires_sigmoid = is_classifier

        # Define output constraints based on model type
        output_constraints = {}
        if model_type == 'success':
            output_constraints = {'min': 0.0, 'max': 1.0}
        elif model_type == 'duration':
            output_constraints = {'min': 0.0, 'max': None}
        elif model_type == 'profit':
            output_constraints = {'min': None, 'max': None}

        # Create metadata
        metadata = {
            'model_type': model_type,
            'objective': objective,
            'is_classifier': is_classifier,
            'requires_sigmoid': requires_sigmoid,
            'output_constraints': output_constraints,
            'feature_count': len(feature_names),
            'description': f'{model_type.capitalize()} prediction model'
        }

        # Save metadata as separate JSON file
        metadata_path = output_path.with_suffix('.metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Exported metadata: {metadata_path.name}")

        return True, json_path

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
        'profit_model.pkl': ('profit_model.json', 'profit'),
        'success_model.pkl': ('success_model.json', 'success'),
        'duration_model.pkl': ('duration_model.json', 'duration')
    }

    print("\nExporting models...")
    success_count = 0
    exported_files = []
    metadata_files = []

    for pkl_name, (json_name, model_type) in models.items():
        print(f"\n📦 Processing {pkl_name}...")

        # Load model
        model_dict = joblib.load(model_dir / pkl_name)

        # Export to JSON with metadata
        output_path = output_dir / json_name
        success, exported_path = export_model_to_json(model_dict, feature_names, output_path, model_type)
        if success:
            success_count += 1
            exported_files.append(exported_path)
            metadata_files.append(output_path.with_suffix('.metadata.json'))

    # Export feature mapping
    print("\nGenerating feature mapping documentation...")
    export_feature_mapping(
        feature_names,
        output_dir / 'feature_mapping.md'
    )

    # Create C# integration guide
    print("\nGenerating C# integration guide...")
    create_csharp_integration_guide(
        output_dir,
        feature_names,
        output_dir / 'csharp_integration_guide.md'
    )

    # Summary
    print("\n" + "="*80)
    print("EXPORT COMPLETE")
    print("="*80)
    print(f"\n✅ Successfully exported {success_count}/{len(models)} models")
    print(f"\nOutput files:")
    for file_path in exported_files:
        print(f"  • {file_path}")
    for metadata_path in metadata_files:
        print(f"  • {metadata_path}")
    print(f"  • {output_dir / 'preprocessor_params.json'}")
    print(f"  • {output_dir / 'feature_mapping.md'}")
    print(f"  • {output_dir / 'csharp_integration_guide.md'}")

    print(f"\n📝 Note: Models exported in XGBoost JSON format with metadata")
    print(f"   Each model has a corresponding .metadata.json file")
    print(f"   Metadata includes objective, sigmoid requirements, and output constraints")

    if success_count == len(models):
        print("\n🎉 All models exported successfully!")
        return 0
    else:
        print(f"\n⚠️  {len(models) - success_count} models failed to export")
        return 1


if __name__ == '__main__':
    sys.exit(main())
