"""
ONNX Export Utilities

Helper functions for exporting models to ONNX format for C# integration.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np


def validate_onnx_model(onnx_path: Path, feature_names: List[str], test_input: Optional[np.ndarray] = None) -> bool:
    """
    Validate ONNX model can be loaded and run inference.

    Args:
        onnx_path: Path to ONNX model
        feature_names: List of feature names
        test_input: Optional test input for inference (shape: [1, n_features])

    Returns:
        True if validation passes
    """
    try:
        import onnx
        import onnxruntime as ort

        # Load and check model
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)

        print(f"✅ ONNX model structure valid: {onnx_path.name}")

        # Test inference
        session = ort.InferenceSession(str(onnx_path))

        # Get input name
        input_name = session.get_inputs()[0].name

        # Create test input if not provided
        if test_input is None:
            test_input = np.random.randn(1, len(feature_names)).astype(np.float32)

        # Run inference
        outputs = session.run(None, {input_name: test_input})

        print(f"✅ ONNX inference test passed: {onnx_path.name}")
        print(f"   Input shape: {test_input.shape}")
        print(f"   Output shape: {outputs[0].shape}")

        return True

    except Exception as e:
        print(f"❌ ONNX validation failed: {e}")
        return False


def export_feature_mapping(
    feature_names: List[str],
    output_path: Path,
    include_descriptions: bool = True
) -> None:
    """
    Export feature mapping for C# integration.

    Creates a file documenting:
    - Feature names in exact order
    - Feature indices
    - Brief descriptions

    Args:
        feature_names: List of feature names (in order)
        output_path: Path to save mapping file
        include_descriptions: Include feature descriptions
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# Feature Mapping for C# ONNX Inference\n\n")
        f.write(f"Total Features: {len(feature_names)}\n\n")
        f.write("## Feature Order\n\n")
        f.write("**IMPORTANT**: Features must be provided in this exact order for ONNX inference.\n\n")

        f.write("| Index | Feature Name | Description |\n")
        f.write("|-------|--------------|-------------|\n")

        for idx, feature in enumerate(feature_names):
            # Basic description based on feature name
            description = _get_feature_description(feature) if include_descriptions else ""
            f.write(f"| {idx} | `{feature}` | {description} |\n")

        f.write("\n## C# Example\n\n")
        f.write("```csharp\n")
        f.write("// Create feature array (must match order above)\n")
        f.write(f"var features = new float[{len(feature_names)}];\n\n")

        # Show first few features as example
        for idx, feature in enumerate(feature_names[:5]):
            safe_name = feature.replace('-', '_').replace('.', '_')
            f.write(f"features[{idx}] = (float)opportunity.{_to_pascal_case(safe_name)};\n")

        f.write("// ... (continue for all features)\n")
        f.write("\n// Create ONNX tensor\n")
        f.write("var inputTensor = new DenseTensor<float>(features, new[] { 1, features.Length });\n")
        f.write("```\n")

    print(f"✅ Feature mapping saved to {output_path}")


def _get_feature_description(feature_name: str) -> str:
    """Get basic description for a feature."""
    descriptions = {
        'fund_profit_8h': '8-hour profit % (current rate)',
        'fund_apr': 'Annualized percentage rate',
        'fund_profit_8h_24h_proj': '8h profit using 24h avg rate',
        'fund_profit_8h_3d_proj': '8h profit using 3D avg rate',
        'break_even_hours': 'Hours to recover position cost',
        'spread_volatility_cv': 'Spread volatility (CV)',
        'volume_24h': '24-hour trading volume',
        'btc_price_at_entry': 'BTC price at entry',
        'hour_sin': 'Hour (sine encoding)',
        'hour_cos': 'Hour (cosine encoding)',
        'day_sin': 'Day of week (sine encoding)',
        'day_cos': 'Day of week (cosine encoding)',
        'rate_momentum_24h': 'Rate change vs 24h avg',
        'rate_momentum_3d': 'Rate change vs 3D avg',
        'volatility_risk_score': 'Volatility risk (0-1)',
        'liquidity_score': 'Liquidity quality (0-1)',
        'profit_x_liquidity': 'Profit weighted by liquidity',
        'risk_adjusted_return': 'Profit / (1 + volatility)'
    }

    return descriptions.get(feature_name, '')


def _to_pascal_case(snake_str: str) -> str:
    """Convert snake_case to PascalCase."""
    components = snake_str.split('_')
    return ''.join(x.title() for x in components)


def create_csharp_integration_guide(
    model_dir: Path,
    feature_names: List[str],
    output_path: Path
) -> None:
    """
    Create C# integration guide with code examples.

    Args:
        model_dir: Directory containing ONNX models
        feature_names: List of feature names
        output_path: Path to save guide
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("# C# ONNX Integration Guide\n\n")
        f.write("## Prerequisites\n\n")
        f.write("```bash\n")
        f.write("dotnet add package Microsoft.ML.OnnxRuntime --version 1.16.3\n")
        f.write("```\n\n")

        f.write("## Load Models\n\n")
        f.write("```csharp\n")
        f.write("using Microsoft.ML.OnnxRuntime;\n")
        f.write("using Microsoft.ML.OnnxRuntime.Tensors;\n\n")
        f.write("public class OpportunityMLScorer\n")
        f.write("{\n")
        f.write("    private readonly InferenceSession _profitModel;\n")
        f.write("    private readonly InferenceSession _successModel;\n")
        f.write("    private readonly InferenceSession _durationModel;\n\n")
        f.write("    public OpportunityMLScorer(string modelDir)\n")
        f.write("    {\n")
        f.write("        _profitModel = new InferenceSession($\"{modelDir}/profit_model.onnx\");\n")
        f.write("        _successModel = new InferenceSession($\"{modelDir}/success_model.onnx\");\n")
        f.write("        _durationModel = new InferenceSession($\"{modelDir}/duration_model.onnx\");\n")
        f.write("    }\n")
        f.write("}\n")
        f.write("```\n\n")

        f.write("## Extract Features\n\n")
        f.write("```csharp\n")
        f.write(f"private float[] ExtractFeatures(ArbitrageOpportunityDto opp)\n")
        f.write("{\n")
        f.write(f"    var features = new float[{len(feature_names)}];\n\n")
        f.write("    // Raw features\n")

        for idx, feature in enumerate(feature_names[:10]):
            safe_name = _to_pascal_case(feature.replace('-', '_').replace('.', '_'))
            f.write(f"    features[{idx}] = (float)opp.{safe_name};\n")

        f.write("    // ... (see feature_mapping.md for complete list)\n\n")
        f.write("    return features;\n")
        f.write("}\n")
        f.write("```\n\n")

        f.write("## Run Inference\n\n")
        f.write("```csharp\n")
        f.write("public MLPrediction ScoreOpportunity(ArbitrageOpportunityDto opp)\n")
        f.write("{\n")
        f.write("    var features = ExtractFeatures(opp);\n")
        f.write("    var inputTensor = new DenseTensor<float>(features, new[] { 1, features.Length });\n\n")
        f.write("    var inputs = new List<NamedOnnxValue>\n")
        f.write("    {\n")
        f.write("        NamedOnnxValue.CreateFromTensor(\"float_input\", inputTensor)\n")
        f.write("    };\n\n")
        f.write("    // Get predictions\n")
        f.write("    using var profitResults = _profitModel.Run(inputs);\n")
        f.write("    using var successResults = _successModel.Run(inputs);\n")
        f.write("    using var durationResults = _durationModel.Run(inputs);\n\n")
        f.write("    var predictedProfit = profitResults.First().AsEnumerable<float>().First();\n")
        f.write("    var successProbability = successResults.First().AsEnumerable<float>().ElementAt(1);\n")
        f.write("    var predictedDuration = durationResults.First().AsEnumerable<float>().First();\n\n")
        f.write("    return new MLPrediction\n")
        f.write("    {\n")
        f.write("        PredictedProfitPercent = (decimal)predictedProfit,\n")
        f.write("        SuccessProbability = (decimal)successProbability,\n")
        f.write("        PredictedHoldHours = (decimal)predictedDuration\n")
        f.write("    };\n")
        f.write("}\n")
        f.write("```\n\n")

    print(f"✅ C# integration guide saved to {output_path}")
