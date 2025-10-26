# C# ONNX Integration Guide

## Prerequisites

```bash
dotnet add package Microsoft.ML.OnnxRuntime --version 1.16.3
```

## Load Models

```csharp
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

public class OpportunityMLScorer
{
    private readonly InferenceSession _profitModel;
    private readonly InferenceSession _successModel;
    private readonly InferenceSession _durationModel;

    public OpportunityMLScorer(string modelDir)
    {
        _profitModel = new InferenceSession($"{modelDir}/profit_model.onnx");
        _successModel = new InferenceSession($"{modelDir}/success_model.onnx");
        _durationModel = new InferenceSession($"{modelDir}/duration_model.onnx");
    }
}
```

## Extract Features

```csharp
private float[] ExtractFeatures(ArbitrageOpportunityDto opp)
{
    var features = new float[54];

    // Raw features
    features[0] = (float)opp.HourOfDay;
    features[1] = (float)opp.DayOfWeek;
    features[2] = (float)opp.LongFundingRate;
    features[3] = (float)opp.ShortFundingRate;
    features[4] = (float)opp.LongFundingIntervalHours;
    features[5] = (float)opp.ShortFundingIntervalHours;
    features[6] = (float)opp.LongNextFundingMinutes;
    features[7] = (float)opp.ShortNextFundingMinutes;
    features[8] = (float)opp.CurrentPriceSpreadPct;
    features[9] = (float)opp.FundProfit8H;
    // ... (see feature_mapping.md for complete list)

    return features;
}
```

## Run Inference

```csharp
public MLPrediction ScoreOpportunity(ArbitrageOpportunityDto opp)
{
    var features = ExtractFeatures(opp);
    var inputTensor = new DenseTensor<float>(features, new[] { 1, features.Length });

    var inputs = new List<NamedOnnxValue>
    {
        NamedOnnxValue.CreateFromTensor("float_input", inputTensor)
    };

    // Get predictions
    using var profitResults = _profitModel.Run(inputs);
    using var successResults = _successModel.Run(inputs);
    using var durationResults = _durationModel.Run(inputs);

    var predictedProfit = profitResults.First().AsEnumerable<float>().First();
    var successProbability = successResults.First().AsEnumerable<float>().ElementAt(1);
    var predictedDuration = durationResults.First().AsEnumerable<float>().First();

    return new MLPrediction
    {
        PredictedProfitPercent = (decimal)predictedProfit,
        SuccessProbability = (decimal)successProbability,
        PredictedHoldHours = (decimal)predictedDuration
    };
}
```

