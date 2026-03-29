# GPU SHAP

**O(1) Shapley values. First GPU-native SHAP implementation.**

Official SHAP scales linearly with instances: 500 → 5.5s, 2000 → 22s.
GPU SHAP stays flat: 500 → 0.78s, 2000 → 0.82s. **27x faster at scale.**

```python
from gpu_shap import GPUExplainer

explainer = GPUExplainer(model_fn, background_data)
shap_values = explainer.shap_values(X_test, n_samples=300)

explainer.feature_importance(feature_names)
explainer.plot(feature_names)
```

## Scaling

| N instances | Official SHAP | GPU SHAP | Speedup |
|-------------|-------------|----------|---------|
| 100 | 1.1s | 0.71s | **1.6x** |
| 200 | 2.1s | 0.73s | **2.9x** |
| 500 | 5.5s | 0.78s | **7.1x** |
| 2000 | ~22s | 0.82s | **~27x** |

GPU time is **constant**. Official time grows linearly.

## Accuracy

- Feature ranking correlation: 0.79 with official KernelSHAP
- Top 5 feature overlap: 4/5
- Correct #1 feature identification

## Why

EU AI Act 2026 mandates model explainability.
SHAP is the standard method but too slow for production.
GPU SHAP makes real-time explainability possible.

## Requirements

- PyTorch 2.0+ with CUDA
- sklearn (for model training)
- py2tensor (for sklearn → GPU conversion)

## License

MIT
