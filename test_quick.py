"""Quick test for GPU SHAP."""
import sys

# Test 1: Import
try:
    from gpu_shap import GPUExplainer
    print("[OK] GPUExplainer import")
except Exception as e:
    print(f"[FAIL] import: {e}")

# Test 2: Basic usage with simple model
try:
    import torch
    import numpy as np

    device = torch.device('cuda')

    # Simple linear model as test
    def model_fn(X):
        """X: [batch, features] -> [batch]"""
        weights = torch.tensor([1.0, 2.0, -1.0, 0.5], device=device)
        return (X * weights).sum(dim=1)

    # Background data
    bg = torch.randn(50, 4, device=device)

    # Create explainer
    explainer = GPUExplainer(model_fn, bg)
    print("[OK] GPUExplainer created")

    # Compute SHAP values
    test_data = torch.randn(10, 4, device=device)
    shap_vals = explainer.shap_values(test_data)
    print(f"[OK] SHAP values computed: shape={shap_vals.shape}")

    # Feature importance
    imp = explainer.feature_importance()
    print(f"[OK] Feature importance: {imp}")

except Exception as e:
    print(f"[FAIL] usage: {e}")
    import traceback; traceback.print_exc()

# Test 3: Demo with sklearn (the problematic part)
try:
    sys.path.insert(0, r"C:\Users\salih\Desktop\py2tensor")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification
    from sklearn_to_gpu import convert_rf
    print("[OK] sklearn_to_gpu available")
except Exception as e:
    print(f"[SKIP] sklearn demo: {e}")
