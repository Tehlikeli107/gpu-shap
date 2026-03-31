from setuptools import setup

setup(
    name="gpu-shap",
    version="1.0.0",
    description="O(1) GPU-native SHAP implementation. First GPU-native Shapley values.",
    author="Salih Can Kurnaz",
    url="https://github.com/Tehlikeli107/gpu-shap",
    py_modules=["gpu_shap"],
    install_requires=["torch>=2.0", "numpy"],
    extras_require={"sklearn": ["scikit-learn"]},
    python_requires=">=3.8",
)
