from setuptools import setup, find_packages

setup(
    name="docimind",
    version="1.0.0",
    author="DociMind Engineering Team",
    description="DociMind — Intelligent Document Classification & Information Extraction Platform",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
    install_requires=[
        "streamlit>=1.30.0",
        "easyocr>=1.7.1",
        "opencv-python-headless>=4.8.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "joblib>=1.3.2",
        "spacy>=3.7.2",
        "pillow>=10.2.0",
        "lightgbm>=4.3.0",
        "xgboost>=2.0.0",
    ],
)

