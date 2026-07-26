# DociMind — Intelligent Document Classification & Information Extraction Platform

**Assigned Internship Topic:** OCR-based Document Information Extraction  
**Offer ID:** CAX-OL-2026-283  
**Role:** Principal AI / ML / CV Engineer Portfolio Project  

---

## 📌 Project Overview

**DociMind** is an end-to-end, production-grade document processing platform that seamlessly integrates 5 core AI pillars:

- 👁️ **Computer Vision (OpenCV)**: Automated deskewing, noise reduction, and adaptive thresholding.
- 🔤 **OCR Engine (EasyOCR)**: High-accuracy text bounding box detection and text extraction.
- 🤖 **Machine Learning (Scikit-Learn & LightGBM)**: Automatic document type classification across 9 categories.
- 🧠 **NLP & Named Entity Recognition (spaCy & Regex)**: Field-level key-value extraction for structured data.
- 🎨 **Streamlit Deployment**: Dynamic glassmorphic Web UI with interactive bounding box visualization and JSON/CSV export.

1. 📄 **Invoice**
2. 🧾 **Receipt**
3. 👤 **Resume**
4. 🏥 **Medical Report**
5. 🛂 **Passport**
6. 📇 **National ID Card**
7. 🪪 **Driver License**
8. 💡 **Utility Bill**
9. 🏦 **Bank Statement**

---

## 🛠️ Architecture & Tech Stack

- **Training Environment**: Google Colab (Python 3.12, GPU enabled)
- **Dataset Source**: Kaggle Datasets loaded directly via `kagglehub` API
- **Computer Vision**: OpenCV (Resizing, Grayscale, Noise Reduction, Adaptive Thresholding, Deskewing)
- **OCR Engine**: EasyOCR (Pretrained text & bounding box detection)
- **Machine Learning**: Scikit-Learn (TF-IDF Vectorization + Classifier models: Logistic Regression, Random Forest, SVM, LightGBM exported via `Joblib`)
- **NLP & Field Extraction**: spaCy Named Entity Recognition (NER) + Custom Regex Rule Engines
- **Frontend / Deployment**: Streamlit Dashboard with dynamic visual bounding box overlays, confidence metrics, JSON viewers, and CSV/JSON export capabilities.

---

## 📁 Repository Structure

```
DociMind/
├── notebooks/
│   └── 01_docimind_training_pipeline.ipynb   # Colab notebook for dataset acquisition, EDA, OCR dataset creation, model training & Joblib export
├── models/
│   ├── docimind_classifier.joblib             # Trained Scikit-Learn Classifier
│   ├── tfidf_vectorizer.joblib                # TF-IDF Feature Extractor
│   └── label_encoder.joblib                   # Document Class Encoder
├── src/
│   └── docimind/
│       ├── preprocessor/                      # OpenCV image preprocessing & deskewing
│       ├── ocr/                               # EasyOCR wrapper & bounding box extraction
│       ├── ml/                                # Text cleaning, vectorization & classifier inference
│       ├── nlp/                               # spaCy loader & 9 specialized document extractors
│       │   └── extractors/                    # Individual field extraction rule engines
│       ├── utils/                             # Bounding box visualizer & JSON/CSV exporters
│       └── pipeline.py                        # Unified end-to-end document orchestrator
├── app.py                                     # Streamlit Web Application entry point
├── requirements.txt                           # Dependency specifications
└── setup.py                                   # Package setup
```

---

## 🚀 Quick Start & Installation

### 1. Prerequisites & Virtual Environment

Ensure Python 3.12+ is installed.

```bash
git clone https://github.com/your-username/DociMind.git
cd DociMind
python -m venv venv
# Activate on Windows:
venv\Scripts\activate
# Activate on Linux/macOS:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Training / Model Generation (Colab)

Open `notebooks/01_docimind_training_pipeline.ipynb` in Google Colab to download datasets from Kaggle, run OCR text generation, train the document classification model, and export model `.joblib` files to the `models/` directory.

### 4. Run Streamlit Application Locally

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 📊 Features & Outputs

- 🔍 **Document Type Classification**: Real-time classification into 9 document categories with confidence probability scores.
- 🎯 **Visual Bounding Boxes**: Visual interactive overlay of detected text regions on uploaded document images.
- 📋 **Structured JSON Extraction**: Automated key-value entity extraction tailored per document class.
- 💾 **Export Manager**: Export results directly as standard `.json` or flattened `.csv`.

---

## 📜 License & Acknowledgments

Developed as part of the AI/ML Engineering Internship Program (Offer ID: `CAX-OL-2026-283`). Built with open-source libraries including OpenCV, EasyOCR, Scikit-Learn, spaCy, and Streamlit.
