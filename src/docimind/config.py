import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = BASE_DIR / "models"

CLASSIFIER_PATH = MODELS_DIR / "docimind_classifier.joblib"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"

# Supported Document Categories (9 Classes)
SUPPORTED_DOCUMENTS = [
    "Invoice",
    "Receipt",
    "Resume",
    "Medical Report",
    "Passport",
    "National ID Card",
    "Driver License",
    "Utility Bill",
    "Bank Statement"
]

# Standard Normalized Labels (lowercase slug mapping)
DOCUMENT_LABEL_MAP = {
    "invoice": "Invoice",
    "receipt": "Receipt",
    "resume": "Resume",
    "medical_report": "Medical Report",
    "passport": "Passport",
    "national_id": "National ID Card",
    "national_id_card": "National ID Card",
    "driver_license": "Driver License",
    "utility_bill": "Utility Bill",
    "bank_statement": "Bank Statement"
}

# Image Preprocessing Defaults
MAX_IMAGE_DIMENSION = 1920
DEFAULT_EASYOCR_LANGUAGES = ['en']

# Regex Common Patterns
EMAIL_REGEX = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
PHONE_REGEX = r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}'
DATE_REGEX = r'\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}[/-]\d{1,2}[/-]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*,?\s+\d{4})\b'

AMOUNT_REGEX = r'\$?\s?\b\d{1,3}(?:,\d{3})*(?:\.\d{2})?\b'
PASSPORT_NUM_REGEX = r'\b[A-PR-WYa-pr-wy][0-9]{7,8}\b'
