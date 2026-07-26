import os
import joblib
import numpy as np
from typing import Dict, Any, Tuple
from docimind.config import CLASSIFIER_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH, SUPPORTED_DOCUMENTS
from docimind.ml.text_cleaner import TextCleaner


class DocumentClassifier:
    """
    Scikit-Learn Document Classification Inference Wrapper.
    Loads trained model, vectorizer, and label encoder exported via Joblib.
    Includes a fallback rule engine if model files are missing before training.
    """

    def __init__(
        self,
        model_path=CLASSIFIER_PATH,
        vectorizer_path=VECTORIZER_PATH,
        encoder_path=LABEL_ENCODER_PATH
    ):
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path
        self.encoder_path = encoder_path

        self.classifier = None
        self.vectorizer = None
        self.label_encoder = None
        self.is_loaded = False

        self._load_artifacts()

    def _load_artifacts(self):
        """Loads trained Joblib artifacts from models directory if available."""
        if (
            os.path.exists(self.model_path)
            and os.path.exists(self.vectorizer_path)
            and os.path.exists(self.encoder_path)
        ):
            try:
                self.classifier = joblib.load(self.model_path)
                self.vectorizer = joblib.load(self.vectorizer_path)
                self.label_encoder = joblib.load(self.encoder_path)
                self.is_loaded = True
            except Exception as e:
                print(f"Warning: Could not load trained joblib models ({e}). Using rule fallback.")
                self.is_loaded = False
        else:
            self.is_loaded = False

    def predict(self, raw_ocr_text: str) -> Dict[str, Any]:
        """
        Predicts document category and returns label + confidence score.

        Returns:
            Dict:
                - "predicted_class": String name of class (e.g., "Invoice")
                - "confidence": Float probability score (0.0 to 1.0)
                - "class_probabilities": Dict mapping class names to probabilities
        """
        clean_text = TextCleaner.clean_for_classification(raw_ocr_text)

        if not clean_text:
            return {
                "predicted_class": "Unknown",
                "confidence": 0.0,
                "class_probabilities": {doc: 0.0 for doc in SUPPORTED_DOCUMENTS}
            }

        if self.is_loaded and self.classifier and self.vectorizer:
            X = self.vectorizer.transform([clean_text])
            probs = self.classifier.predict_proba(X)[0]
            top_idx = int(np.argmax(probs))
            predicted_label = str(self.label_encoder.inverse_transform([top_idx])[0])
            confidence = float(probs[top_idx])

            class_probs = {
                str(cls): float(prob)
                for cls, prob in zip(self.label_encoder.classes_, probs)
            }

            return {
                "predicted_class": predicted_label,
                "confidence": round(confidence, 4),
                "class_probabilities": class_probs
            }
        else:
            # Fallback Rule-Based Classifier until Colab notebook generates Joblib files
            return self._heuristic_rule_predict(raw_ocr_text)

    def _heuristic_rule_predict(self, text: str) -> Dict[str, Any]:
        """Heuristic rule-based fallback classification when joblib model is not present."""
        lower = text.lower()
        scores = {doc: 0.1 for doc in SUPPORTED_DOCUMENTS}

        if "invoice" in lower or "bill to" in lower or "total due" in lower:
            scores["Invoice"] += 0.8
        if "receipt" in lower or "cashier" in lower or "change due" in lower:
            scores["Receipt"] += 0.8
        if "curriculum vitae" in lower or "resume" in lower or "work experience" in lower or "education" in lower:
            scores["Resume"] += 0.8
        if "patient" in lower or "blood pressure" in lower or "diagnosis" in lower or "prescription" in lower:
            scores["Medical Report"] += 0.8
        if "passport" in lower or "republic" in lower or "mrz" in lower or "P<" in lower:
            scores["Passport"] += 0.8
        if "national id" in lower or "identity card" in lower or "cnic" in lower:
            scores["National ID Card"] += 0.8
        if "driver license" in lower or "driving licence" in lower or "dl no" in lower:
            scores["Driver License"] += 0.8
        if "utility bill" in lower or "kwh" in lower or "meter number" in lower or "electricity" in lower:
            scores["Utility Bill"] += 0.8
        if "bank statement" in lower or "account balance" in lower or "opening balance" in lower or "closing balance" in lower:
            scores["Bank Statement"] += 0.8

        total = sum(scores.values())
        norm_probs = {k: v / total for k, v in scores.items()}

        best_class = max(norm_probs, key=norm_probs.get)
        confidence = norm_probs[best_class]

        return {
            "predicted_class": best_class,
            "confidence": round(confidence, 4),
            "class_probabilities": {k: round(v, 4) for k, v in norm_probs.items()}
        }
