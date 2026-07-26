import numpy as np
import pandas as pd
from typing import List, Dict, Any, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from docimind.ml.text_cleaner import TextCleaner


class FeatureExtractor:
    """
    Feature Engineering class that combines TF-IDF n-gram text vectorization
    with statistical and domain-specific text heuristic features.
    """

    KEYWORD_INDICATORS = [
        "invoice", "receipt", "total", "subtotal", "tax", "vat",
        "curriculum vitae", "resume", "skills", "experience", "education",
        "patient", "doctor", "diagnosis", "hospital", "lab report",
        "passport", "republic", "nationality", "surname", "mrz",
        "identity card", "national id", "dob", "address",
        "driver license", "dl no", "class", "expires",
        "utility bill", "kwh", "meter", "account number", "due date",
        "bank statement", "balance", "credit", "debit", "deposit", "transaction"
    ]

    def __init__(self, max_features: int = 5000, ngram_range: tuple = (1, 3)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True
        )

    def extract_heuristic_features(self, raw_text: str) -> Dict[str, float]:
        """
        Extracts structural metrics and domain keyword counts from text.
        """
        clean_text = TextCleaner.clean_text(raw_text, preserve_case=True)
        char_count = len(clean_text)
        word_count = len(clean_text.split())

        if char_count == 0:
            return {kw: 0.0 for kw in self.KEYWORD_INDICATORS} | {
                "word_count": 0.0, "char_count": 0.0, "digit_ratio": 0.0, "upper_ratio": 0.0
            }

        digit_count = sum(c.isdigit() for c in clean_text)
        upper_count = sum(c.isupper() for c in clean_text)

        features = {
            "word_count": float(word_count),
            "char_count": float(char_count),
            "digit_ratio": float(digit_count / char_count),
            "upper_ratio": float(upper_count / char_count)
        }

        lower_text = clean_text.lower()
        for kw in self.KEYWORD_INDICATORS:
            features[f"kw_{kw.replace(' ', '_')}"] = float(lower_text.count(kw))

        return features

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fits TF-IDF vectorizer on training text corpus."""
        cleaned_texts = [TextCleaner.clean_for_classification(t) for t in texts]
        return self.vectorizer.fit_transform(cleaned_texts)

    def transform(self, texts: List[str]) -> np.ndarray:
        """Transforms input texts using fitted TF-IDF vectorizer."""
        cleaned_texts = [TextCleaner.clean_for_classification(t) for t in texts]
        return self.vectorizer.transform(cleaned_texts)
