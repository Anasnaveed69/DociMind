from abc import ABC, abstractmethod
from typing import Dict, Any, List


class BaseFieldExtractor(ABC):
    """
    Abstract Base Class for Document-Specific Key Entity Extractors.
    Each document class (Invoice, Resume, Medical Report, etc.) inherits
    from this base class.
    """

    def __init__(self, spacy_nlp=None):
        self.nlp = spacy_nlp

    @abstractmethod
    def extract_fields(self, raw_text: str, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extracts key-value fields from OCR text and bounding box detections.

        Args:
            raw_text (str): Full concatenated text string from OCR.
            ocr_data (dict): Complete OCR detections with bounding boxes & confidences.

        Returns:
            Dict[str, Any]: Extracted structured fields.
        """
        pass

    @staticmethod
    def sanitize_field(val: Any) -> str:
        """Sanitizes extracted string value."""
        if val is None:
            return "N/A"
        cleaned = str(val).strip()
        return cleaned if cleaned else "N/A"
