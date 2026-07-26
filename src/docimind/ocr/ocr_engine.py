import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from docimind.config import DEFAULT_EASYOCR_LANGUAGES

try:
    import easyocr
    HAS_EASYOCR = True
except ImportError:
    HAS_EASYOCR = False



class OCREngine:
    """
    Pretrained EasyOCR Wrapper Engine for Document Text & Bounding Box Extraction.
    Does NOT train or mutate OCR weights; uses pretrained EasyOCR models.
    """

    def __init__(self, languages: Optional[List[str]] = None, gpu: bool = False):
        self.languages = languages or DEFAULT_EASYOCR_LANGUAGES
        self.gpu = gpu
        self.reader = None

    def _initialize_reader(self):
        """Lazy initialization of EasyOCR reader to save startup memory."""
        if HAS_EASYOCR and self.reader is None:
            try:
                self.reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
            except Exception as e:
                print(f"Warning: Failed to initialize EasyOCR reader: {e}")
                self.reader = None

    def extract(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Extracts OCR text, bounding boxes, and confidence scores.
        Falls back gracefully if easyocr package is not installed or reader fails to initialize.
        """
        if not HAS_EASYOCR:
            print("Warning: EasyOCR package not installed. Returning empty OCR detection.")
            return {
                "full_text": "",
                "lines": [],
                "detections": [],
                "avg_confidence": 0.0
            }

        self._initialize_reader()
        if self.reader is None:
            print("Warning: EasyOCR reader is uninitialized. Returning empty OCR detection.")
            return {
                "full_text": "",
                "lines": [],
                "detections": [],
                "avg_confidence": 0.0
            }

        results = self.reader.readtext(image)

        detections = []
        lines = []
        confidences = []

        for bbox, text, prob in results:
            clean_text = str(text).strip()
            if not clean_text:
                continue

            formatted_bbox = [[int(pt[0]), int(pt[1])] for pt in bbox]
            confidence = float(prob)

            detections.append({
                "text": clean_text,
                "bbox": formatted_bbox,
                "confidence": confidence
            })
            lines.append(clean_text)
            confidences.append(confidence)

        full_text = "\n".join(lines)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return {
            "full_text": full_text,
            "lines": lines,
            "detections": detections,
            "avg_confidence": round(avg_confidence, 4)
        }
