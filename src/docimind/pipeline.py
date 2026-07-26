import time
import numpy as np
from typing import Union, Dict, Any
from PIL import Image

from docimind.preprocessor.image_preprocessor import ImagePreprocessor
from docimind.ocr.ocr_engine import OCREngine
from docimind.ml.classifier import DocumentClassifier
from docimind.nlp.spacy_loader import load_spacy_model
from docimind.nlp.extractors import get_field_extractor
from docimind.utils.visualizer import Visualizer


class DociMindPipeline:
    """
    End-to-End Document Information Extraction & Classification Pipeline.
    Orchestrates Computer Vision preprocessing, EasyOCR, ML Document Classification,
    and spaCy NLP / Regex Field Extraction.
    """

    def __init__(self, use_gpu: bool = False):
        self.preprocessor = ImagePreprocessor()
        self.ocr_engine = OCREngine(gpu=use_gpu)
        self.classifier = DocumentClassifier()
        self.spacy_nlp = load_spacy_model()

    def process(
        self,
        image_input: Union[str, bytes, Image.Image, np.ndarray],
        apply_denoise: bool = True,
        apply_deskew: bool = True
    ) -> Dict[str, Any]:
        """
        Executes complete document processing pipeline on input document.

        Returns:
            Dict[str, Any] containing full pipeline results:
                - document_classification: predicted_class, confidence, class_probabilities
                - extracted_fields: structured dict of key-value entities
                - ocr_summary: full_text, avg_confidence, word_count
                - annotated_image_bgr: OpenCV numpy array with drawn bounding boxes
                - processing_time_ms: float latency in milliseconds
        """
        start_time = time.time()

        # Step 1: Preprocess Image
        color_bgr, processed_gray = self.preprocessor.preprocess_pipeline(
            image_input=image_input,
            apply_denoise=apply_denoise,
            apply_deskew=apply_deskew
        )

        # Step 2: Extract Text & Bounding Boxes using EasyOCR
        ocr_results = self.ocr_engine.extract(processed_gray)
        raw_ocr_text = ocr_results["full_text"]

        # Step 3: Document Classification via Scikit-Learn Joblib Model
        classification_res = self.classifier.predict(raw_ocr_text)
        predicted_class = classification_res["predicted_class"]

        # Step 4: Structured Field Extraction using Document-Specific Extractor
        extractor = get_field_extractor(predicted_class, spacy_nlp=self.spacy_nlp)
        extracted_fields = extractor.extract_fields(raw_ocr_text, ocr_results)

        # Step 5: Visual Overlay Bounding Boxes
        annotated_bgr = Visualizer.draw_bounding_boxes(color_bgr, ocr_results["detections"])

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "document_classification": classification_res,
            "extracted_fields": extracted_fields,
            "ocr_summary": {
                "full_text": raw_ocr_text,
                "lines": ocr_results["lines"],
                "avg_confidence": ocr_results["avg_confidence"],
                "word_count": len(raw_ocr_text.split()),
                "detections_count": len(ocr_results["detections"])
            },
            "annotated_image_bgr": annotated_bgr,
            "color_bgr": color_bgr,
            "processing_time_ms": elapsed_ms
        }
