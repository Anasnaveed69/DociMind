import unittest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from docimind.config import SUPPORTED_DOCUMENTS, DOCUMENT_LABEL_MAP
from docimind.ml.text_cleaner import TextCleaner
from docimind.nlp.extractors import get_field_extractor
from docimind.utils.export_utils import export_to_json, export_to_csv, export_to_pdf
from docimind.utils.db_utils import init_db, save_prediction, get_prediction_history, clear_prediction_history, DB_PATH


class TestDociMindCore(unittest.TestCase):

    def test_supported_documents_count(self):
        """Verify all 9 target document types are listed in configuration."""
        self.assertEqual(len(SUPPORTED_DOCUMENTS), 9)
        self.assertIn("Invoice", SUPPORTED_DOCUMENTS)
        self.assertIn("Receipt", SUPPORTED_DOCUMENTS)
        self.assertIn("Resume", SUPPORTED_DOCUMENTS)
        self.assertIn("Medical Report", SUPPORTED_DOCUMENTS)
        self.assertIn("Passport", SUPPORTED_DOCUMENTS)
        self.assertIn("National ID Card", SUPPORTED_DOCUMENTS)
        self.assertIn("Driver License", SUPPORTED_DOCUMENTS)
        self.assertIn("Utility Bill", SUPPORTED_DOCUMENTS)
        self.assertIn("Bank Statement", SUPPORTED_DOCUMENTS)

    def test_text_cleaner(self):
        """Verify text normalization and cleaning routines."""
        raw = "INVOICE   #12345 \n\n Total: $500.00 \xa0"
        cleaned = TextCleaner.clean_text(raw, preserve_case=True)
        self.assertEqual(cleaned, "INVOICE #12345 \n Total: $500.00")

        clf_clean = TextCleaner.clean_for_classification(raw)
        self.assertIn("invoice", clf_clean)
        self.assertIn("total", clf_clean)

    def test_field_extractors(self):
        """Verify field extractors instantiation and structure for all 9 classes."""
        for doc_type in SUPPORTED_DOCUMENTS:
            extractor = get_field_extractor(doc_type)
            dummy_text = f"Sample text for {doc_type} document."
            fields = extractor.extract_fields(dummy_text, {})
            self.assertIsInstance(fields, dict)
            self.assertGreater(len(fields), 0)

    def test_export_utilities(self):
        """Verify JSON, CSV, and PDF exporters produce valid outputs."""
        sample_data = {
            "document_classification": {"predicted_class": "Invoice", "confidence": 0.95},
            "ocr_summary": {"avg_confidence": 0.92, "word_count": 45, "full_text": "INVOICE #99402 Total $500"},
            "extracted_fields": {"Invoice Number": "INV-999", "Total Amount": "99.99"},
            "processing_time_ms": 120.5
        }

        json_out = export_to_json(sample_data)
        self.assertIn('"predicted_class": "Invoice"', json_out)

        csv_out = export_to_csv(sample_data)
        self.assertIn("Invoice", csv_out)
        self.assertIn("INV-999", csv_out)

        pdf_bytes = export_to_pdf(sample_data)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 100)

    def test_database_operations(self):
        """Verify SQLite prediction history storage operations."""
        test_db = os.path.join(os.path.dirname(__file__), "test_docimind.db")
        if os.path.exists(test_db):
            os.remove(test_db)

        init_db(test_db)
        sample_data = {
            "document_classification": {"predicted_class": "Receipt", "confidence": 0.98},
            "ocr_summary": {"avg_confidence": 0.95, "word_count": 25},
            "extracted_fields": {"Store Name": "Supermarket", "Total Amount": "45.00"},
            "processing_time_ms": 85.0
        }

        rec_id = save_prediction("test_receipt.png", sample_data, db_path=test_db)
        self.assertGreater(rec_id, 0)

        history = get_prediction_history(db_path=test_db)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["predicted_class"], "Receipt")
        self.assertEqual(history[0]["filename"], "test_receipt.png")

        clear_prediction_history(db_path=test_db)
        history_cleared = get_prediction_history(db_path=test_db)
        self.assertEqual(len(history_cleared), 0)

        if os.path.exists(test_db):
            os.remove(test_db)


if __name__ == "__main__":
    unittest.main()
