import re
from typing import Dict, Any
from docimind.nlp.extractors.base_extractor import BaseFieldExtractor
from docimind.config import EMAIL_REGEX, DATE_REGEX, AMOUNT_REGEX


class InvoiceExtractor(BaseFieldExtractor):
    """
    Extracts key fields for Invoice documents:
    Invoice Number, Vendor Name, Invoice Date, Due Date, Total Amount, Tax/VAT, Email.
    """

    def extract_fields(self, raw_text: str, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        invoice_num = self._extract_invoice_number(raw_text)
        invoice_date = self._extract_invoice_date(raw_text)
        due_date = self._extract_due_date(raw_text)
        total_amount = self._extract_total_amount(raw_text)
        tax_amount = self._extract_tax(raw_text)
        vendor_name = self._extract_vendor(lines)
        vendor_email = self._extract_email(raw_text)

        return {
            "Invoice Number": self.sanitize_field(invoice_num),
            "Vendor Name": self.sanitize_field(vendor_name),
            "Invoice Date": self.sanitize_field(invoice_date),
            "Due Date": self.sanitize_field(due_date),
            "Total Amount": self.sanitize_field(total_amount),
            "Tax / VAT Amount": self.sanitize_field(tax_amount),
            "Vendor Email": self.sanitize_field(vendor_email)
        }

    def _extract_invoice_number(self, text: str) -> str | None:
        patterns = [
            r'(?:invoice|transaction|txn|ref|reference|receipt)\s*(?:no|number|#|id)?[:\s]*([A-Za-z0-9\-]+)',
            r'inv\s*(?:no|#)?[:\s]*([A-Za-z0-9\-]+)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        hex_match = re.search(r'\b([0-9a-fA-F]{16,64})\b', text)
        if hex_match:
            return hex_match.group(1)
        return None

    def _extract_invoice_date(self, text: str) -> str | None:
        match = re.search(r'(?:invoice\s*date|date|txn\s*date)[:\s]*(' + DATE_REGEX[2:-2] + r')', text, re.IGNORECASE)
        if match:
            return match.group(1)
        dates = re.findall(DATE_REGEX, text, re.IGNORECASE)
        return dates[0] if dates else None

    def _extract_due_date(self, text: str) -> str | None:
        match = re.search(r'(?:due\s*date|payment\s*due)[:\s]*(' + DATE_REGEX[2:-2] + r')', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_total_amount(self, text: str) -> str | None:
        patterns = [
            r'(?:total\s*amount|total\s*due|grand\s*total|balance\s*due|amount\s*sent|total)[:\s]*(?:Rs\.?|\$)?\s*([\d,]+(?:\.\d{2})?)',
            r'(?:Rs\.?|\$)\s*([\d,]+(?:\.\d{2})?)'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1)
        amounts = re.findall(AMOUNT_REGEX, text)
        return amounts[-1] if amounts else None


    def _extract_tax(self, text: str) -> str | None:
        match = re.search(r'(?:tax|vat|gst)[:\s]*\$?\s*([\d,]+\.\d{2})', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_vendor(self, lines: list[str]) -> str | None:
        if self.nlp:
            doc = self.nlp("\n".join(lines[:5]))
            for ent in doc.ents:
                if ent.label_ == "ORG":
                    return ent.text
        return lines[0] if lines else None

    def _extract_email(self, text: str) -> str | None:
        match = re.search(EMAIL_REGEX, text)
        return match.group(0) if match else None
