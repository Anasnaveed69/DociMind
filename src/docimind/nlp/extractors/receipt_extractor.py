import re
from typing import Dict, Any
from docimind.nlp.extractors.base_extractor import BaseFieldExtractor
from docimind.config import DATE_REGEX, AMOUNT_REGEX


class ReceiptExtractor(BaseFieldExtractor):
    """
    Extracts key fields for Receipt documents:
    Store Name, Transaction Date, Transaction Time, Total Amount, Cashier ID, Payment Method.
    """

    def extract_fields(self, raw_text: str, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        store_name = lines[0] if lines else "N/A"
        txn_date = self._extract_date(raw_text)
        txn_time = self._extract_time(raw_text)
        total_amount = self._extract_total(raw_text)
        payment_method = self._extract_payment_method(raw_text)

        return {
            "Store / Merchant Name": self.sanitize_field(store_name),
            "Transaction Date": self.sanitize_field(txn_date),
            "Transaction Time": self.sanitize_field(txn_time),
            "Total Amount Paid": self.sanitize_field(total_amount),
            "Payment Method": self.sanitize_field(payment_method)
        }

    def _extract_date(self, text: str) -> str:
        dates = re.findall(DATE_REGEX, text)
        return dates[0] if dates else None

    def _extract_time(self, text: str) -> str:
        match = re.search(r'\b(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s*[AP]M)?\b', text, re.IGNORECASE)
        return match.group(0) if match else None

    def _extract_total(self, text: str) -> str:
        match = re.search(r'(?:total|cash|balance due|amount paid)[:\s]*\$?\s*([\d,]+\.\d{2})', text, re.IGNORECASE)
        if match:
            return match.group(1)
        amounts = re.findall(AMOUNT_REGEX, text)
        return amounts[-1] if amounts else None

    def _extract_payment_method(self, text: str) -> str:
        lower = text.lower()
        if "visa" in lower:
            return "VISA Card"
        elif "mastercard" in lower or "mc" in lower:
            return "MasterCard"
        elif "amex" in lower or "american express" in lower:
            return "American Express"
        elif "cash" in lower:
            return "Cash"
        elif "debit" in lower:
            return "Debit Card"
        return "Card / Cash"
