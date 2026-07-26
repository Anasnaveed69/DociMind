import re
from typing import Dict, Any
from docimind.nlp.extractors.base_extractor import BaseFieldExtractor
from docimind.config import DATE_REGEX, AMOUNT_REGEX


class UtilityBillExtractor(BaseFieldExtractor):
    """
    Extracts key fields for Utility Bill documents:
    Account Number, Customer Name, Billing Date, Due Date, Total Amount Due, Utility Provider.
    """

    def extract_fields(self, raw_text: str, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        account_num = self._extract_account_no(raw_text)
        customer_name = self._extract_customer(raw_text)
        billing_date = self._extract_billing_date(raw_text)
        due_date = self._extract_due_date(raw_text)
        total_due = self._extract_total_due(raw_text)
        provider = self._extract_provider(raw_text)

        return {
            "Account Number": self.sanitize_field(account_num),
            "Customer Name": self.sanitize_field(customer_name),
            "Utility Provider": self.sanitize_field(provider),
            "Billing Date": self.sanitize_field(billing_date),
            "Due Date": self.sanitize_field(due_date),
            "Total Amount Due": self.sanitize_field(total_due)
        }

    def _extract_account_no(self, text: str) -> str:
        match = re.search(r'(?:account\s*no|account\s*#|acct)[:\s]*([A-Za-z0-9\-]+)', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_customer(self, text: str) -> str:
        match = re.search(r'(?:customer|bill\s*to)[:\s]*([A-Za-z\s]+)', text, re.IGNORECASE)
        if match:
            return match.group(1).split('\n')[0].strip()
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text
        return None

    def _extract_billing_date(self, text: str) -> str:
        match = re.search(r'(?:bill\s*date|statement\s*date)[:\s]*(' + DATE_REGEX[2:-2] + r')', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_due_date(self, text: str) -> str:
        match = re.search(r'(?:due\s*date|pay\s*by)[:\s]*(' + DATE_REGEX[2:-2] + r')', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_total_due(self, text: str) -> str:
        match = re.search(r'(?:total\s*due|amount\s*due|total\s*payable)[:\s]*\$?\s*([\d,]+\.\d{2})', text, re.IGNORECASE)
        if match:
            return match.group(1)
        amounts = re.findall(AMOUNT_REGEX, text)
        return amounts[-1] if amounts else None

    def _extract_provider(self, text: str) -> str:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return lines[0] if lines else None
