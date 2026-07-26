import re
from typing import Dict, Any
from docimind.nlp.extractors.base_extractor import BaseFieldExtractor
from docimind.config import DATE_REGEX, AMOUNT_REGEX


class BankStatementExtractor(BaseFieldExtractor):
    """
    Extracts key fields for Bank Statement documents:
    Account Holder Name, Account Number, Bank Name, Statement Period, Opening Balance, Closing Balance.
    """

    def extract_fields(self, raw_text: str, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        account_holder = self._extract_holder(raw_text)
        account_no = self._extract_account_no(raw_text)
        bank_name = self._extract_bank_name(raw_text)
        statement_period = self._extract_period(raw_text)
        opening_bal = self._extract_opening_bal(raw_text)
        closing_bal = self._extract_closing_bal(raw_text)

        return {
            "Account Holder": self.sanitize_field(account_holder),
            "Account Number": self.sanitize_field(account_no),
            "Bank Name": self.sanitize_field(bank_name),
            "Statement Period": self.sanitize_field(statement_period),
            "Opening Balance": self.sanitize_field(opening_bal),
            "Closing Balance": self.sanitize_field(closing_bal)
        }

    def _extract_holder(self, text: str) -> str:
        match = re.search(r'(?:account\s*holder|name)[:\s]*([A-Za-z\s]+)', text, re.IGNORECASE)
        if match:
            return match.group(1).split('\n')[0].strip()
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ["PERSON", "ORG"]:
                    return ent.text
        return None

    def _extract_account_no(self, text: str) -> str:
        match = re.search(r'(?:account\s*no|account\s*number|acct\s*#)[:\s]*([X0-9\-]+)', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_bank_name(self, text: str) -> str:
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        return lines[0] if lines else None

    def _extract_period(self, text: str) -> str:
        match = re.search(r'(?:statement\s*period|period)[:\s]*([A-Za-z0-9\s,\.-/-]+)', text, re.IGNORECASE)
        return match.group(1).split('\n')[0].strip() if match else None

    def _extract_opening_bal(self, text: str) -> str:
        match = re.search(r'(?:opening\s*balance|beginning\s*balance)[:\s]*\$?\s*([\d,]+\.\d{2})', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_closing_bal(self, text: str) -> str:
        match = re.search(r'(?:closing\s*balance|ending\s*balance|new\s*balance)[:\s]*\$?\s*([\d,]+\.\d{2})', text, re.IGNORECASE)
        if match:
            return match.group(1)
        amounts = re.findall(AMOUNT_REGEX, text)
        return amounts[-1] if amounts else None
