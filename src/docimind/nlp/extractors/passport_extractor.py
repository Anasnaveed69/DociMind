import re
from typing import Dict, Any
from docimind.nlp.extractors.base_extractor import BaseFieldExtractor
from docimind.config import PASSPORT_NUM_REGEX, DATE_REGEX


class PassportExtractor(BaseFieldExtractor):
    """
    Extracts key fields for Passport documents:
    Passport Number, Surname, Given Names, Nationality, Date of Birth, Expiry Date, MRZ string.
    """

    def extract_fields(self, raw_text: str, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        passport_no = self._extract_passport_no(raw_text)
        surname = self._extract_surname(raw_text)
        given_names = self._extract_given_names(raw_text)
        nationality = self._extract_nationality(raw_text)
        dob = self._extract_dob(raw_text)
        expiry = self._extract_expiry(raw_text)
        mrz = self._extract_mrz(raw_text)

        return {
            "Passport Number": self.sanitize_field(passport_no),
            "Surname": self.sanitize_field(surname),
            "Given Names": self.sanitize_field(given_names),
            "Nationality": self.sanitize_field(nationality),
            "Date of Birth": self.sanitize_field(dob),
            "Expiry Date": self.sanitize_field(expiry),
            "MRZ Line": self.sanitize_field(mrz)
        }

    def _extract_passport_no(self, text: str) -> str:
        match = re.search(PASSPORT_NUM_REGEX, text)
        if match:
            return match.group(0)
        match_label = re.search(r'(?:passport\s*no|passport\s*number)[:\s]*([A-Z0-9]{8,9})', text, re.IGNORECASE)
        return match_label.group(1) if match_label else None

    def _extract_surname(self, text: str) -> str:
        match = re.search(r'(?:surname|last\s*name)[:\s]*([A-Za-z]+)', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_given_names(self, text: str) -> str:
        match = re.search(r'(?:given\s*names?|first\s*name)[:\s]*([A-Za-z\s]+)', text, re.IGNORECASE)
        return match.group(1).split('\n')[0].strip() if match else None

    def _extract_nationality(self, text: str) -> str:
        match = re.search(r'(?:nationality|code)[:\s]*([A-Za-z]{3,15})', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_dob(self, text: str) -> str:
        match = re.search(r'(?:date\s*of\s*birth|dob)[:\s]*(' + DATE_REGEX[2:-2] + r')', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_expiry(self, text: str) -> str:
        match = re.search(r'(?:date\s*of\s*expiry|expiry\s*date)[:\s]*(' + DATE_REGEX[2:-2] + r')', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_mrz(self, text: str) -> str:
        match = re.search(r'P<[A-Z0-9<]{40,}', text)
        return match.group(0) if match else None
