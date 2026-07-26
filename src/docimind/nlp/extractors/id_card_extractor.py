import re
from typing import Dict, Any
from docimind.nlp.extractors.base_extractor import BaseFieldExtractor
from docimind.config import DATE_REGEX


class IDCardExtractor(BaseFieldExtractor):
    """
    Extracts key fields for National ID Card documents:
    ID Number, Full Name, Date of Birth, Gender, Address, Issue/Expiry Date.
    """

    def extract_fields(self, raw_text: str, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        id_number = self._extract_id_number(raw_text)
        full_name = self._extract_name(raw_text)
        dob = self._extract_dob(raw_text)
        gender = self._extract_gender(raw_text)
        address = self._extract_address(raw_text)

        return {
            "ID Card Number": self.sanitize_field(id_number),
            "Full Name": self.sanitize_field(full_name),
            "Date of Birth": self.sanitize_field(dob),
            "Gender": self.sanitize_field(gender),
            "Address": self.sanitize_field(address)
        }

    def _extract_id_number(self, text: str) -> str:
        patterns = [
            r'(?:id\s*no|id\s*number|identity\s*no|cnic)[:\s]*([A-Z0-9\-]+)',
            r'\b\d{5}-\d{7}-\d{1}\b',
            r'\b[A-Z0-9]{9,12}\b'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1 if "(" in pattern or ":" in pattern else 0)
        return None

    def _extract_name(self, text: str) -> str:
        match = re.search(r'(?:name|full\s*name)[:\s]*([A-Za-z\s]+)', text, re.IGNORECASE)
        if match:
            return match.group(1).split('\n')[0].strip()
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text
        return None

    def _extract_dob(self, text: str) -> str:
        match = re.search(r'(?:dob|date\s*of\s*birth|birth\s*date)[:\s]*(' + DATE_REGEX[2:-2] + r')', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_gender(self, text: str) -> str:
        match = re.search(r'(?:sex|gender)[:\s]*(male|female|m|f)', text, re.IGNORECASE)
        if match:
            val = match.group(1).upper()
            return "Male" if val in ["M", "MALE"] else "Female"
        return None

    def _extract_address(self, text: str) -> str:
        match = re.search(r'(?:address|residence)[:\s]*([A-Za-z0-9\s,\.-]+)', text, re.IGNORECASE)
        return match.group(1).split('\n')[0].strip() if match else None
