import re
from typing import Dict, Any
from docimind.nlp.extractors.base_extractor import BaseFieldExtractor
from docimind.config import DATE_REGEX


class DriverLicenseExtractor(BaseFieldExtractor):
    """
    Extracts key fields for Driver License documents:
    License Number, Full Name, DOB, Expiry Date, License Class, Address.
    """

    def extract_fields(self, raw_text: str, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        dl_number = self._extract_dl_number(raw_text)
        full_name = self._extract_name(raw_text)
        dob = self._extract_dob(raw_text)
        expiry = self._extract_expiry(raw_text)
        dl_class = self._extract_class(raw_text)

        return {
            "License Number": self.sanitize_field(dl_number),
            "Full Name": self.sanitize_field(full_name),
            "Date of Birth": self.sanitize_field(dob),
            "Expiry Date": self.sanitize_field(expiry),
            "License Class": self.sanitize_field(dl_class)
        }

    def _extract_dl_number(self, text: str) -> str:
        patterns = [
            r'(?:dl\s*no|licence\s*no|license\s*no|dl#)[:\s]*([A-Z0-9\-]+)',
            r'\b[A-Z]\d{7,14}\b'
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1 if ":" in pattern or "no" in pattern else 0)
        return None

    def _extract_name(self, text: str) -> str:
        match = re.search(r'(?:name|fn|ln)[:\s]*([A-Za-z\s]+)', text, re.IGNORECASE)
        if match:
            return match.group(1).split('\n')[0].strip()
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text
        return None

    def _extract_dob(self, text: str) -> str:
        match = re.search(r'(?:dob|birth)[:\s]*(' + DATE_REGEX[2:-2] + r')', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_expiry(self, text: str) -> str:
        match = re.search(r'(?:exp|expires|expiry)[:\s]*(' + DATE_REGEX[2:-2] + r')', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_class(self, text: str) -> str:
        match = re.search(r'(?:class|cat)[:\s]*([A-Z0-9]+)', text, re.IGNORECASE)
        return match.group(1) if match else "C"
