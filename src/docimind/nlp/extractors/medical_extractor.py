import re
from typing import Dict, Any
from docimind.nlp.extractors.base_extractor import BaseFieldExtractor
from docimind.config import DATE_REGEX


class MedicalReportExtractor(BaseFieldExtractor):
    """
    Extracts key fields for Medical Report documents:
    Patient Name, Age/Gender, Doctor/Hospital Name, Report Date, Primary Diagnosis/Test.
    """

    def extract_fields(self, raw_text: str, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        patient_name = self._extract_patient(raw_text)
        age_gender = self._extract_age_gender(raw_text)
        doctor_hospital = self._extract_doctor_hospital(raw_text)
        report_date = self._extract_date(raw_text)
        diagnosis = self._extract_diagnosis(raw_text)

        return {
            "Patient Name": self.sanitize_field(patient_name),
            "Age / Gender": self.sanitize_field(age_gender),
            "Doctor / Hospital": self.sanitize_field(doctor_hospital),
            "Report Date": self.sanitize_field(report_date),
            "Diagnosis / Observations": self.sanitize_field(diagnosis)
        }

    def _extract_patient(self, text: str) -> str:
        match = re.search(r'(?:patient\s*name|patient)[:\s]*([A-Za-z\s]+)', text, re.IGNORECASE)
        if match:
            return match.group(1).split('\n')[0].strip()
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text
        return None

    def _extract_age_gender(self, text: str) -> str:
        match = re.search(r'(?:age/gender|age\s*&\s*sex|age)[:\s]*(\d{1,3}\s*(?:yrs|years)?\s*/?\s*(?:male|female|m|f)?)', text, re.IGNORECASE)
        return match.group(1) if match else None

    def _extract_doctor_hospital(self, text: str) -> str:
        match = re.search(r'(?:dr\.|doctor|referred by)[:\s]*([A-Za-z\s\.]+)', text, re.IGNORECASE)
        if match:
            return "Dr. " + match.group(1).split('\n')[0].strip()
        match_hosp = re.search(r'([A-Za-z\s]+(?:hospital|clinic|labs|laboratory))', text, re.IGNORECASE)
        return match_hosp.group(1) if match_hosp else None

    def _extract_date(self, text: str) -> str:
        dates = re.findall(DATE_REGEX, text)
        return dates[0] if dates else None

    def _extract_diagnosis(self, text: str) -> str:
        match = re.search(r'(?:diagnosis|impression|findings|test)[:\s]*([A-Za-z0-9\s,\.-]+)', text, re.IGNORECASE)
        if match:
            return match.group(1).split('\n')[0].strip()
        return None
