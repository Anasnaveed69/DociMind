import re
from typing import Dict, Any
from docimind.nlp.extractors.base_extractor import BaseFieldExtractor
from docimind.config import EMAIL_REGEX, PHONE_REGEX


class ResumeExtractor(BaseFieldExtractor):
    """
    Extracts key fields for Resume documents:
    Candidate Name, Email, Phone Number, Primary Skills, Education Level.
    """

    SKILL_LIST = [
        "python", "java", "c++", "javascript", "react", "node.js", "sql", "aws",
        "docker", "kubernetes", "machine learning", "deep learning", "nlp", "computer vision",
        "pytorch", "tensorflow", "scikit-learn", "pandas", "numpy", "opencv", "easyocr",
        "git", "rest api", "fastapi", "flask", "django", "html", "css"
    ]

    def extract_fields(self, raw_text: str, ocr_data: Dict[str, Any]) -> Dict[str, Any]:
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]

        candidate_name = self._extract_candidate_name(lines)
        email = self._extract_email(raw_text)
        phone = self._extract_phone(raw_text)
        skills = self._extract_skills(raw_text)
        education = self._extract_education(raw_text)

        return {
            "Candidate Name": self.sanitize_field(candidate_name),
            "Email Address": self.sanitize_field(email),
            "Phone Number": self.sanitize_field(phone),
            "Extracted Skills": ", ".join(skills) if skills else "N/A",
            "Education": self.sanitize_field(education)
        }

    def _extract_candidate_name(self, lines: list) -> str:
        if self.nlp:
            doc = self.nlp("\n".join(lines[:5]))
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    return ent.text
        return lines[0] if lines else None

    def _extract_email(self, text: str) -> str:
        match = re.search(EMAIL_REGEX, text)
        return match.group(0) if match else None

    def _extract_phone(self, text: str) -> str:
        match = re.search(PHONE_REGEX, text)
        return match.group(0) if match else None

    def _extract_skills(self, text: str) -> list:
        lower = text.lower()
        found = []
        for skill in self.SKILL_LIST:
            if re.search(r'\b' + re.escape(skill) + r'\b', lower):
                found.append(skill.title())
        return found[:10]

    def _extract_education(self, text: str) -> str:
        degrees = ["Ph.D", "Master", "Bachelor", "B.S.", "M.S.", "B.Tech", "M.Tech", "B.E.", "B.A.", "High School"]
        for d in degrees:
            if re.search(r'\b' + re.escape(d) + r'\b', text, re.IGNORECASE):
                return d
        return None
