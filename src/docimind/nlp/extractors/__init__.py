from typing import Dict, Type
from docimind.nlp.extractors.base_extractor import BaseFieldExtractor
from docimind.nlp.extractors.invoice_extractor import InvoiceExtractor
from docimind.nlp.extractors.receipt_extractor import ReceiptExtractor
from docimind.nlp.extractors.resume_extractor import ResumeExtractor
from docimind.nlp.extractors.medical_extractor import MedicalReportExtractor
from docimind.nlp.extractors.passport_extractor import PassportExtractor
from docimind.nlp.extractors.id_card_extractor import IDCardExtractor
from docimind.nlp.extractors.driver_license_extractor import DriverLicenseExtractor
from docimind.nlp.extractors.utility_bill_extractor import UtilityBillExtractor
from docimind.nlp.extractors.bank_statement_extractor import BankStatementExtractor

EXTRACTOR_REGISTRY: Dict[str, Type[BaseFieldExtractor]] = {
    "Invoice": InvoiceExtractor,
    "Receipt": ReceiptExtractor,
    "Resume": ResumeExtractor,
    "Medical Report": MedicalReportExtractor,
    "Passport": PassportExtractor,
    "National ID Card": IDCardExtractor,
    "Driver License": DriverLicenseExtractor,
    "Utility Bill": UtilityBillExtractor,
    "Bank Statement": BankStatementExtractor
}


def get_field_extractor(document_type: str, spacy_nlp=None) -> BaseFieldExtractor:
    """
    Factory function returning the specialized Field Extractor instance
    corresponding to the predicted document type.
    """
    extractor_cls = EXTRACTOR_REGISTRY.get(document_type, InvoiceExtractor)
    return extractor_cls(spacy_nlp=spacy_nlp)
