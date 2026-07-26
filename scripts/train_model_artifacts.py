import os
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import sys
sys.path.insert(0, "src")

from docimind.config import MODELS_DIR, CLASSIFIER_PATH, VECTORIZER_PATH, LABEL_ENCODER_PATH, SUPPORTED_DOCUMENTS


# Document Corpus dataset for training the 9 document categories
CORPUS = {
    "Invoice": [
        "INVOICE Bill To Total Amount Due Invoice Number Tax ID Item Description Unit Price Subtotal Balance Payable Date Payment Terms Vendor Name Account",
        "Tax Invoice Commercial Invoice Order Number Amount Due Net Total Tax Rate Vat ID Customer Reference Shipping Address Bank Transfer Instructions",
        "INVOICE # 10492 Date 2026-05-12 Due Date 2026-06-12 Total Amount $1,450.00 Subtotal $1,300.00 Tax $150.00 Vendor Supplies Inc",
        "Standard Commercial Invoice Client Billing Total Payment Due $890.50 Line Items Quantity Unit Cost Total Price Account Receivable Invoice Date",
        "Invoice Summary Billing Statement Total Charges $3,400.00 Remit Payment To Vendor Company PO Number 98412 Terms Net 30 Days"
    ],
    "Receipt": [
        "Store Receipt Cashier Transaction Total Change Due Card Number Cash Payment Date Merchant Subtotal Items Purchased Store Location Sales Tax",
        "Sales Receipt Cashier 04 Receipt Number 8412 Subtotal $24.50 Tax $2.10 Total Amount Paid $26.60 Cash $30.00 Change $3.40 Thank You Come Again",
        "Supermarket Grocery Receipt Items Total $45.80 VISA Debit Card Approved Ref 9041 Date 2026-07-20 Merchant ID Store #102",
        "Restaurant Receipt Guest Check Table 14 Server John Subtotal $88.00 Tip $15.00 Total Amount $103.00 Card Payment Merchant Copy",
        "Retail Store Receipt Cash Transaction Item 1 Item 2 Subtotal Tax Included Total Paid Merchant Copy Keep Receipt For Returns"
    ],
    "Resume": [
        "Curriculum Vitae Resume Work Experience Education Skills Employment History Personal Profile Projects Contact Email Phone Degree Bachelor Science Master",
        "Professional Resume Software Engineer Skills Python Machine Learning Deep Learning Experience Senior AI Developer Education Computer Science Contact Phone Email",
        "Resume John Doe Email john.doe@example.com Phone +1-555-0192 Technical Skills Data Science Scikit-Learn PyTorch Work History Full-Stack Developer",
        "Curriculum Vitae Academic Background Technical Proficiencies Experience Technical Lead Lead AI Engineer Publications Certifications References",
        "Executive Resume Career Summary Leadership Project Management Skills Education Bachelor of Technology Master of Business Administration Contact Information"
    ],
    "Medical Report": [
        "Medical Report Patient Name Age Gender Diagnosis Doctor Physician Clinical Notes Prescription Examination Blood Pressure Lab Results Hospital Clinic",
        "Patient Health Record Hospital Medical Center Patient Name Jane Smith Age 42 Gender Female Primary Diagnosis Hypertension Doctor Dr Robert Examination Date",
        "Clinical Laboratory Report Patient Identification Test Description Result Reference Range Units Blood Glucose Cholesterol Doctor Signature Hospital",
        "Physician Diagnostic Report Hospital Clinical Findings Treatment Plan Recommended Medication Patient History Symptoms Notes Doctor Specialist",
        "Medical Examination Report Outpatient Record Patient ID 90412 Symptoms Fever Cough Assessment Prescription Advised Rest Hospital Department"
    ],
    "Passport": [
        "Passport Republic Official Travel Document Surname Given Names Nationality Date of Birth Sex Place of Birth Date of Issue Date of Expiry Passport No MRZ P<",
        "International Passport Document Type P Country Code Surname Smith Given Names Alexander Nationality USA Date of Birth 15 AUG 1990 Authority Department of State MRZ P<USASMITH",
        "Passport Identity Page Document Number A8491204 Surname Butt Given Names Anas Date of Expiry 2030-10-12 Sex M Issuing Country MRZ Code P<PAK",
        "Republic Official Passport Personal Details Full Name Place of Issue Date of Issue Expiry Date Machine Readable Zone P<CANLAPOINTE",
        "Passport Document Surname Garcia Given Names Maria Type P Code GBR Passport Number 90412851 National Status British Citizen Sex F"
    ],
    "National ID Card": [
        "National ID Card Identity Card Identity Number CNIC Full Name Date of Birth Father Name Address Issue Date Expiry Date Citizenship Citizen",
        "National Identity Card Identity Number 35202-1940129-3 Full Name Anas Naveed Butt Gender Male Date of Birth 1998-04-12 Country Republic",
        "State Identity Card ID Number 901-491-019 Full Name Robert Johnson Address Main Street Issue Date 2020-01-15 Expiry Date 2030-01-15 Citizen Card",
        "Government Identity Card National ID # ID-8491204 Surname Davis Given Name Sarah DOB 1995-11-20 Resident Card Citizen Status Active",
        "National Registration Card Card Number 491049102 Name Michael Brown Date of Issue Date of Expiry Signature Republic Identity Office"
    ],
    "Driver License": [
        "Driver License Driving Licence License Number DL No Class Organ Donor Restrictions Endorsements Date of Birth Issue Date Expiry Date Driver Name Address",
        "State Driver License DL Number D-9041285-1 Full Name David Miller DOB 1988-06-14 Class C Restrictions None Expiry Date 2028-06-14 Department Motor Vehicles",
        "Driving Licence License No DL-840192 Class Motor Vehicle Expiry Date 2029-09-01 Driver Name James Wilson Address Oak Street Motor Vehicle Authority",
        "Driver License State Department Motor Vehicles DL ID 84910291 Driver Name Emma Watson Height Weight Eyes Class Operator Issue Date Expiry Date",
        "Commercial Driver License CDL Number 9410294-A Name Thomas Anderson Class A Endorsements Air Brakes Issue Date Expiry Date License Branch"
    ],
    "Utility Bill": [
        "Utility Bill Electricity Water Gas Power Account Number Meter Number Reading Due Date Amount Payable Total Due Energy Usage Billing Period Consumer Name",
        "Electric Utility Bill Account Number 904-192-491 Billing Period June 2026 Meter Reading 4810 kWh Due Date 2026-07-15 Total Amount Due $142.50 Power Company",
        "Water & Sewage Utility Bill Account # W-84192 Consumer Name John Taylor Billing Date Total Charges $68.40 Due Date 2026-07-25 Pay Online Utility Company",
        "Natural Gas Utility Bill Account Number 4810294 Billing Summary Monthly Usage Therms Current Charges $89.15 Total Due Payable Utility Services",
        "Monthly Power Utility Bill Account No 104-912 Customer Address Current Meter Reading Previous Reading Units Consumed Total Bill Amount Due Date"
    ],
    "Bank Statement": [
        "Bank Statement Account Statement Account Number Opening Balance Closing Balance Total Deposits Total Withdrawals Transaction Date Description Balance Credit Debit Bank",
        "Bank Account Statement Account Number 401928491 Statement Period July 1 to July 31 Opening Balance $5,420.00 Total Credits $2,100.00 Total Debits $1,850.00 Closing Balance $5,670.00",
        "Financial Bank Statement Customer Name Business Corp Account No 9041824-01 Transaction History Deposit Withdrawal Balance Account Summary Savings Checking",
        "Bank Monthly Statement Account Summary Beginning Balance Credits Debits Ending Balance Statement Date Interest Earned Account Details National Bank",
        "Commercial Bank Statement Account # 8419204-A Statement Date 2026-06-30 Opening Balance Closing Balance Daily Transactions Transfer Credit Debit Balance"
    ]
}

def train_and_export():
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    texts = []
    labels = []
    
    for category, samples in CORPUS.items():
        for text in samples:
            texts.append(text.lower())
            labels.append(category)
            
    print(f"Training on {len(texts)} corpus samples across {len(set(labels))} classes...")
    
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)
    X = vectorizer.fit_transform(texts)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(labels)
    
    classifier = LogisticRegression(C=1.0, max_iter=500, class_weight='balanced')
    classifier.fit(X, y)
    
    # Save Joblib artifacts
    joblib.dump(classifier, CLASSIFIER_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    
    print(f"Successfully exported models to:")
    print(f" - Classifier: {CLASSIFIER_PATH}")
    print(f" - Vectorizer: {VECTORIZER_PATH}")
    print(f" - Encoder: {LABEL_ENCODER_PATH}")

if __name__ == "__main__":
    train_and_export()
