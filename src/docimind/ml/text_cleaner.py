import re
import string


class TextCleaner:
    """
    Text Normalization & Cleaning module for raw OCR text.
    """

    @staticmethod
    def clean_text(text: str, preserve_case: bool = False) -> str:
        """
        Cleans raw OCR output by removing garbage characters while retaining
        document structure.
        """
        if not text:
            return ""

        # Replace non-breaking spaces and non-printable characters
        text = text.replace('\xa0', ' ').replace('\r', '\n')
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        # Standardize multiple spaces into single space
        text = re.sub(r'[ \t]+', ' ', text)

        # Standardize multiple newlines
        text = re.sub(r'\n+', '\n', text)

        if not preserve_case:
            text = text.lower()

        return text.strip()

    @staticmethod
    def clean_for_classification(text: str) -> str:
        """
        Aggressive cleaning for vectorization (TF-IDF bag of words).
        Strips isolated punctuation, numbers, and extra spaces.
        """
        clean = TextCleaner.clean_text(text, preserve_case=False)
        # Replace numbers with a token
        clean = re.sub(r'\b\d+\b', 'NUM', clean)
        # Remove standalone punctuation
        clean = re.sub(r'\s+[{}]\s+'.format(re.escape(string.punctuation)), ' ', clean)
        # Normalize spaces
        return re.sub(r'\s+', ' ', clean).strip()
