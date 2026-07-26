import json
import pandas as pd
import numpy as np
from typing import Dict, Any


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types gracefully."""
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.int64, np.int32, np.int16, np.int8, np.integer)):
            return int(o)
        if isinstance(o, (np.float64, np.float32, np.float16, np.floating)):
            return float(o)
        return super().default(o)


def export_to_json(data: Dict[str, Any], indent: int = 4) -> str:
    """Converts structured result dictionary to formatted JSON string."""
    # Exclude heavy raw OpenCV numpy images from the JSON export
    clean_data = {
        k: v for k, v in data.items()
        if k not in ("annotated_image_bgr", "color_bgr") and not isinstance(v, np.ndarray)
    }
    return json.dumps(clean_data, indent=indent, ensure_ascii=False, cls=NumpyEncoder)


def export_to_csv(data: Dict[str, Any]) -> str:
    """
    Flattens structured pipeline result dictionary into a single-row CSV string representation.
    """
    flat_record = {
        "Document Class": data.get("document_classification", {}).get("predicted_class", "N/A"),
        "Classifier Confidence": data.get("document_classification", {}).get("confidence", 0.0),
        "OCR Avg Confidence": data.get("ocr_summary", {}).get("avg_confidence", 0.0),
        "Total Word Count": data.get("ocr_summary", {}).get("word_count", 0)
    }

    # Merge extracted key fields
    extracted_fields = data.get("extracted_fields", {})
    for key, val in extracted_fields.items():
        flat_record[f"Field_{key}"] = val

    df = pd.DataFrame([flat_record])
    return df.to_csv(index=False)

