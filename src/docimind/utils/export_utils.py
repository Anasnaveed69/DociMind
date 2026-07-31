import json
import io
import pandas as pd
import numpy as np
from typing import Dict, Any

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


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


def export_to_pdf(data: Dict[str, Any]) -> bytes:
    """
    Generates a professional PDF report bytes for document prediction & extraction results.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=20,
        leading=24,
        textColor=colors.HexColor("#1e1b4b"),
        spaceAfter=10
    )
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=20
    )
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=13,
        leading=16,
        textColor=colors.HexColor("#4338ca"),
        spaceBefore=12,
        spaceAfter=8
    )
    normal_style = styles['Normal']
    
    elements = []
    
    # Title & Subtitle
    elements.append(Paragraph("DociMind — Document Prediction & Extraction Report", title_style))
    elements.append(Paragraph("Generated automatically by DociMind Intelligent OCR & Classification Engine", subtitle_style))
    elements.append(Spacer(1, 10))
    
    # Section 1: Classification Summary Table
    cls_res = data.get("document_classification", {})
    ocr_res = data.get("ocr_summary", {})
    
    predicted_class = cls_res.get("predicted_class", "Unknown")
    confidence = f"{float(cls_res.get('confidence', 0.0)) * 100:.1f}%"
    ocr_conf = f"{float(ocr_res.get('avg_confidence', 0.0)) * 100:.1f}%"
    word_count = str(ocr_res.get("word_count", 0))
    proc_time = f"{float(data.get('processing_time_ms', 0.0)):.1f} ms"
    
    summary_table_data = [
        [Paragraph("<b>Metric Name</b>", normal_style), Paragraph("<b>Value</b>", normal_style)],
        ["Predicted Document Category", predicted_class],
        ["Classifier Confidence", confidence],
        ["OCR Average Confidence", ocr_conf],
        ["Extracted Word Count", word_count],
        ["Pipeline Processing Latency", proc_time]
    ]
    
    summary_table = Table(summary_table_data, colWidths=[220, 300])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#f1f5f9")),
        ('TEXTCOLOR', (0, 0), (1, 0), colors.HexColor("#0f172a")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    
    elements.append(Paragraph("1. Classification & OCR Summary", heading_style))
    elements.append(summary_table)
    elements.append(Spacer(1, 15))
    
    # Section 2: Extracted Fields Table
    elements.append(Paragraph(f"2. Extracted Structured Entities ({predicted_class})", heading_style))
    extracted_fields = data.get("extracted_fields", {})
    
    if extracted_fields:
        fields_table_data = [[Paragraph("<b>Field Name</b>", normal_style), Paragraph("<b>Extracted Value</b>", normal_style)]]
        for k, v in extracted_fields.items():
            fields_table_data.append([str(k), str(v)])
        
        fields_table = Table(fields_table_data, colWidths=[200, 320])
        fields_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#eef2ff")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#c7d2fe")),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(fields_table)
    else:
        elements.append(Paragraph("<i>No domain-specific structured fields were extracted for this document.</i>", normal_style))
    
    elements.append(Spacer(1, 15))
    
    # Section 3: OCR Text Summary
    elements.append(Paragraph("3. Raw OCR Text Snippet", heading_style))
    raw_text = ocr_res.get("full_text", "").strip()
    if len(raw_text) > 600:
        raw_text = raw_text[:600] + "... [truncated]"
    if not raw_text:
        raw_text = "No text content extracted."
        
    elements.append(Paragraph(f"<font color='#334155'>{raw_text}</font>", normal_style))
    
    doc.build(elements)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
