import streamlit as st
import cv2
import numpy as np
from PIL import Image
import json
import sys
import os

# Add src to python path for module loading
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "src")))



from docimind.pipeline import DociMindPipeline
from docimind.config import SUPPORTED_DOCUMENTS
from docimind.utils.export_utils import export_to_json, export_to_csv


# Page Configuration
st.set_page_config(
    page_title="DociMind - Intelligent Document Classification & Extraction Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject Custom Modern Glassmorphic UI CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark Mode Glassmorphism Container */
    .main {
        background: linear-gradient(135deg, #0b0f19 0%, #1e1b4b 50%, #0f172a 100%);
        color: #f8fafc;
    }

    .glass-card {
        background: rgba(30, 41, 59, 0.7);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255, 255, 255, 0.125);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }

    .badge-primary {
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: #ffffff;
        font-weight: 700;
        padding: 6px 16px;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.95rem;
    }

    .badge-success {
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid #10b981;
        color: #34d399;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 48px;
        background-color: rgba(30, 41, 59, 0.5);
        border-radius: 10px 10px 0 0;
        color: #94a3b8;
        font-weight: 600;
    }

    .stTabs [aria-selected="true"] {
        background-color: #6366f1 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    """Singleton pipeline instance cached across Streamlit reruns."""
    return DociMindPipeline(use_gpu=False)


def main():
    # Sidebar Navigation & Settings
    with st.sidebar:
        st.image("https://img.icons8.com/isometric/100/document.png", width=70)
        st.title("DociMind Engine")
        st.caption("Intelligent Document Classification & Extraction Platform")
        st.markdown("---")

        st.subheader("⚙️ Preprocessing Controls")
        apply_deskew = st.checkbox("Auto Deskew & Orientation", value=True, help="Correct image orientation and skew")
        apply_denoise = st.checkbox("Fast N-Means Denoise", value=True, help="Remove scanned document noise")

        st.markdown("---")
        st.subheader("📋 Document Types Covered")
        for doc in SUPPORTED_DOCUMENTS:
            st.markdown(f"• `{doc}`")

        st.markdown("---")
        st.caption("Author: **Anas Naveed Butt**")
        st.caption("Live App: [docimind.streamlit.app](https://docimind.streamlit.app/)")
        st.caption("Project Deliverables: [Google Drive Zip](https://drive.google.com/drive/folders/1rDw9sQYvrQKqLGqtV5Se5RRu5LgWFRKK?usp=drive_link)")
        st.caption("Assigned Topic: **OCR Information Extraction**")
        st.caption("Offer ID: **CAX-OL-2026-283**")

    # Header Banner Card
    st.markdown("""
        <div class="glass-card">
            <h1 style="margin: 0; color: #f8fafc; font-size: 2.2rem; font-weight: 700;">
                🧠 DociMind — Intelligent Document Classification & Information Extraction Platform
            </h1>
            <p style="color: #94a3b8; margin-top: 8px; font-size: 1.05rem;">
                End-to-end multi-domain platform combining <b>Computer Vision</b> (OpenCV), <b>OCR</b> (EasyOCR), <b>Machine Learning</b> classification (Scikit-Learn), and <b>NLP</b> key-field extraction (spaCy/Regex).
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Document Uploader (Supports both Images and PDFs)
    uploaded_file = st.file_uploader(
        "Upload Document Image or PDF",
        type=["png", "jpg", "jpeg", "webp", "pdf"],
        help="Upload a scanned document image or PDF for intelligent extraction."
    )

    if uploaded_file is not None:
        try:
            file_bytes = uploaded_file.getvalue()
            if not file_bytes:
                st.error("Uploaded file is empty (0 bytes). Please select a valid document.")
                return

            pipeline = get_pipeline()


            # Execute Extraction Pipeline with Loading Spinner
            with st.spinner("⚡ Running DociMind Pipeline (OpenCV -> EasyOCR -> Classifier -> Field Extractor)..."):
                results = pipeline.process(
                    image_input=file_bytes,
                    apply_denoise=apply_denoise,
                    apply_deskew=apply_deskew
                )

            cls_res = results["document_classification"]
            predicted_class = cls_res["predicted_class"]
            confidence = cls_res["confidence"]
            ocr_summary = results["ocr_summary"]
            ocr_conf = ocr_summary["avg_confidence"]
            latency = results["processing_time_ms"]

            # Key Metric Summary Display Cards
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Detected Document Type", predicted_class)
            with m2:
                st.metric("Classifier Confidence", f"{int(confidence * 100)}%")
            with m3:
                st.metric("OCR Avg Confidence", f"{int(ocr_conf * 100)}%")
            if ocr_conf == 0.0 or not ocr_summary["full_text"].strip():
                st.warning("⚠️ **No readable text detected by OCR.** If you uploaded a PDF or image, ensure it is clear, properly oriented, and contains selectable/rasterized text. Try disabling *Auto Deskew & Orientation* in the sidebar if the document is already upright.")

            st.markdown("---")


            # Main Application Multi-Tab Interface
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "📷 OCR Preview & Visual Overlay",
                "📊 Document Prediction Metrics",
                "🔑 Extracted Key Fields",
                "📝 Raw OCR Text Output",
                "💾 Export Results"
            ])

            with tab1:
                st.subheader("Document Visual Inspection & OCR Bounding Boxes")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Processed Document Base Image**")
                    color_rgb = cv2.cvtColor(results["color_bgr"], cv2.COLOR_BGR2RGB)
                    st.image(color_rgb, use_container_width=True)
                with c2:
                    st.markdown("**EasyOCR Bounding Box Annotations (Green: High Conf, Red: Low Conf)**")
                    annotated_rgb = cv2.cvtColor(results["annotated_image_bgr"], cv2.COLOR_BGR2RGB)
                    st.image(annotated_rgb, use_container_width=True)

            with tab2:
                st.subheader("Machine Learning Document Classification")
                st.markdown(f"**Predicted Category:** <span class='badge-primary'>{predicted_class}</span>", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                st.write("**Class Probability Distribution:**")
                probs = cls_res["class_probabilities"]
                for doc_name, prob in probs.items():
                    st.progress(float(prob), text=f"{doc_name}: {int(prob * 100)}%")

            with tab3:
                st.subheader(f"Extracted Structured Entities ({predicted_class})")
                fields = results["extracted_fields"]

                if fields:
                    field_data = [{"Field Name": k, "Extracted Value": str(v)} for k, v in fields.items()]
                    st.table(field_data)
                else:
                    st.info("No specific key fields detected for this document.")

            with tab4:
                st.subheader("Raw OCR Extracted Text & Line Detections")
                st.text_area("Concatenated Full OCR Text", value=ocr_summary["full_text"], height=250)

                st.markdown("**Individual Line Detections:**")
                st.json(ocr_summary["lines"])

            with tab5:
                st.subheader("Download Extraction Results")
                json_str = export_to_json(results)
                csv_str = export_to_csv(results)

                exp1, exp2 = st.columns(2)
                with exp1:
                    st.download_button(
                        label="📥 Download JSON Payload",
                        data=json_str,
                        file_name=f"{predicted_class.lower().replace(' ', '_')}_extracted.json",
                        mime="application/json",
                        use_container_width=True
                    )
                with exp2:
                    st.download_button(
                        label="📥 Download CSV Payload",
                        data=csv_str,
                        file_name=f"{predicted_class.lower().replace(' ', '_')}_extracted.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

        except Exception as e:
            st.error(f"❌ An error occurred while processing document: {str(e)}")
            st.warning("Please ensure the uploaded file is a valid image or PDF document.")

    else:
        # Initial Welcome Cards
        st.info("👆 Upload a document image (PNG, JPG, WEBP) or PDF from the file uploader above to process.")

        st.markdown("### Supported Document Types:")
        cols = st.columns(3)
        for idx, doc in enumerate(SUPPORTED_DOCUMENTS):
            with cols[idx % 3]:
                st.markdown(f"""
                    <div class="glass-card" style="text-align: center;">
                        <h4 style="margin: 0; color: #a855f7;">{doc}</h4>
                        <p style="font-size: 0.85rem; color: #94a3b8; margin-top: 4px;">Automated OCR & NLP Field Extraction</p>
                    </div>
                """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
