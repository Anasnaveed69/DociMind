import sqlite3
import json
import os
from datetime import datetime
from typing import Dict, Any, List

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "..", "docimind_history.db")
DB_PATH = os.path.abspath(DB_PATH)


def init_db(db_path: str = DB_PATH) -> None:
    """Initializes SQLite database schema for storing prediction history."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            filename TEXT NOT NULL,
            predicted_class TEXT NOT NULL,
            confidence REAL NOT NULL,
            ocr_confidence REAL NOT NULL,
            extracted_fields_json TEXT NOT NULL,
            processing_time_ms REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def save_prediction(
    filename: str,
    results: Dict[str, Any],
    db_path: str = DB_PATH
) -> int:
    """Saves a single document prediction record into the SQLite database."""
    init_db(db_path)
    
    cls_res = results.get("document_classification", {})
    ocr_res = results.get("ocr_summary", {})
    fields = results.get("extracted_fields", {})
    
    predicted_class = cls_res.get("predicted_class", "Unknown")
    confidence = float(cls_res.get("confidence", 0.0))
    ocr_confidence = float(ocr_res.get("avg_confidence", 0.0))
    processing_time_ms = float(results.get("processing_time_ms", 0.0))
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fields_json = json.dumps(fields, ensure_ascii=False)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO prediction_history (
            timestamp, filename, predicted_class, confidence,
            ocr_confidence, extracted_fields_json, processing_time_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        timestamp, filename, predicted_class, confidence,
        ocr_confidence, fields_json, processing_time_ms
    ))
    conn.commit()
    inserted_id = cursor.lastrowid
    conn.close()
    if inserted_id is None:
        raise RuntimeError("Failed to retrieve row ID for inserted prediction record.")
    return inserted_id


def get_prediction_history(db_path: str = DB_PATH, limit: int = 100) -> List[Dict[str, Any]]:
    """Retrieves stored prediction history records from SQLite database."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, timestamp, filename, predicted_class, confidence,
               ocr_confidence, extracted_fields_json, processing_time_ms
        FROM prediction_history
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    
    history = []
    for r in rows:
        row_dict = dict(r)
        try:
            row_dict["extracted_fields"] = json.loads(row_dict["extracted_fields_json"])
        except Exception:
            row_dict["extracted_fields"] = {}
        history.append(row_dict)
    return history


def clear_prediction_history(db_path: str = DB_PATH) -> None:
    """Clears all records from prediction history table."""
    init_db(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM prediction_history")
    conn.commit()
    conn.close()
