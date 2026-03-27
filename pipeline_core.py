"""
Document AI System - Professional Pipeline Orchestrator
Architecture: EasyOCR → CLIP → LayoutLMv3 → Regex + Distance Scoring → Qwen2.5-0.5B
"""

import json
import time
import logging
import cv2
import numpy as np
import torch
from PIL import Image
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Standardize path for professional namespaced imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Standardized logging and config
try:
    from document_ai_system.config import DEVICE, OCR_MAX_WIDTH, OCR_MAX_HEIGHT, SHORT_CIRCUIT_THRESHOLD
    from document_ai_system.utils.logger import logger
except ImportError:
    DEVICE = "cpu"
    OCR_MAX_WIDTH, OCR_MAX_HEIGHT = 800, 1200
    SHORT_CIRCUIT_THRESHOLD = 20
    logger = logging.getLogger("DocumentAI")

from document_ai_system.pipeline.ocr_engine import extract_text
from document_ai_system.pipeline.date_extractor import find_candidate_dates, score_dates, classify_by_keywords
from document_ai_system.pipeline.vlm_reasoning import reason_expiry_date
from document_ai_system.utils.date_normalizer import normalize_to_iso
from document_ai_system.utils.image_utils import preprocess_image

def process_document(image_path: str) -> Dict[str, Any]:
    """
    Executes the optimized extraction pipeline (OCR -> Regex -> LLM Fallback).
    """
    timers = {}
    total_start = time.time()
    logger.info(f"Processing Request (Optimized): {image_path}")

    try:
        # Step 1: Load image
        s = time.time()
        img_cv = cv2.imread(image_path)
        if img_cv is None: 
            raise ValueError(f"Failed to read image at {image_path}")
        timers['load_image'] = time.time() - s

        # Step 2: Preprocess
        s = time.time()
        # Sub-3s target dimension: 700px
        img_processed = preprocess_image(img_cv, 700, 1000)
        timers['preprocess'] = time.time() - s

        # Step 3: OCR
        s = time.time()
        raw_text, word_boxes = extract_text(img_processed)
        timers['ocr'] = time.time() - s

        if not raw_text.strip():
            logger.warning("No text found in document.")
            return {"document_type": "None", "expiry_date": None, "confidence": 0.0, "processing_time": f"{time.time()-total_start:.2f}s"}

        # Step 4: High-Accuracy Heuristic Analysis (One-Go)
        s = time.time()
        doc_type = classify_by_keywords(raw_text)
        scoring = score_dates(raw_text, doc_type)
        
        heuristic_best = scoring.get("best_match_heuristic")
        temporal_best = scoring.get("temporal_best")
        h_confidence = scoring.get("confidence", 0.0)
        timers['heuristics'] = time.time() - s
        
        # Step 5: "Zero-LLM" Fast Path (Heuristic Only)
        if h_confidence >= 1.0 and temporal_best:
            logger.info(f"Zero-LLM Fast Path selected: {temporal_best}")
            final_date_str = temporal_best
            final_confidence = 0.99
            method = "fast-path-heuristic"
        else:
            # Step 6: Targeted LLM Judgment (Type + Date verification)
            s = time.time()
            logger.info(f"Fallback to AI Reasoning (Joint Type & Date Extraction).")
            v_type, v_date, v_conf = reason_expiry_date(
                document_type=doc_type,
                candidate_dates=scoring.get("candidate_dates", []),
                relevant_text_snippet=raw_text[:1500]
            )
            doc_type = v_type # Use verified type
            final_date_str = v_date
            final_confidence = v_conf
            method = "ai-judgment"
            timers['llm'] = time.time() - s

        # Step 7: Normalization
        final_date = normalize_to_iso(final_date_str)
        total_time = time.time() - total_start
        logger.info(f"Pipeline finished ({method}) in {total_time:.2f}s")

        return {
            "document_type": doc_type,
            "expiry_date": final_date if final_date != "NULL" else None,
            "confidence": float(final_confidence),
            "processing_time": f"{float(total_time):.2f}s",
            "method": method,
            "timers": timers
        }

    except Exception as e:
        logger.critical(f"Critical Pipeline Failure: {e}", exc_info=True)
        return {"error": "Pipeline failure", "message": str(e)}

    except Exception as e:
        logger.critical(f"Critical Pipeline Failure: {e}", exc_info=True)
        return {"error": "Pipeline failure", "message": str(e)}



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Document AI Pipeline Orchestrator")
    parser.add_argument("--image", required=True, help="Path to document image")
    args = parser.parse_args()
    print(json.dumps(process_document(args.image), indent=2))
