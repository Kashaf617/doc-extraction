import easyocr
import numpy as np
import logging
from typing import Tuple, List, Dict, Any

# Standardized logging
logger = logging.getLogger("DocumentAI.OCR")

# Full Absolute Namespace Import for professional resolution
try:
    from document_ai_system.config import DEVICE
except ImportError:
    DEVICE = "cpu"

_reader = None

def get_reader() -> easyocr.Reader:
    """
    Initializes and returns the EasyOCR Reader instance (Singleton).
    """
    global _reader
    if _reader is None:
        logger.info("Initializing EasyOCR Engine...")
        try:
            # gpu=True will fallback to CPU automatically if CUDA is missing
            _reader = easyocr.Reader(['en'], gpu="cuda" in str(DEVICE).lower())
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise RuntimeError("OCR Initialization Failure") from e
    return _reader

def extract_text(image_np: np.ndarray) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Extracts text from the image using EasyOCR.
    
    Args:
        image_np: Preprocessed image as a numpy array.
        
    Returns:
        A tuple containing the full extracted text and a list of word bounding boxes.
    """
    try:
        reader = get_reader()
        results = reader.readtext(image_np, paragraph=False)
        
        word_boxes = []
        extracted_lines = []
        
        for (bbox, text, prob) in results:
            if prob > 0.1:
                extracted_lines.append(text)
                word_boxes.append({
                    "bbox": bbox,
                    "text": text,
                    "confidence": float(prob)
                })
                
        full_text = "\n".join(extracted_lines)
        logger.debug(f"OCR extracted {len(extracted_lines)} lines.")
        return full_text, word_boxes
    except Exception as e:
        logger.error(f"Text extraction failed: {e}")
        return "", []
