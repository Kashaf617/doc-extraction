import cv2
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger("DocumentAI.Utils")

def preprocess_image(img_cv: np.ndarray, max_w: int = 800, max_h: int = 1200) -> np.ndarray:
    """
    Standard professional preprocessing: Resize and Blur for OCR stability.
    
    Args:
        img_cv: Input image in BGR format.
        max_w: Maximum width constraint.
        max_h: Maximum height constraint.
        
    Returns:
        Preprocessed grayscale image.
    """
    try:
        # Get current dimensions
        h, w = img_cv.shape[:2]
        
        # Calculate aspect ratio preserving scale
        scale = min(max_w / w, max_h / h)
        
        # Only resize if the image is larger than the constraints
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image for OCR speed: {w}x{h} -> {new_w}x{new_h}")

        # Convert to grayscale for better OCR contrast
        if len(img_cv.shape) == 3:
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_cv
            
        return gray
    except Exception as e:
        logger.error(f"Image preprocessing failed: {e}")
        return cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY) if len(img_cv.shape) == 3 else img_cv
