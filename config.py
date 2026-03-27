import os
import torch
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Model Configurations
MODEL_CLIP = "openai/clip-vit-base-patch32"
MODEL_LAYOUTLM = "microsoft/layoutlmv3-base"
MODEL_QWEN = "Qwen/Qwen2.5-0.5B-Instruct"

# Pipeline Settings
OCR_MAX_WIDTH = 800
OCR_MAX_HEIGHT = 1200
SHORT_CIRCUIT_THRESHOLD = 20  # Character distance for confidence bypass
LLM_MAX_NEW_TOKENS = 300
LLM_TEMPERATURE = 0.0

# Device settings
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = DEVICE == "cuda"

# Upload Settings
UPLOAD_DIR = BASE_DIR / "temp_uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
