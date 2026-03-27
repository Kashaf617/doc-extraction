import re
import json
import torch
import logging
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Tuple, List, Optional

try:
    from document_ai_system.config import MODEL_QWEN, DEVICE, USE_FP16, LLM_MAX_NEW_TOKENS, LLM_TEMPERATURE
except ImportError:
    MODEL_QWEN = "Qwen/Qwen2.5-0.5B-Instruct"
    DEVICE = "cpu"
    USE_FP16 = False
    LLM_MAX_NEW_TOKENS = 300
    LLM_TEMPERATURE = 0.0

# Standardized logging
logger = logging.getLogger("DocumentAI.Reasoner")

_tokenizer = None
_model = None

def load_qwen_model() -> Tuple[AutoTokenizer, AutoModelForCausalLM]:
    """
    Loads and returns the Qwen model and tokenizer (Singleton).
    """
    global _tokenizer, _model
    if _model is None:
        logger.info(f"Loading Reasoner Model: {MODEL_QWEN}...")
        try:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_QWEN, local_files_only=True)
            _model = AutoModelForCausalLM.from_pretrained(
                MODEL_QWEN,
                torch_dtype=torch.float16 if USE_FP16 else torch.float32,
                device_map=DEVICE,
                local_files_only=True
            )
        except Exception as e:
            logger.error(f"Failed to load reasoner: {e}")
            raise RuntimeError("Reasoner Initialization Failure") from e
    return _tokenizer, _model

def reason_expiry_date(
    document_type: str,
    candidate_dates: List[str],
    relevant_text_snippet: str
) -> Tuple[str, str, float]:
    """
    STRICT JSON REASONING for small models.
    """
    try:
        tokenizer, model = load_qwen_model()
        
        system_prompt = (
            "You are a Senior Document AI.\n"
            "Identify the Final Type and Expiry Date from the text.\n"
            "RULES:\n"
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=100, temperature=0.01, do_sample=False)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        full_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        logger.info(f"LLM RAW: {full_response}")
        
        # Robust JSON extraction
        v_type, v_date = document_type, "NULL"
        try:
            # Find JSON block
            match = re.search(r'\{.*\}', full_response, re.DOTALL)
            if match:
                data = json.loads(match.group())
                v_type = data.get("type", document_type)
                v_date = data.get("date", "NULL")
        except:
            # Fallback regex if JSON fails
            if "driving" in full_response.lower() or "licence" in full_response.lower(): v_type = "driving_license"
            date_match = re.search(r'\d{4}[-/]\d{2}[-/]\d{2}', full_response)
            if date_match: v_date = date_match.group()
            
        return str(v_type), str(v_date), 0.95 if v_date != "NULL" else 0.4
    except Exception as e:
        logger.error(f"LLM failure: {e}")
        return document_type, "NULL", 0.0
