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
    Uses LLM reasoning to determine BOTH document type and expiry date.
    
    Returns:
        A tuple of (verified_type, verified_date, confidence_score).
    """
    try:
        tokenizer, model = load_qwen_model()
        
        system_prompt = (
            "You are a Senior Document AI Judge.\n"
            "Tasks:\n"
            "1. IDENTIFY THE DOCUMENT TYPE (e.g. passport, id_card, driving_license, cnic, bill).\n"
            "2. IDENTIFY THE EXPIRY DATE.\n\n"
            "STRICT RULES:\n"
            "1. DATES: Exclude Date of Birth and Issue Date. Pick the logical expiration date.\n"
            "2. TYPES: If 'Punjab' or 'Traffic Police' or 'License' is seen, it is a driving_license.\n"
            "3. OUTPUT FORMAT: 'Final Type: <type> | Final Date: <YYYY-MM-DD or NULL>'. No extra text."
        )

        cand_str = ", ".join(candidate_dates) if candidate_dates else "None"
        user_prompt = (
            f"Heuristic Type: {document_type}\n"
            f"Candidate Dates: [{cand_str}]\n"
            f"Context Snippet: {relevant_text_snippet}\n\n"
            "Identify the Final Type and Final Date."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(DEVICE)

        s = time.time()
        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=60, 
            temperature=0.1,
            do_sample=False
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        full_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        logger.info(f"LLM Full Response:\n{full_response}")
        
        # Robust Parsing
        verified_type = document_type
        if "Final Type:" in full_response:
            v_type_raw = full_response.split("Final Type:")[-1].split("|")[0].strip().lower()
            if v_type_raw in ["passport", "id_card", "driving_license", "cnic", "bill", "utility_bill"]:
                verified_type = v_type_raw
        
        verified_date = "NULL"
        date_patterns = [
            r'\d{4}[-/]\d{2}[-/]\d{2}',     
            r'\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}', 
            r'\d{1,2}[\s.\-/][A-Z]{3,9}[\s.\-/]\d{2,4}'  
        ]
        
        found_dates = []
        for p in date_patterns:
            matches = re.findall(p, full_response, re.IGNORECASE)
            if matches: found_dates.extend(matches)
        
        if found_dates:
            verified_date = found_dates[0]
            confidence = 0.95 if verified_date in candidate_dates else 0.80
        elif "NULL" in full_response.upper():
            verified_date = "NULL"
            confidence = 0.90
        else:
            verified_date = "NULL"
            confidence = 0.4 # Uncertain
        
        return verified_type, verified_date, confidence
    except Exception as e:
        logger.error(f"LLM unified reasoning failed: {e}")
        return document_type, "NULL", 0.0
