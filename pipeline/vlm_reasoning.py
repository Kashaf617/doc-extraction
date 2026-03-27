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
    detected_keywords: List[str],
    candidate_dates: List[str],
    relevant_text_snippet: str
) -> Tuple[str, float]:
    """
    Uses LLM reasoning to determine the expiry date from context.
    
    Args:
        document_type: Classified type of document.
        detected_keywords: Keywords found in OCR text.
        candidate_dates: List of date strings found by regex.
        relevant_text_snippet: Text segment containing potential dates.
        
    Returns:
        A tuple of (reasoned_date, confidence_score).
    """
    try:
        tokenizer, model = load_qwen_model()
        
        system_prompt = (
            "You are a Senior Document AI Judge.\n"
            "Task: Identify the 'Expiry Date' from the candidates based on document context.\n\n"
            "STRICT RULES:\n"
            "1. EXCLUDE: 'Date of Birth', 'Issue Date', 'Date of Issue'.\n"
            "2. PRIORITY: The LATEST date that is explicitly labeled as 'Expiry', 'Until', 'Valid', or 'Due'.\n"
            "3. JUDGMENT: If multiple future dates exist, choose the most logical expiration date.\n"
            "4. OUTPUT: 'Final Date: <YYYY-MM-DD or NULL>'. No extra text."
        )

        kw_str = ", ".join(detected_keywords) if detected_keywords else "None"
        cand_str = ", ".join(candidate_dates) if candidate_dates else "None"
        
        user_prompt = (
            f"Document Type: {document_type}\n"
            f"Keywords Found: [{kw_str}]\n"
            f"Candidate Dates: [{cand_str}]\n"
            f"Context Snippet: {relevant_text_snippet}\n\n"
            "Identify the Expiry Date."
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
            max_new_tokens=40, # Reduced for speed
            temperature=0.1,
            do_sample=False
        )
        
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        
        full_response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        logger.info(f"LLM Full Response:\n{full_response}")
        
        # Robust Parsing: Scan for the date anywhere in the response
        final_date = "NULL"
        # Look for ISO, DD.MM.YYYY, DD-MM-YYYY, DD MMM YYYY patterns
        date_patterns = [
            r'\d{4}[-/]\d{2}[-/]\d{2}',     # 2024-05-22
            r'\d{1,2}[.\-/]\d{1,2}[.\-/]\d{2,4}', # 22.05.2024
            r'\d{1,2}[\s.\-/][A-Z]{3,9}[\s.\-/]\d{2,4}'  # 22-MAY-2024
        ]
        
        found_dates = []
        for p in date_patterns:
            matches = re.findall(p, full_response, re.IGNORECASE)
            if matches: found_dates.extend(matches)
        
        if found_dates:
            # Use the first date found in the response
            final_date = found_dates[0]
            confidence = 0.95 if final_date in candidate_dates else 0.80
        elif "NULL" in full_response.upper():
            final_date = "NULL"
            confidence = 0.90
        else:
            final_date = "NULL"
            confidence = 0.0
        
        return final_date, confidence
    except Exception as e:
        logger.error(f"LLM reasoning failed: {e}")
        return "NULL", 0.0
