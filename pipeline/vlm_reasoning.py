import os
import sys
import time
import logging
import re
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Import configurations
try:
    from document_ai_system.config import DEVICE
except ImportError:
    DEVICE = "cpu"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Standardized logging
logger = logging.getLogger("DocumentAI.VLM")

_MODEL = None
_TOKENIZER = None

def load_qwen_model():
    """Lazy loader for the 0.5B model."""
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        logger.info(f"Loading local VLM: {model_name} on {DEVICE}")
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map=DEVICE
        )
    return _TOKENIZER, _MODEL

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
            "1. TYPE: [passport, id_card, driving_license, cnic, bill]\n"
            "2. DATE: Look for 'VALIDITY' or 'UNTIL'. Skip 'Issue' and 'Birth'.\n"
            "3. FORMAT: YOU MUST OUTPUT ONLY A JSON OBJECT: {\"type\": \"...\", \"date\": \"YYYY-MM-DD\"}"
        )

        cand_str = ", ".join(candidate_dates) if candidate_dates else "None"
        snippet_clean = str(relevant_text_snippet)[:1200]
        user_prompt = f"Heuristic: {document_type}\nCandidates: {cand_str}\nText: {snippet_clean}\nOutput JSON:"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
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
            low_res = full_response.lower()
            if "driving" in low_res or "licence" in low_res: v_type = "driving_license"
            elif "passport" in low_res: v_type = "passport"
            
            date_match = re.search(r'\d{4}[-/]\d{2}[-/]\d{2}', full_response)
            if date_match: v_date = date_match.group()
            
        return str(v_type), str(v_date), 0.95 if v_date != "NULL" else 0.4
    except Exception as e:
        logger.error(f"LLM failure: {e}")
        return document_type, "NULL", 0.0
