import re
import logging
from typing import List, Dict, Any, Optional

# Standardized logging
logger = logging.getLogger("DocumentAI.DateExtractor")

# Negative keywords to filter out non-expiry dates
ISSUE_KEYWORDS: List[str] = [
    "issue", "vydania", "narodenia", "born", "dob", "birth", "period", "stmt", "invoice", "entry",
    "issued", "vydanie", "narodenie", "billing", "start date", "renewal", "renewal date", "issue/renewal"
]

# Standard date patterns for extraction
DATE_PATTERNS = [
    r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',        # DD-MM-YYYY
    r'\d{4}[/-]\d{1,2}[/-]\d{1,2}',        # YYYY-MM-DD
    r'\d{1,2}[\s.-]+[A-Za-z]{3,9}[\s.-]+\d{2,4}', # DD MMM YYYY
    r'[A-Za-z]{3,9}[\s.-]+\d{1,2}[\s.-]+\d{2,4}', # MMM DD YYYY
    r'\d{1,2}[A-Z]{3}\d{2,4}',              # DDMMMYYYY
    r'\d{8}',                               # YYYYMMDD
    r'\d{1,2}[\s/-]+\d{1,2}[\s/-]+\d{2}'    # DD-MM-YY
]

# Ultimate Consolidated Weighted Keyword Map
WEIGHTED_KEYWORDS: Dict[str, Any] = {
    "passport": {
        "passport": 1000, "republic": 10
    },
    "id_card": {
        "id-card": 1000, "identity card": 1000, "občiansky": 1000, "national id": 500, "citizen": 500
    },
    "driving_license": {
        "driving license": 1000, "driving licence": 1000, "licence": 500, "license": 500, 
        "traffic police": 1000, "punjab": 500, "permit": 500, "bangladesh": 1000, "brta": 1000, "motor": 500
    },
    "cnic": {
        "cnic": 1000, "nadra": 1000, "government of pakistan": 1000
    },
    "bill": {
        "electricity": 500, "bill": 500, "consumer": 500, "wapda": 500, "fesco": 500, "lesco": 500, "account no": 500
    }
}

def find_candidate_dates(text: str) -> List[str]:
    """Finds all candidate dates in the text."""
    res = []
    for p in DATE_PATTERNS:
        for m in re.finditer(p, text):
            res.append(m.group())
    return list(set(res))

def classify_by_keywords(text: str) -> str:
    """Consolidated weighted classification."""
    t = text.lower()
    res = {k: 0 for k in WEIGHTED_KEYWORDS}
    for k, weights in WEIGHTED_KEYWORDS.items():
        for kw, w in weights.items():
            if kw in t: res[k] += w
    if not res or max(res.values()) == 0: return "other"
    return max(res, key=lambda k: res[k])

def parse_date(date_str: str) -> Optional[int]:
    """Numeric YYYYMMDD parsing."""
    try:
        nums = re.findall(r'\d+', date_str)
        if len(nums) < 3: return None
        d, m, y = nums[0], nums[1], nums[2]
        if len(y) == 2: y = "20" + y
        if len(d) == 1: d = "0" + d
        if len(m) == 1: m = "0" + m
        if len(d) == 4: return int(f"{d}{m}{y}")
        return int(f"{y}{m}{d}")
    except: return None

def resolve_expiry_heuristically(candidates: List[str], text: str) -> Optional[str]:
    """Picks the latest date as the primary expiry candidate."""
    valid = []
    for c in candidates:
        v = parse_date(c)
        if v: valid.append((c, v))
    if not valid: return None
    valid.sort(key=lambda x: x[1], reverse=True)
    return valid[0][0]

def find_keyword_positions(text: str) -> List[Dict[str, Any]]:
    """Generic proximity indicators."""
    keys = ["expiry", "valid", "until", "due", "splatnosti", "platnosti", "till", "expiry date", "exp"]
    t = text.lower()
    res = []
    for k in keys:
        for m in re.finditer(re.escape(k), t):
            res.append({"keyword": k, "start": m.start(), "end": m.end()})
    return res

def score_dates(text: str, document_type: str) -> Dict[str, Any]:
    """High-confidence heuristic scoring."""
    try:
        cands = []
        for p in DATE_PATTERNS:
            for m in re.finditer(p, text):
                cands.append({"date": m.group(), "start": m.start(), "end": m.end()})
        
        target_keys = find_keyword_positions(text)
        issue_keys = []
        t = text.lower()
        for ik in ISSUE_KEYWORDS:
            for m in re.finditer(re.escape(ik), t):
                issue_keys.append({"start": m.start(), "end": m.end()})
        
        if not cands: return {"candidate_dates": []}
        
        best = None
        min_s = 1000000.0
        for c in cands:
            d_s, d_e = int(c["start"]), int(c["end"])
            dist = 5000.0
            for tk in target_keys:
                cur = float(abs(d_s - int(tk["end"]))) if d_s > int(tk["end"]) else float(abs(int(tk["start"]) - d_e))
                dist = min(dist, cur)
            
            penalty = 0.0
            for ik in issue_keys:
                cur = abs(d_s - int(ik["end"]))
                if cur < 60: penalty += (120 - cur) * 20 # Severe issue penalty
            
            score = dist + penalty
            if score < min_s:
                min_s = score
                best = {"date": c["date"], "score": score}
        
        unique = list(set([str(ca["date"]) for ca in cands]))
        temporal = resolve_expiry_heuristically(unique, text)
        
        # High confidence signal: Heuristic matches Temporal
        confidence = 0.0
        if best and temporal and best["date"] == temporal:
            confidence = 1.0 # Maximum heuristic confidence
            
        return {
            "candidate_dates": unique,
            "best_match_heuristic": best,
            "temporal_best": temporal,
            "confidence": confidence
        }
    except: return {"candidate_dates": []}
