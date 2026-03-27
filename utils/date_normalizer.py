import re
from datetime import datetime

def normalize_to_iso(date_str: str) -> str:
    """
    Converts various date formats to ISO format (YYYY-MM-DD).
    Returns the normalized string, or the original string if parsing fails.
    """
    if date_str == "NULL" or not date_str:
        return "NULL"
        
    date_str = str(date_str).strip()
    
    # Common format mappings
    formats_to_try = [
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d.%m.%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%d-%b-%Y",
        "%d-%b-%y"
    ]
    
    for fmt in formats_to_try:
        try:
            parsed = datetime.strptime(date_str, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
            
    # Try slightly fuzzier parsing if direct formats fail
    parts = re.split(r'[\s./-]+', date_str)
    if len(parts) == 3:
        # Check for DD MMM YYYY / DD-MMM-YY
        try:
            day = parts[0]
            month = parts[1]
            year = parts[2]
            
            # Normalize year if 2 digits (assume 2000s for expiry dates)
            if len(year) == 2:
                year = "20" + year
                
            if not month.isdigit():
                month_str = str(month)
                month_obj = datetime.strptime(month_str[:3], "%b") # type: ignore
                month = month_obj.strftime("%m")
                
            # Reconstruct
            iso_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            # Validate
            datetime.strptime(iso_str, "%Y-%m-%d")
            return iso_str
            
        except Exception:
            pass
            
    # Fallback to original if we can't easily parse
    return str(date_str)
