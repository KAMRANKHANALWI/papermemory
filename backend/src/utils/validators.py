"""
Validation utilities for collection names and other inputs
"""
import re
from typing import Tuple


def validate_collection_name(name: str) -> Tuple[bool, str, str]:
    """
    Validate and clean collection name for ChromaDB requirements.
    
    ChromaDB requirements:
    - Length: 3-63 characters
    - Pattern: [a-zA-Z0-9][a-zA-Z0-9._-]*
    - No leading/trailing periods or underscores
    
    Args:
        name: Collection name to validate
        
    Returns:
        Tuple of (is_valid, cleaned_name, message)
    """
    if not name:
        return False, "", "Collection name cannot be empty"
    
    # Clean the name
    cleaned = name.strip()
    
    # Replace spaces with underscores
    cleaned = cleaned.replace(" ", "_")
    
    # Remove invalid characters
    cleaned = re.sub(r'[^a-zA-Z0-9._-]', '', cleaned)
    
    # Remove leading/trailing periods and underscores
    cleaned = cleaned.strip("._")
    
    # Ensure it starts with alphanumeric
    if cleaned and not cleaned[0].isalnum():
        cleaned = "col_" + cleaned
    
    # Check length
    if len(cleaned) < 3:
        return False, cleaned, "Collection name must be at least 3 characters"
    
    if len(cleaned) > 63:
        cleaned = cleaned[:63]
    
    # Validate pattern
    pattern = r'^[a-zA-Z0-9][a-zA-Z0-9._-]*$'
    if not re.match(pattern, cleaned):
        return False, cleaned, "Collection name contains invalid characters"
    
    return True, cleaned, "Valid collection name"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe storage.
    
    Args:
        filename: Original filename
        
    Returns:
        Sanitized filename
    """
    # Keep only alphanumeric, spaces, dots, hyphens, underscores
    sanitized = re.sub(r'[^a-zA-Z0-9\s._-]', '', filename)
    
    # Replace multiple spaces with single space
    sanitized = re.sub(r'\s+', ' ', sanitized)
    
    # Trim
    sanitized = sanitized.strip()
    
    return sanitized


def is_valid_pdf_filename(filename: str) -> bool:
    """
    Check if filename is a valid PDF filename.
    
    Args:
        filename: Filename to check
        
    Returns:
        True if valid PDF filename
    """
    return filename.lower().endswith('.pdf') and len(filename) > 4