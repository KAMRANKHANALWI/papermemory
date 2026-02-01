# utils/pdf_processor.py
import fitz
import re
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter

def extract_headings_from_blocks(blocks: List[Dict]) -> List[str]:
    """Extract potential headings from text blocks based on font size and formatting."""
    headings = []
    for block in blocks.get("blocks", []):
        if block.get("type") == 0:  # Text block
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    font_size = span.get("size", 0)
                    font_flags = span.get("flags", 0)
                    if (font_size > 14 or (font_flags & 16) or (len(text) < 100 and text.isupper())):
                        if text and len(text.strip()) > 3:
                            headings.append(text)
    return headings[:3]

def process_pdf_with_pymupdf(pdf_path: str) -> List[Dict]:
    """Extract text and metadata from PDF using PyMuPDF."""
    try:
        doc = fitz.open(pdf_path)
        pages_data = []
        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            blocks = page.get_text("dict")
            headings = extract_headings_from_blocks(blocks)
            text = re.sub(r"\n+", "\n", text)
            text = re.sub(r"\s+", " ", text)
            pages_data.append({
                "text": text.strip(),
                "page_num": page_num + 1,
                "headings": headings,
                "char_count": len(text)
            })
        doc.close()
        return pages_data
    except Exception as e:
        return []

def chunk_text_content(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    """Split text into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max_chars,
        chunk_overlap=overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ", ", " ", ""]
    )
    return splitter.split_text(text)