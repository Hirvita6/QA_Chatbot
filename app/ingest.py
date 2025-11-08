import os
from typing import List
from pathlib import Path
import pdfplumber
from pypdf import PdfReader
from docx import Document
import pandas as pd

# Parsers
def parse_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def parse_pdf(path: str) -> str:
    try:
        text = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                text.append(page.extract_text() or "")
        return "".join(text)
    except Exception:
    # fallback to pypdf
        reader = PdfReader(path)
        text = []
        for p in reader.pages:
            text.append(p.extract_text() or "")
        return "".join(text)


def parse_docx(path: str) -> str:
    doc = Document(path)
    paras = [p.text for p in doc.paragraphs]
    return "".join(paras)



def parse_xlsx(path: str) -> str:
    # read all sheets and concatenate
    df = pd.read_excel(path, sheet_name=0)
    # if QA column present, join Q/A pairs, else join textual cells
    key_cols = [c for c in df.columns if str(c).lower() in ("q", "question", "a", "answer")]
    if key_cols:
        texts = []
        for _, row in df.iterrows():
            pieces = []
            for c in key_cols:
                val = row.get(c)
                if pd.notna(val):
                    pieces.append(str(val))
            if pieces:
                texts.append(" -- ".join(pieces))
            return "".join(texts)
    else:
        return df.astype(str).apply(lambda row: " ".join(row.values), axis=1).str.cat(sep="")


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    """Simple character-based chunker."""
    if not text:
        return []
    text = text.replace("", "")
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def parse_and_chunk_file(path: str):
    p = Path(path)
    ext = p.suffix.lower()
    if ext in (".txt",):
        content = parse_txt(path)
    elif ext in (".pdf",):
        content = parse_pdf(path)
    elif ext in (".docx", ".doc"):
        content = parse_docx(path)
    elif ext in (".xlsx", ".xls", ".csv"):
        content = parse_xlsx(path)
    else:
        # attempt to read as text
        content = parse_txt(path)
    chunks = chunk_text(content)
    return chunks