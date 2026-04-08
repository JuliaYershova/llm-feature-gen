from pathlib import Path
from typing import List

def extract_text_from_file(path: Path) -> List[str]:
    """
    Extracts text from a file and returns a list of text chunks (strings).
    """

    suffix = path.suffix.lower()

    if suffix == ".txt" or suffix == ".md":
        with open(path, "r", encoding="utf-8") as f:
            return [f.read()]

    if suffix == ".pdf":
        from pypdf import PdfReader
        reader = PdfReader(str(path))
        return [page.extract_text() or "" for page in reader.pages]

    if suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        return [p.text for p in doc.paragraphs if p.text.strip()]

    if suffix == ".html":
        from bs4 import BeautifulSoup
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            return [soup.get_text(separator="\n")]

    raise ValueError(f"Unsupported file type: {path.suffix}")