# SPDX-License-Identifier: MIT
# Licensed under the MIT License. See the LICENSE file in the repository root.
from pathlib import Path
from typing import List


def _missing_dependency_error(import_name: str, package_name: str, suffix: str) -> ImportError:
    return ImportError(
        f"Reading {suffix} files requires the optional dependency '{package_name}'. "
        f"Install it with `pip install {package_name}`."
    )


def extract_text_from_file(path: Path) -> List[str]:
    """
    Extracts text from a file and returns a list of text chunks (strings).
    """

    suffix = path.suffix.lower()

    if suffix == ".txt" or suffix == ".md":
        with open(path, "r", encoding="utf-8") as f:
            return [f.read()]

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise _missing_dependency_error("pypdf", "pypdf", suffix) from exc
        reader = PdfReader(str(path))
        return [page.extract_text() or "" for page in reader.pages]

    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise _missing_dependency_error("docx", "python-docx", suffix) from exc
        doc = Document(str(path))
        return [p.text for p in doc.paragraphs if p.text.strip()]

    if suffix == ".html":
        try:
            from bs4 import BeautifulSoup
        except ImportError as exc:
            raise _missing_dependency_error("bs4", "beautifulsoup4", suffix) from exc
        with open(path, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
            return [soup.get_text(separator="\n")]

    raise ValueError(f"Unsupported file type: {path.suffix}")
