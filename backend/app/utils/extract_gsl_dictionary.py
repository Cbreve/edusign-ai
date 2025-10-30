import os
import re
import csv
import json
from uuid import uuid4
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional

# Optional deps (graceful fallback)
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None  # type: ignore

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore


WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW_PDF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "Harmonized_GSL_Dictionary_v3_2023.pdf"))
RAW_PDF_FALLBACK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "Harmonized_GSL_Dictionary.pdf"))

PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
IMAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sign_images"))


def ensure_dirs() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)


def pick_pdf_path() -> str:
    if os.path.exists(RAW_PDF):
        return RAW_PDF
    if os.path.exists(RAW_PDF_FALLBACK):
        return RAW_PDF_FALLBACK
    raise FileNotFoundError("No raw PDF found in backend/app/data/raw/")


def detect_pdf_type(pdf_path: str) -> str:
    if pdfplumber is None:
        return "image_or_unknown"
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:2]:
                txt = page.extract_text() or ""
                if txt.strip():
                    return "text"
    except Exception:
        pass
    return "image_or_unknown"


def extract_text_lines(pdf_path: str) -> List[Tuple[int, str]]:
    lines: List[Tuple[int, str]] = []
    if pdfplumber is None:
        return lines
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for raw_line in text.splitlines():
                ln = raw_line.strip()
                if not ln:
                    continue
                lines.append((i, ln))
    return lines


SIGN_SEP = re.compile(r"^\s*([A-Z0-9\s\-\,\(\)\/]+?)\s*[:\–\-—]\s*(.+)$")
SIGN_IMPLICIT = re.compile(r"^\s*([A-Z0-9\s\-\,\(\)\/]{2,})\s+([A-Za-z].+)$")
ONLY_UPPER = re.compile(r"^[A-Z0-9\s\-\,\(\)\/]+$")


DROP_SUBSTRINGS = {
    "PREFACE", "FOREWORD", "ACKNOWLEDGEMENT", "ACKNOWLEDGEMENTS", "COPYRIGHT", "CONTENTS",
    "INDEX", "INTRODUCTION", "TABLE OF CONTENTS", "LIST OF FIGURES", "LIST OF TABLES",
    "GLOSSARY", "BIBLIOGRAPHY", "APPENDIX", "HARMONIZED GSL DICTIONARY", "GSL DICTIONARY",
}


def normalize_sign(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().upper()


def normalize_meaning(text: str) -> str:
    t = re.sub(r"\s+", " ", text).strip()
    # ensure sentence-like
    return t


def parse_entries(lines: List[Tuple[int, str]], source_pdf: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    buffer_sign: Optional[str] = None
    buffer_page: Optional[int] = None
    buffer_meaning_parts: List[str] = []

    def flush() -> None:
        nonlocal buffer_sign, buffer_meaning_parts, buffer_page
        if not buffer_sign:
            return
        meaning = normalize_meaning(" ".join(buffer_meaning_parts).strip())
        if not meaning or re.search(r"^[A-Z\s\-\,\(\)\/]+$", meaning):
            buffer_sign = None
            buffer_meaning_parts = []
            buffer_page = None
            return
        entry = {
            "id": str(uuid4()),
            "sign": normalize_sign(buffer_sign),
            "meaning": meaning,
            "page": buffer_page or 0,
            "image_refs": [],
            "source": os.path.relpath(source_pdf, WORKSPACE),
            "last_updated": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "language": "en",
        }
        entries.append(entry)
        buffer_sign = None
        buffer_meaning_parts = []
        buffer_page = None

    for page, raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.upper() in DROP_SUBSTRINGS:
            continue
        # explicit separators
        m = SIGN_SEP.match(line)
        if m:
            flush()
            lhs = normalize_sign(m.group(1))
            rhs = m.group(2).strip()
            # Split multiple signs if space-separated (short tokens)
            tokens = [t for t in re.split(r"\s+", lhs) if t]
            if len(tokens) > 1 and all(len(t) <= 12 for t in tokens):
                for t in tokens:
                    buffer_sign = t
                    buffer_page = page
                    buffer_meaning_parts = [rhs]
                    flush()
            else:
                buffer_sign = lhs
                buffer_page = page
                buffer_meaning_parts = [rhs]
            continue

        # implicit separator on same line
        m2 = SIGN_IMPLICIT.match(line)
        if m2:
            flush()
            buffer_sign = normalize_sign(m2.group(1))
            buffer_page = page
            buffer_meaning_parts = [m2.group(2).strip()]
            continue

        # standalone uppercase line -> start or continue sign
        if ONLY_UPPER.match(line) and 1 <= len(line.split()) <= 3 and not any(ch.isdigit() for ch in line):
            flush()
            buffer_sign = normalize_sign(line)
            buffer_page = page
            buffer_meaning_parts = []
            continue

        # otherwise likely meaning continuation
        if buffer_sign:
            buffer_meaning_parts.append(line)

    flush()
    # deduplicate by (sign, meaning)
    seen = set()
    unique: List[Dict[str, Any]] = []
    for e in entries:
        key = (e["sign"], e["meaning"]) 
        if key in seen:
            continue
        seen.add(key)
        unique.append(e)
    return unique


def extract_images(pdf_path: str) -> Dict[int, List[str]]:
    page_to_imgs: Dict[int, List[str]] = {}
    if fitz is None:
        return page_to_imgs
    try:
        doc = fitz.open(pdf_path)
    except Exception:
        return page_to_imgs
    for page_index in range(len(doc)):
        page = doc[page_index]
        img_list = page.get_images(full=True)
        if not img_list:
            continue
        saved: List[str] = []
        for img_idx, img in enumerate(img_list, start=1):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                out_name = f"page_{page_index+1}_img_{img_idx}.png"
                out_path = os.path.join(IMAGES_DIR, out_name)
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                pix.save(out_path)
                saved.append(os.path.relpath(out_path, WORKSPACE))
            except Exception:
                continue
        if saved:
            page_to_imgs[page_index + 1] = saved
    return page_to_imgs


def attach_images(entries: List[Dict[str, Any]], page_to_imgs: Dict[int, List[str]]) -> None:
    for e in entries:
        imgs = page_to_imgs.get(e.get("page", 0), [])
        if imgs:
            e["image_refs"] = imgs


def save_outputs(entries: List[Dict[str, Any]]) -> Tuple[str, str]:
    json_path = os.path.join(PROCESSED_DIR, "gsl_dictionary.json")
    csv_path = os.path.join(PROCESSED_DIR, "gsl_dictionary.csv")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    fieldnames = ["id", "sign", "meaning", "page", "image_refs", "source", "last_updated", "language"]
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for e in entries:
            row = dict(e)
            # serialize list
            row["image_refs"] = ",".join(row.get("image_refs", []))
            writer.writerow(row)
    return json_path, csv_path


def main() -> None:
    ensure_dirs()
    pdf_path = pick_pdf_path()
    kind = detect_pdf_type(pdf_path)
    print(f"PDF type: {kind}")
    if kind != "text":
        print("Warning: falling back to text extraction attempt; OCR is not enabled in this script run.")

    lines = extract_text_lines(pdf_path)
    entries = parse_entries(lines, pdf_path)
    page_to_imgs = extract_images(pdf_path)
    attach_images(entries, page_to_imgs)

    jp, cp = save_outputs(entries)
    print(f"Saved JSON: {jp}")
    print(f"Saved CSV:  {cp}")
    print(f"Total entries: {len(entries)}")
    for sample in entries[:5]:
        print({k: sample[k] for k in ("sign", "meaning", "page")})


if __name__ == "__main__":
    main()
