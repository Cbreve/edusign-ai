import os
import re
import csv
import json
from uuid import uuid4
from datetime import datetime
from typing import List, Tuple, Dict, Any, Optional
import sys

# Optional deps (graceful fallback)
try:
    import pdfplumber  # type: ignore
except Exception:
    pdfplumber = None  # type: ignore

try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None  # type: ignore

try:
    import pytesseract  # type: ignore
    from PIL import Image  # type: ignore
except Exception:
    pytesseract = None  # type: ignore
    Image = None  # type: ignore


WORKSPACE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
RAW_PDF = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "Harmonized_GSL_Dictionary_v3_2023.pdf"))
RAW_PDF_FALLBACK = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "Harmonized_GSL_Dictionary.pdf"))

PROCESSED_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "processed"))
IMAGES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw", "sign_images"))
GOLD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "gold"))

# Heuristic: ignore likely front matter pages to reduce noise
FRONT_MATTER_CUTOFF_PAGE = 10


def ensure_dirs() -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)
    os.makedirs(GOLD_DIR, exist_ok=True)


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
            # column-aware: use words with coordinates and split into columns
            try:
                words = page.extract_words(x_tolerance=2, y_tolerance=3) or []
            except Exception:
                words = []
            if words:
                # heuristic two-column split by page mid x
                mid_x = (page.bbox[0] + page.bbox[2]) / 2.0
                left_words = [w for w in words if float(w.get("x0", 0)) < mid_x]
                right_words = [w for w in words if float(w.get("x0", 0)) >= mid_x]

                def words_to_lines(ws: List[Dict[str, Any]]) -> List[str]:
                    if not ws:
                        return []
                    # group by line using y0 proximity
                    ws_sorted = sorted(ws, key=lambda w: (round(float(w.get("top", 0))/3)*3, float(w.get("x0", 0))))
                    lines_acc: List[List[str]] = []
                    last_top: Optional[float] = None
                    cur: List[str] = []
                    for w in ws_sorted:
                        top = float(w.get("top", 0))
                        text = (w.get("text") or "").strip()
                        if not text:
                            continue
                        if last_top is None or abs(top - last_top) <= 5:
                            cur.append(text)
                            last_top = top if last_top is None else (last_top + top) / 2
                        else:
                            if cur:
                                lines_acc.append(cur)
                            cur = [text]
                            last_top = top
                    if cur:
                        lines_acc.append(cur)
                    return [" ".join(parts).strip() for parts in lines_acc if parts]

                left_lines = words_to_lines(left_words)
                right_lines = words_to_lines(right_words)
                for ln in left_lines + right_lines:  # read left column fully, then right
                    if ln:
                        lines.append((i, ln))
            else:
                text = page.extract_text() or ""
                for raw_line in text.splitlines():
                    ln = raw_line.strip()
                    if not ln:
                        continue
                    lines.append((i, ln))
    return lines


def ocr_page_to_lines(pdf_path: str, page_index: int, conf_threshold: int = 70) -> List[str]:
    if fitz is None or pytesseract is None or Image is None:
        return []
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_index]
        pix = page.get_pixmap(dpi=300)
        img_data = pix.tobytes("png")
        from io import BytesIO
        img = Image.open(BytesIO(img_data))
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        n = len(data.get("text", []))
        words = []
        for idx in range(n):
            txt = (data["text"][idx] or "").strip()
            conf = int(data.get("conf", ["-1"])[idx]) if str(data.get("conf", ["-1"])[idx]).isdigit() else -1
            if not txt or conf < conf_threshold:
                continue
            top = int(data.get("top", [0])[idx])
            left = int(data.get("left", [0])[idx])
            words.append({"text": txt, "top": top, "x0": left})
        if not words:
            return []
        # group into lines similar to words_to_lines above
        words_sorted = sorted(words, key=lambda w: (round(float(w.get("top", 0))/3)*3, float(w.get("x0", 0))))
        lines_acc: List[List[str]] = []
        last_top: Optional[float] = None
        cur: List[str] = []
        for w in words_sorted:
            top = float(w.get("top", 0))
            text = (w.get("text") or "").strip()
            if not text:
                continue
            if last_top is None or abs(top - last_top) <= 8:
                cur.append(text)
                last_top = top if last_top is None else (last_top + top) / 2
            else:
                if cur:
                    lines_acc.append(cur)
                cur = [text]
                last_top = top
        if cur:
            lines_acc.append(cur)
        return [" ".join(parts).strip() for parts in lines_acc if parts]
    except Exception:
        return []


SIGN_SEP = re.compile(r"^\s*([A-Z0-9\s\-\,\(\)\/]+?)\s*[:\–\-—]\s*(.+)$")
SIGN_IMPLICIT = re.compile(r"^\s*([A-Z0-9\s\-\,\(\)\/]{2,})\s+([A-Za-z].+)$")
ONLY_UPPER = re.compile(r"^[A-Z0-9\s\-\,\(\)\/]+$")


DROP_SUBSTRINGS = {
    "PREFACE", "FOREWORD", "ACKNOWLEDGEMENT", "ACKNOWLEDGEMENTS", "COPYRIGHT", "CONTENTS",
    "INDEX", "INTRODUCTION", "TABLE OF CONTENTS", "LIST OF FIGURES", "LIST OF TABLES",
    "GLOSSARY", "BIBLIOGRAPHY", "APPENDIX", "HARMONIZED GSL DICTIONARY", "GSL DICTIONARY",
    "GOVERNMENT OF GHANA", "MINISTRY", "GHANA", "EDITION", "PUBLISHED", "AUTHOR", "AUTHORS",
}

# Additional noise terms to exclude when they appear as signs
NOISE_SIGN_TERMS = {
    "DICTIONARY", "COPYRIGHT", "GOVERNMENT", "GHANA", "PUBLISHED", "EDITION", "FOREWORD",
    "PREFACE", "ACKNOWLEDGEMENT", "ACKNOWLEDGEMENTS", "MINISTRY", "SERVICE", "SCHOOL",
}


def normalize_sign(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().upper()


def normalize_meaning(text: str) -> str:
    # fix smart punctuation and dashes, remove extra spaces
    t = text.replace("\u2013", "-").replace("\u2014", "-")
    t = t.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    # collapse spaces
    t = re.sub(r"\s+", " ", t).strip()
    # ensure sentence-like
    return t


def is_noise_sign(sign: str) -> bool:
    tokens = [t for t in sign.split() if t]
    if len(tokens) > 4:  # overly long sign names are often headings
        return True
    if any(term in sign for term in NOISE_SIGN_TERMS):
        return True
    # reject signs that are mostly punctuation/symbols
    if len(re.sub(r"[A-Z0-9\s\-\,\(\)\/]", "", sign)) > 0:
        return True
    return False


def is_noise_meaning(meaning: str) -> bool:
    # must contain lowercase letters and at least 3 words to be sentence-like
    if not re.search(r"[a-z]", meaning):
        return True
    if len(meaning.split()) < 3:
        return True
    # exclude boilerplate legal/metadata lines
    if any(term in meaning.upper() for term in ["COPYRIGHT", "ALL RIGHTS RESERVED", "PUBLISHED BY", "MINISTRY"]):
        return True
    return False


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
        # page-based filter to avoid front matter noise
        if (buffer_page or 0) < FRONT_MATTER_CUTOFF_PAGE:
            buffer_sign = None
            buffer_meaning_parts = []
            buffer_page = None
            return
        # content-based noise checks
        if not meaning or re.search(r"^[A-Z\s\-\,\(\)\/]+$", meaning):
            buffer_sign = None
            buffer_meaning_parts = []
            buffer_page = None
            return
        if is_noise_sign(buffer_sign):
            buffer_sign = None
            buffer_meaning_parts = []
            buffer_page = None
            return
        if is_noise_meaning(meaning):
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


def validate_against_gold(entries: List[Dict[str, Any]], gold_path: str) -> Dict[str, Any]:
    report = {"gold_items": 0, "found": 0, "missing": 0, "missing_items": []}
    try:
        with open(gold_path, "r", encoding="utf-8") as f:
            gold = json.load(f)
    except Exception:
        return {"error": f"No gold file at {gold_path}"}
    report["gold_items"] = len(gold)
    by_sign = {e["sign"]: e for e in entries}
    for g in gold:
        sign = g.get("sign", "").upper()
        expected = g.get("meaning_contains", "").lower()
        ok = False
        e = by_sign.get(sign)
        if e and expected and expected in e.get("meaning", "").lower():
            ok = True
        if ok:
            report["found"] += 1
        else:
            report["missing"] += 1
            report["missing_items"].append(g)
    return report


def main() -> None:
    ensure_dirs()
    synthetic_forced = os.getenv("FORCE_SYNTHETIC") == "1"
    lines: List[Tuple[int, str]] = []
    if synthetic_forced:
        print("Using synthetic dataset (FORCE_SYNTHETIC=1)")
        lines = [
            (16, "COLOUR: Wiggle the fingers and move the hand down the face."),
            (16, "BLUE: Shake the \"B\" hand at the wrist."),
            (16, "GREEN: Shake the \"G\" hand at the wrist."),
            (16, "RED: Move the index finger down your chin."),
            (16, "YELLOW: Shake the \"Y\" hand at the wrist."),
            (17, "PURPLE: Place the index finger of the \"P\" hand on your temple and rotate your hand at the wrist."),
            (17, "PINK: Move the middle finger of the \"P\" hand down your lips."),
        ]
        pdf_path = "backend/app/data/raw/SYNTHETIC.pdf"
        kind = "synthetic"
    else:
        try:
            pdf_path = pick_pdf_path()
            kind = detect_pdf_type(pdf_path)
            print(f"PDF type: {kind}")
            lines = extract_text_lines(pdf_path)
            # OCR fallback per page (confidence gated) if nothing extracted
            if not lines and fitz is not None:
                print("No text extracted; attempting OCR on all pages with confidence gating...")
                try:
                    doc = fitz.open(pdf_path)
                    for idx in range(len(doc)):
                        ocr_lines = ocr_page_to_lines(pdf_path, idx)
                        for ln in ocr_lines:
                            if ln:
                                lines.append((idx + 1, ln))
                except Exception:
                    pass
        except FileNotFoundError:
            if os.getenv("ALLOW_SYNTHETIC") == "1":
                print("Raw PDF not found; using synthetic dataset (ALLOW_SYNTHETIC=1)")
                lines = [
                    (16, "COLOUR: Wiggle the fingers and move the hand down the face."),
                    (16, "BLUE: Shake the \"B\" hand at the wrist."),
                    (16, "GREEN: Shake the \"G\" hand at the wrist."),
                    (16, "RED: Move the index finger down your chin."),
                    (16, "YELLOW: Shake the \"Y\" hand at the wrist."),
                    (17, "PURPLE: Place the index finger of the \"P\" hand on your temple and rotate your hand at the wrist."),
                    (17, "PINK: Move the middle finger of the \"P\" hand down your lips."),
                ]
                pdf_path = "backend/app/data/raw/SYNTHETIC.pdf"
                kind = "synthetic"
            else:
                raise
    entries = parse_entries(lines, pdf_path)
    page_to_imgs = extract_images(pdf_path)
    attach_images(entries, page_to_imgs)

    jp, cp = save_outputs(entries)
    print(f"Saved JSON: {jp}")
    print(f"Saved CSV:  {cp}")
    print(f"Total entries: {len(entries)}")
    for sample in entries[:5]:
        print({k: sample[k] for k in ("sign", "meaning", "page")})

    # Quality validation against gold set (if present)
    gold_path = os.path.join(GOLD_DIR, "gsl_gold.json")
    if os.path.exists(gold_path):
        report = validate_against_gold(entries, gold_path)
        print("Gold validation:", json.dumps(report, ensure_ascii=False))
        if os.getenv("FAIL_ON_GOLD_MISSING") == "1":
            if isinstance(report, dict) and report.get("missing", 0) > 0:
                print("Gold validation failed: missing items detected", file=sys.stderr)
                sys.exit(1)


if __name__ == "__main__":
    main()
