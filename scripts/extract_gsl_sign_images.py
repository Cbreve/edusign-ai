#!/usr/bin/env python3
"""
Extract sign language images from GSL Dictionary PDF (pages 1-292).
Filters out non-sign images like title pages, text-only pages, and logos.
"""
import os
import json
import sys
import io
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF

# Project root paths
PROJECT_ROOT = Path(__file__).parent.parent
PDF_PATH = PROJECT_ROOT / "backend/app/data/raw/Harmonized_GSL_Dictionary.pdf"
OUTPUT_DIR = PROJECT_ROOT / "backend/app/data/raw/sign_images"
DICT_PATH = PROJECT_ROOT / "backend/app/data/processed/gsl_dictionary.json"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def is_sign_image(image: Image.Image, page_num: int, img_index: int) -> bool:
    """
    Filter function to identify actual sign images.
    More permissive filter - extract all reasonable images, filtering only obvious non-sign content.
    """
    width, height = image.size
    
    # Skip very tiny images (icons, thumbnails)
    if width < 30 or height < 30:
        return False
    
    # Skip extremely large images (full-page scans > 3500px) - these are likely not sign images
    if width > 3500 or height > 3500:
        return False
    
    # Very permissive: extract from all pages including 1-2 (user said pages 1-292 contain signs)
    # Only skip if image is clearly not a sign (e.g., extremely wide banners)
    
    # Skip extremely wide banner-like images (width > 5x height) - these are headers
    if width > height * 5:
        return False
    
    # Extract everything else that passes basic size checks
    return True


def extract_sign_images_from_pdf(pdf_path: Path, output_dir: Path, start_page=1, end_page=292):
    """Extract and save sign images from PDF pages 1-292."""
    if not pdf_path.exists():
        print(f"Error: PDF not found at {pdf_path}")
        return []
    
    doc = fitz.open(pdf_path)
    extracted_images = []
    
    print(f"Processing PDF: {pdf_path.name}")
    print(f"Total pages in PDF: {len(doc)}")
    print(f"Extracting from pages {start_page} to {min(end_page, len(doc))}...")
    
    for page_num in range(start_page - 1, min(end_page, len(doc))):
        page = doc[page_num]
        image_list = page.get_images()
        
        for img_idx, img in enumerate(image_list):
            try:
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Convert to PIL Image for analysis
                image = Image.open(io.BytesIO(image_bytes))
                
                # Filter: only keep actual sign images
                if is_sign_image(image, page_num + 1, img_idx + 1):
                    # Save image
                    filename = f"page_{page_num + 1}_img_{img_idx + 1}.{image_ext}"
                    filepath = output_dir / filename
                    
                    with open(filepath, "wb") as f:
                        f.write(image_bytes)
                    
                    extracted_images.append({
                        "page": page_num + 1,
                        "image_index": img_idx + 1,
                        "filename": filename,
                        "width": image.width,
                        "height": image.height
                    })
                    
                    print(f"  ✓ Extracted: {filename} (page {page_num + 1})")
                else:
                    print(f"  ✗ Skipped non-sign image on page {page_num + 1} (img {img_idx + 1})")
                    
            except Exception as e:
                print(f"  ✗ Error extracting image {img_idx + 1} from page {page_num + 1}: {e}")
                continue
    
    doc.close()
    return extracted_images


def update_dictionary_with_images(dict_path: Path, extracted_images: list):
    """Update dictionary JSON to link extracted images to sign entries."""
    if not dict_path.exists():
        print(f"Warning: Dictionary not found at {dict_path}")
        return
    
    with open(dict_path, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    
    # Group images by page
    images_by_page = {}
    for img_info in extracted_images:
        page = img_info["page"]
        if page not in images_by_page:
            images_by_page[page] = []
        images_by_page[page].append(img_info["filename"])
    
    # Update dictionary entries
    updated_count = 0
    for entry in dictionary:
        page = entry.get("page")
        if page and page in images_by_page:
            entry["image_refs"] = images_by_page[page]
            updated_count += 1
    
    # Save updated dictionary
    with open(dict_path, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Updated {updated_count} dictionary entries with image references")


if __name__ == "__main__":
    print("=" * 60)
    print("GSL Sign Image Extraction Script")
    print("=" * 60)
    
    # Extract images
    extracted = extract_sign_images_from_pdf(PDF_PATH, OUTPUT_DIR, start_page=1, end_page=292)
    
    print(f"\n✓ Extraction complete: {len(extracted)} sign images extracted")
    
    # Update dictionary
    if extracted:
        update_dictionary_with_images(DICT_PATH, extracted)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

