#!/usr/bin/env python3
"""
Sync dictionary JSON with actual image files - remove references to deleted images.
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_ROOT / "backend/app/data/raw/sign_images"
DICT_PATH = PROJECT_ROOT / "backend/app/data/processed/gsl_dictionary.json"

def sync_dictionary():
    """Update dictionary to match actual image files."""
    if not DICT_PATH.exists():
        print(f"Error: Dictionary not found at {DICT_PATH}")
        return
    
    # Get list of actual image files
    image_files = set()
    for ext in ['*.png', '*.jpeg', '*.jpg']:
        image_files.update(f.name for f in IMAGES_DIR.glob(ext))
    
    print(f"Found {len(image_files)} image files in {IMAGES_DIR}")
    
    # Load dictionary
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    
    # Update dictionary entries
    updated_count = 0
    removed_refs_count = 0
    
    for entry in dictionary:
        if 'image_refs' in entry and entry['image_refs']:
            original_refs = entry['image_refs']
            # Keep only references that exist as files
            valid_refs = [ref for ref in original_refs if ref in image_files]
            
            if len(valid_refs) != len(original_refs):
                removed_count = len(original_refs) - len(valid_refs)
                removed_refs_count += removed_count
                entry['image_refs'] = valid_refs
                updated_count += 1
                print(f"  Updated {entry.get('sign', 'unknown')} (page {entry.get('page', '?')}): removed {removed_count} missing image reference(s)")
    
    # Save updated dictionary
    with open(DICT_PATH, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Synced dictionary:")
    print(f"  - Updated {updated_count} entries")
    print(f"  - Removed {removed_refs_count} references to deleted images")
    
    # Generate summary
    entries_with_images = sum(1 for e in dictionary if e.get('image_refs'))
    entries_without_images = len(dictionary) - entries_with_images
    
    print(f"\nDictionary summary:")
    print(f"  - Total entries: {len(dictionary)}")
    print(f"  - Entries with images: {entries_with_images}")
    print(f"  - Entries without images: {entries_without_images}")

if __name__ == "__main__":
    print("=" * 60)
    print("Syncing Dictionary with Image Files")
    print("=" * 60)
    sync_dictionary()
    print("\n" + "=" * 60)
    print("Sync complete!")
    print("=" * 60)

