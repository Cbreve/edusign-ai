#!/usr/bin/env python3
"""
Automatically fix dataset quality issues where possible.
"""
import json
from pathlib import Path
from PIL import Image
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_ROOT / "backend/app/data/raw/sign_images"
DICT_PATH = PROJECT_ROOT / "backend/app/data/processed/gsl_dictionary.json"
QA_REPORT_PATH = PROJECT_ROOT / "backend/app/data/processed/qa_report.json"

def fix_orphaned_images():
    """Link orphaned images to dictionary entries by page number."""
    print("Fixing orphaned images...")
    
    # Load dictionary
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    
    # Get actual image files
    image_files = {f.name: f for f in list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpeg")) + list(IMAGES_DIR.glob("*.jpg"))}
    
    # Get images that aren't referenced
    all_references = set()
    for entry in dictionary:
        all_references.update(entry.get('image_refs', []))
    
    orphaned = [name for name in image_files.keys() if name not in all_references]
    
    # Group entries by page
    entries_by_page = defaultdict(list)
    for entry in dictionary:
        page = entry.get('page')
        if page:
            entries_by_page[page].append(entry)
    
    linked_count = 0
    
    for img_name in orphaned:
        # Extract page number from filename
        try:
            parts = img_name.replace('.png', '').replace('.jpeg', '').replace('.jpg', '').split('_')
            if len(parts) >= 2 and parts[0] == 'page':
                page_num = int(parts[1])
                
                # Find entries on this page that don't have images
                entries_on_page = entries_by_page.get(page_num, [])
                for entry in entries_on_page:
                    if not entry.get('image_refs'):
                        if 'image_refs' not in entry:
                            entry['image_refs'] = []
                        entry['image_refs'].append(img_name)
                        linked_count += 1
                        print(f"  ✓ Linked {img_name} to '{entry.get('sign', 'unknown')}' on page {page_num}")
                        break  # Link to first entry without images
        except (ValueError, IndexError):
            continue
    
    if linked_count > 0:
        with open(DICT_PATH, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Linked {linked_count} orphaned images to dictionary entries")
    
    return linked_count


def enhance_low_quality_images():
    """Enhance low-quality images if possible (contrast, sharpness)."""
    print("Enhancing low-quality images...")
    
    if not QA_REPORT_PATH.exists():
        print("  QA report not found. Run dataset_quality_assurance.py first.")
        return 0
    
    with open(QA_REPORT_PATH, 'r') as f:
        report = json.load(f)
    
    # Find low quality images from warnings
    low_quality_files = []
    for warning in report.get('issues', {}).get('warnings', []):
        if 'Low quality:' in warning:
            # Extract filename
            parts = warning.split('Low quality: ')[1].split(' - ')
            if parts:
                low_quality_files.append(parts[0])
    
    enhanced_count = 0
    
    for filename in low_quality_files:
        img_path = IMAGES_DIR / filename
        if not img_path.exists():
            continue
        
        try:
            img = Image.open(img_path)
            
            # Enhance contrast
            from PIL import ImageEnhance
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(img)
            enhanced = enhancer.enhance(1.2)  # Increase contrast by 20%
            
            # Save enhanced version
            enhanced.save(img_path, quality=95)
            enhanced_count += 1
            print(f"  ✓ Enhanced contrast for {filename}")
            
        except Exception as e:
            print(f"  ✗ Failed to enhance {filename}: {e}")
    
    if enhanced_count > 0:
        print(f"\n✓ Enhanced {enhanced_count} images")
    
    return enhanced_count


def remove_truly_orphaned_images(dry_run=True):
    """Remove images that can't be linked to any dictionary entry."""
    print(f"{'[DRY RUN] ' if dry_run else ''}Removing truly orphaned images...")
    
    # Load dictionary
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    
    # Get all referenced images
    all_references = set()
    for entry in dictionary:
        all_references.update(entry.get('image_refs', []))
    
    # Get actual image files
    image_files = {f.name: f for f in list(IMAGES_DIR.glob("*.png")) + list(IMAGES_DIR.glob("*.jpeg")) + list(IMAGES_DIR.glob("*.jpg"))}
    
    # Find truly orphaned (not referenced and can't be linked)
    orphaned = []
    for img_name, img_path in image_files.items():
        if img_name not in all_references:
            # Check if we can link it
            try:
                parts = img_name.replace('.png', '').replace('.jpeg', '').replace('.jpg', '').split('_')
                if len(parts) >= 2 and parts[0] == 'page':
                    page_num = int(parts[1])
                    # Check if page is in dictionary
                    has_entry = any(e.get('page') == page_num for e in dictionary)
                    if not has_entry or page_num > 292 or page_num < 1:
                        orphaned.append((img_name, img_path))
            except:
                orphaned.append((img_name, img_path))
    
    if orphaned:
        print(f"  Found {len(orphaned)} truly orphaned images:")
        for img_name, img_path in orphaned[:10]:
            print(f"    - {img_name}")
        if len(orphaned) > 10:
            print(f"    ... and {len(orphaned) - 10} more")
        
        if not dry_run:
            removed = 0
            for img_name, img_path in orphaned:
                img_path.unlink()
                removed += 1
            print(f"\n✓ Removed {removed} orphaned images")
            return removed
    
    return 0


def generate_quality_improvements_report():
    """Generate a report with actionable improvements."""
    improvements = []
    
    if not QA_REPORT_PATH.exists():
        return improvements
    
    with open(QA_REPORT_PATH, 'r') as f:
        report = json.load(f)
    
    score = report.get('quality_score', 0)
    
    # Specific improvements based on issues
    if score < 90:
        improvements.append({
            'priority': 'high',
            'action': 'Link orphaned images to dictionary entries',
            'command': 'python3 scripts/fix_dataset_issues.py --link-orphaned'
        })
        
        if report.get('metrics', {}).get('data_integrity', {}).get('entries_without_images', 0) > 0:
            improvements.append({
                'priority': 'high',
                'action': 'Extract missing images for dictionary entries',
                'command': 'python3 scripts/extract_gsl_sign_images.py'
            })
        
        if report.get('metrics', {}).get('image_quality', {}).get('low_quality_count', 0) > 0:
            improvements.append({
                'priority': 'medium',
                'action': 'Enhance low-quality images',
                'command': 'python3 scripts/fix_dataset_issues.py --enhance'
            })
        
        improvements.append({
            'priority': 'low',
            'action': 'Re-run QA to verify improvements',
            'command': 'python3 scripts/dataset_quality_assurance.py'
        })
    
    return improvements


if __name__ == "__main__":
    import argparse
    from collections import defaultdict
    
    parser = argparse.ArgumentParser(description='Fix dataset quality issues')
    parser.add_argument('--link-orphaned', action='store_true', help='Link orphaned images to dictionary entries')
    parser.add_argument('--enhance', action='store_true', help='Enhance low-quality images')
    parser.add_argument('--remove-orphaned', action='store_true', help='Remove truly orphaned images (use with caution)')
    parser.add_argument('--all', action='store_true', help='Run all fixes')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Issue Fixer")
    print("=" * 60)
    print()
    
    if args.all or args.link_orphaned:
        fix_orphaned_images()
    
    if args.all or args.enhance:
        enhance_low_quality_images()
    
    if args.remove_orphaned:
        remove_truly_orphaned_images(dry_run=False)
    elif args.all:
        # Only suggest removal, don't do it automatically
        remove_truly_orphaned_images(dry_run=True)
        print("\nNote: Truly orphaned images were not removed automatically.")
        print("Review the list above and remove manually if needed.")
    
    print("\n" + "=" * 60)
    print("Fix complete!")
    print("=" * 60)
    print("\nRecommended next steps:")
    print("  1. Run QA again: python3 scripts/dataset_quality_assurance.py")
    print("  2. Review the updated QA report")
    print("  3. Manually review any remaining issues")

