#!/usr/bin/env python3
"""
Filter out non-sign images based on validation report.
Can remove images with specific issue patterns.
"""
import json
from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_ROOT / "backend/app/data/raw/sign_images"
REPORT_PATH = PROJECT_ROOT / "backend/app/data/processed/image_validation_report.json"
DICT_PATH = PROJECT_ROOT / "backend/app/data/processed/gsl_dictionary.json"

def filter_images_by_issues(issues_to_remove: list, min_score: int = 50, dry_run=True):
    """
    Remove images that have specific issues.
    
    Args:
        issues_to_remove: List of issue types to filter (e.g., ['low_color_variance_text_like', 'mostly_white_blank'])
        min_score: Minimum score threshold (images below this will be removed)
        dry_run: If True, only show what would be removed
    """
    if not REPORT_PATH.exists():
        print(f"Error: Validation report not found at {REPORT_PATH}")
        print("Run validate_sign_images.py first to generate the report.")
        return
    
    with open(REPORT_PATH, 'r') as f:
        report = json.load(f)
    
    to_remove = []
    
    # Find images with specified issues or low scores
    for result in report['all_results']:
        issues = result.get('issues', [])
        score = result.get('score', 100)
        
        # Check if image has any of the issues to remove
        has_issue = any(issue in issues for issue in issues_to_remove)
        
        # Check if score is below threshold
        below_threshold = score < min_score
        
        if has_issue or below_threshold:
            to_remove.append({
                'path': result['path'],
                'score': score,
                'issues': issues
            })
    
    if not to_remove:
        print("No images found matching the filter criteria.")
        return
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Images to remove: {len(to_remove)}")
    print(f"Criteria: issues={issues_to_remove}, min_score={min_score}")
    print("-" * 60)
    
    removed = []
    for img_info in to_remove[:20]:  # Show first 20
        print(f"  {'Would remove' if dry_run else 'Removing'}: {img_info['path']}")
        print(f"    Score: {img_info['score']:.0f}, Issues: {', '.join(img_info['issues'])}")
        if not dry_run:
            img_path = IMAGES_DIR / img_info['path']
            if img_path.exists():
                img_path.unlink()
                removed.append(img_info['path'])
    if len(to_remove) > 20:
        print(f"  ... and {len(to_remove) - 20} more")
    
    if not dry_run:
        print(f"\n✓ Removed {len(to_remove)} images")
        # Update dictionary
        update_dictionary_after_cleanup([r['path'] for r in to_remove])
        print(f"✓ Updated dictionary references")
    else:
        print(f"\n[DRY RUN] Would remove {len(to_remove)} images")
        print("\nTo actually remove these images, run:")
        print("  python3 scripts/filter_non_sign_images.py --remove")


def update_dictionary_after_cleanup(removed_images: list):
    """Update dictionary JSON to remove references to deleted images."""
    if not DICT_PATH.exists():
        return
    
    with open(DICT_PATH, 'r', encoding='utf-8') as f:
        dictionary = json.load(f)
    
    updated = 0
    for entry in dictionary:
        if 'image_refs' in entry and entry['image_refs']:
            original_refs = entry['image_refs']
            entry['image_refs'] = [ref for ref in original_refs if ref not in removed_images]
            if len(entry['image_refs']) != len(original_refs):
                updated += 1
    
    with open(DICT_PATH, 'w', encoding='utf-8') as f:
        json.dump(dictionary, f, indent=2, ensure_ascii=False)
    
    return updated


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Filter non-sign images based on validation report')
    parser.add_argument('--remove', action='store_true', help='Actually remove the images (default: dry run)')
    parser.add_argument('--issues', nargs='+', 
                        choices=['low_color_variance_text_like', 'mostly_white_blank', 
                                'low_colorfulness_grayscale', 'full_page_screenshot_size',
                                'extremely_wide_banner', 'too_small', 'suspicious_wide_format'],
                        default=['low_color_variance_text_like', 'mostly_white_blank', 'low_colorfulness_grayscale'],
                        help='Issue types to filter')
    parser.add_argument('--min-score', type=int, default=50,
                        help='Minimum score threshold (default: 50)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Non-Sign Image Filter")
    print("=" * 60)
    
    filter_images_by_issues(args.issues, args.min_score, dry_run=not args.remove)

