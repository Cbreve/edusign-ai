#!/usr/bin/env python3
"""
Validate extracted sign images to identify non-sign images (headers, logos, text, etc.)
Generates a report and optionally removes flagged images.
"""
import os
import json
import io
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_ROOT / "backend/app/data/raw/sign_images"
DICT_PATH = PROJECT_ROOT / "backend/app/data/processed/gsl_dictionary.json"
REPORT_PATH = PROJECT_ROOT / "backend/app/data/processed/image_validation_report.json"

class SignImageValidator:
    """Validates sign images to identify non-sign content."""
    
    def __init__(self):
        self.flags = defaultdict(list)
        self.stats = {
            'total': 0,
            'valid_signs': 0,
            'suspicious': 0,
            'likely_non_signs': 0
        }
    
    def analyze_image(self, image_path: Path) -> dict:
        """Analyze an image and return validation results."""
        try:
            img = Image.open(image_path)
            width, height = img.size
            result = {
                'path': str(image_path.name),
                'width': width,
                'height': height,
                'aspect_ratio': width / height if height > 0 else 0,
                'issues': [],
                'score': 100,  # Start with perfect score, deduct for issues
                'status': 'valid'
            }
            
            # Convert to RGB for analysis
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Issue 1: Extremely wide images (likely headers/banners)
            if width > height * 4:
                result['issues'].append('extremely_wide_banner')
                result['score'] -= 30
            
            # Issue 2: Very small images (likely icons)
            if width < 100 or height < 100:
                result['issues'].append('too_small')
                result['score'] -= 25
            
            # Issue 3: Check for text-heavy content (OCR-like analysis)
            # Sign images typically have diverse colors, text pages are more uniform
            pixels = np.array(img)
            color_variance = np.var(pixels.flatten())
            
            if color_variance < 500:  # Low variance = likely text/blank page
                result['issues'].append('low_color_variance_text_like')
                result['score'] -= 20
            
            # Issue 4: Check for predominantly white/blank areas
            white_pixels = np.sum(np.all(pixels > [240, 240, 240], axis=2))
            white_ratio = white_pixels / (width * height)
            
            if white_ratio > 0.85:  # More than 85% white
                result['issues'].append('mostly_white_blank')
                result['score'] -= 25
            
            # Issue 5: Check for monochrome/grayscale (signs usually have color)
            # Calculate colorfulness
            rgb_mean = np.mean(pixels, axis=(0, 1))
            rgb_std = np.std(pixels, axis=(0, 1))
            colorfulness = np.mean(rgb_std)
            
            if colorfulness < 15:  # Very low color variance = grayscale/text
                result['issues'].append('low_colorfulness_grayscale')
                result['score'] -= 15
            
            # Issue 6: Aspect ratio checks
            # Sign images are typically more square/portrait, not extreme landscape
            if width > height * 3 and width > 1000:
                result['issues'].append('suspicious_wide_format')
                result['score'] -= 10
            
            # Issue 7: Image dimensions that suggest full-page screenshots
            if (width > 2000 and height > 2000) or (width > 2500 or height > 2500):
                result['issues'].append('full_page_screenshot_size')
                result['score'] -= 20
            
            # Determine status based on score
            if result['score'] < 40:
                result['status'] = 'likely_non_sign'
                self.stats['likely_non_signs'] += 1
            elif result['score'] < 70:
                result['status'] = 'suspicious'
                self.stats['suspicious'] += 1
            else:
                result['status'] = 'valid'
                self.stats['valid_signs'] += 1
            
            return result
            
        except Exception as e:
            return {
                'path': str(image_path.name),
                'error': str(e),
                'status': 'error',
                'score': 0
            }
    
    def validate_all(self, images_dir: Path):
        """Validate all images in the directory."""
        print("=" * 60)
        print("Sign Image Validation Tool")
        print("=" * 60)
        print(f"Analyzing images in: {images_dir}")
        print()
        
        image_files = sorted(images_dir.glob("*.png")) + sorted(images_dir.glob("*.jpeg")) + sorted(images_dir.glob("*.jpg"))
        self.stats['total'] = len(image_files)
        
        results = []
        for img_path in image_files:
            print(f"Analyzing: {img_path.name}...", end=' ')
            result = self.analyze_image(img_path)
            results.append(result)
            
            if result.get('status') == 'valid':
                print("✓ Valid")
            elif result.get('status') == 'suspicious':
                print(f"⚠ Suspicious (score: {result['score']:.0f})")
            elif result.get('status') == 'likely_non_sign':
                print(f"✗ Likely non-sign (score: {result['score']:.0f})")
            else:
                print(f"✗ Error: {result.get('error', 'Unknown')}")
            
            # Group by status for summary
            status = result.get('status', 'unknown')
            self.flags[status].append(result['path'])
        
        return results
    
    def generate_report(self, results: list):
        """Generate validation report."""
        report = {
            'summary': {
                'total_images': self.stats['total'],
                'valid_signs': self.stats['valid_signs'],
                'suspicious': self.stats['suspicious'],
                'likely_non_signs': self.stats['likely_non_signs'],
                'validation_rate': {
                    'valid': f"{(self.stats['valid_signs']/self.stats['total']*100):.1f}%",
                    'suspicious': f"{(self.stats['suspicious']/self.stats['total']*100):.1f}%",
                    'likely_non_sign': f"{(self.stats['likely_non_signs']/self.stats['total']*100):.1f}%"
                }
            },
            'by_status': {
                'valid': [r for r in results if r.get('status') == 'valid'],
                'suspicious': [r for r in results if r.get('status') == 'suspicious'],
                'likely_non_sign': [r for r in results if r.get('status') == 'likely_non_sign'],
                'errors': [r for r in results if r.get('status') == 'error']
            },
            'all_results': results
        }
        
        return report
    
    def print_summary(self, report: dict):
        """Print validation summary to console."""
        print("\n" + "=" * 60)
        print("VALIDATION SUMMARY")
        print("=" * 60)
        summary = report['summary']
        print(f"Total images analyzed: {summary['total_images']}")
        print(f"✓ Valid sign images: {summary['valid_signs']} ({summary['validation_rate']['valid']})")
        print(f"⚠ Suspicious (need review): {summary['suspicious']} ({summary['validation_rate']['suspicious']})")
        print(f"✗ Likely non-sign images: {summary['likely_non_signs']} ({summary['validation_rate']['likely_non_sign']})")
        print()
        
        if report['by_status']['likely_non_sign']:
            print("IMAGES FLAGGED AS LIKELY NON-SIGNS:")
            print("-" * 60)
            for img in report['by_status']['likely_non_sign'][:20]:  # Show first 20
                issues = ', '.join(img.get('issues', []))
                print(f"  ✗ {img['path']} (score: {img.get('score', 0):.0f}) - {issues}")
            if len(report['by_status']['likely_non_sign']) > 20:
                print(f"  ... and {len(report['by_status']['likely_non_sign']) - 20} more")
            print()
        
        if report['by_status']['suspicious']:
            print("SUSPICIOUS IMAGES (NEED MANUAL REVIEW):")
            print("-" * 60)
            for img in report['by_status']['suspicious'][:10]:  # Show first 10
                issues = ', '.join(img.get('issues', []))
                print(f"  ⚠ {img['path']} (score: {img.get('score', 0):.0f}) - {issues}")
            if len(report['by_status']['suspicious']) > 10:
                print(f"  ... and {len(report['by_status']['suspicious']) - 10} more")
            print()


def remove_flagged_images(report_path: Path, images_dir: Path, dry_run=True):
    """Remove images flagged as likely non-signs."""
    if not report_path.exists():
        print(f"Error: Report not found at {report_path}")
        return
    
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    flagged = report['by_status']['likely_non_sign']
    
    if not flagged:
        print("No images flagged for removal.")
        return
    
    print(f"\n{'[DRY RUN] ' if dry_run else ''}Removing {len(flagged)} flagged images...")
    
    removed = []
    for img_info in flagged:
        img_path = images_dir / img_info['path']
        if img_path.exists():
            if not dry_run:
                img_path.unlink()
            removed.append(img_info['path'])
            print(f"  {'Would remove' if dry_run else 'Removed'}: {img_info['path']}")
    
    if not dry_run:
        print(f"\n✓ Removed {len(removed)} images")
        # Update dictionary to remove references to deleted images
        update_dictionary_after_cleanup(removed)
    else:
        print(f"\n[DRY RUN] Would remove {len(removed)} images")
        print("Run with --remove flag to actually delete files")


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
    
    print(f"✓ Updated {updated} dictionary entries")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate extracted sign images')
    parser.add_argument('--remove', action='store_true', help='Remove flagged non-sign images (default: dry run)')
    parser.add_argument('--report-only', action='store_true', help='Only generate report, do not suggest removal')
    args = parser.parse_args()
    
    validator = SignImageValidator()
    results = validator.validate_all(IMAGES_DIR)
    report = validator.generate_report(results)
    
    # Save report
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    validator.print_summary(report)
    
    print(f"\n✓ Validation report saved to: {REPORT_PATH}")
    
    # Optionally remove flagged images
    if not args.report_only and report['by_status']['likely_non_sign']:
        remove_flagged_images(REPORT_PATH, IMAGES_DIR, dry_run=not args.remove)
    
    print("\n" + "=" * 60)
    print("Validation complete!")
    print("=" * 60)

