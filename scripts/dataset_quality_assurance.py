#!/usr/bin/env python3
"""
Comprehensive Dataset Quality Assurance Tool
Checks image quality, data integrity, consistency, and completeness.
"""
import json
import io
from pathlib import Path
from PIL import Image
import numpy as np
from collections import defaultdict
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR = PROJECT_ROOT / "backend/app/data/raw/sign_images"
DICT_PATH = PROJECT_ROOT / "backend/app/data/processed/gsl_dictionary.json"
QA_REPORT_PATH = PROJECT_ROOT / "backend/app/data/processed/qa_report.json"

class DatasetQA:
    """Quality Assurance checker for sign language dataset."""
    
    def __init__(self):
        self.issues = defaultdict(list)
        self.metrics = {}
        self.recommendations = []
    
    def check_image_quality(self):
        """Check quality metrics for all images."""
        print("Checking image quality...")
        
        image_files = sorted(IMAGES_DIR.glob("*.png")) + sorted(IMAGES_DIR.glob("*.jpeg")) + sorted(IMAGES_DIR.glob("*.jpg"))
        
        resolutions = []
        formats = defaultdict(int)
        corrupted = []
        low_quality = []
        
        min_resolution = float('inf')
        max_resolution = 0
        
        for img_path in image_files:
            try:
                img = Image.open(img_path)
                width, height = img.size
                total_pixels = width * height
                
                resolutions.append(total_pixels)
                formats[img.format or 'unknown'] += 1
                
                min_resolution = min(min_resolution, total_pixels)
                max_resolution = max(max_resolution, total_pixels)
                
                # Check for corruption
                img.verify()
                img.close()
                
                # Quality checks
                img = Image.open(img_path)  # Reopen after verify
                
                # Check 1: Minimum resolution (signs need detail)
                if total_pixels < 30000:  # Less than ~173x173px
                    low_quality.append({
                        'file': img_path.name,
                        'width': width,
                        'height': height,
                        'pixels': total_pixels,
                        'issue': 'low_resolution'
                    })
                
                # Check 2: Very low contrast (might be blurry)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                pixels = np.array(img)
                contrast = np.std(pixels)
                if contrast < 10:
                    low_quality.append({
                        'file': img_path.name,
                        'issue': 'low_contrast',
                        'contrast': float(contrast)
                    })
                
                # Check 3: Aspect ratio (very extreme ratios might be cropped incorrectly)
                aspect_ratio = width / height if height > 0 else 0
                if aspect_ratio > 5 or aspect_ratio < 0.2:
                    low_quality.append({
                        'file': img_path.name,
                        'issue': 'extreme_aspect_ratio',
                        'ratio': aspect_ratio
                    })
                
                img.close()
                
            except Exception as e:
                corrupted.append({
                    'file': img_path.name,
                    'error': str(e)
                })
        
        self.metrics['image_quality'] = {
            'total_images': len(image_files),
            'corrupted': len(corrupted),
            'low_quality_count': len(low_quality),
            'min_resolution': int(min_resolution) if image_files else 0,
            'max_resolution': int(max_resolution),
            'avg_resolution': int(np.mean(resolutions)) if resolutions else 0,
            'formats': dict(formats)
        }
        
        if corrupted:
            self.issues['critical'].extend([f"Corrupted image: {c['file']}" for c in corrupted])
        
        if low_quality:
            self.issues['warnings'].extend([f"Low quality: {q['file']} - {q['issue']}" for q in low_quality[:10]])
            if len(low_quality) > 10:
                self.issues['warnings'].append(f"... and {len(low_quality) - 10} more low quality images")
        
        # Recommendations
        if len(corrupted) > 0:
            self.recommendations.append("Remove corrupted images and re-extract from PDF")
        
        if any(q['issue'] == 'low_resolution' for q in low_quality):
            self.recommendations.append("Consider re-extracting low-resolution images with higher DPI")
        
        avg_pixels = np.mean(resolutions) if resolutions else 0
        if avg_pixels < 50000:
            self.recommendations.append("Average image resolution is low - consider extracting at higher quality")
    
    def check_data_integrity(self):
        """Check data consistency between images and dictionary."""
        print("Checking data integrity...")
        
        # Load dictionary
        with open(DICT_PATH, 'r', encoding='utf-8') as f:
            dictionary = json.load(f)
        
        # Get actual image files
        image_files = set()
        for ext in ['*.png', '*.jpeg', '*.jpg']:
            image_files.update(f.name for f in IMAGES_DIR.glob(ext))
        
        # Check dictionary entries
        entries_with_images = 0
        entries_without_images = 0
        images_without_entries = []
        orphaned_images = set(image_files)
        
        page_coverage = defaultdict(int)
        
        for entry in dictionary:
            page = entry.get('page')
            if page:
                page_coverage[page] += 1
            
            image_refs = entry.get('image_refs', [])
            if image_refs:
                entries_with_images += 1
                for ref in image_refs:
                    if ref in image_files:
                        orphaned_images.discard(ref)
                    else:
                        self.issues['warnings'].append(f"Entry '{entry.get('sign', 'unknown')}' references missing image: {ref}")
            else:
                entries_without_images += 1
                if page and page <= 292:
                    self.issues['warnings'].append(f"Entry '{entry.get('sign', 'unknown')}' (page {page}) has no image references")
        
        # Find orphaned images
        for img_file in orphaned_images:
            # Extract page number from filename
            try:
                page_num = int(img_file.split('_')[1])
                if page_num <= 292:
                    images_without_entries.append(img_file)
            except:
                images_without_entries.append(img_file)
        
        self.metrics['data_integrity'] = {
            'total_entries': len(dictionary),
            'entries_with_images': entries_with_images,
            'entries_without_images': entries_without_images,
            'orphaned_images': len(images_without_entries),
            'coverage_percentage': (entries_with_images / len(dictionary) * 100) if dictionary else 0,
            'pages_covered': len(page_coverage),
            'pages_in_range_1_292': len([p for p in page_coverage.keys() if 1 <= p <= 292])
        }
        
        if entries_without_images > len(dictionary) * 0.1:  # More than 10% missing
            self.issues['critical'].append(f"{entries_without_images} entries ({entries_without_images/len(dictionary)*100:.1f}%) have no image references")
        
        if images_without_entries:
            self.issues['warnings'].extend([f"Orphaned image (no dictionary entry): {img}" for img in images_without_entries[:10]])
            if len(images_without_entries) > 10:
                self.issues['warnings'].append(f"... and {len(images_without_entries) - 10} more orphaned images")
        
        # Recommendations
        if entries_without_images > 0:
            self.recommendations.append(f"Add image references for {entries_without_images} entries missing images")
        
        if images_without_entries:
            self.recommendations.append("Link orphaned images to dictionary entries or remove them")
    
    def check_dictionary_quality(self):
        """Check quality of dictionary metadata."""
        print("Checking dictionary quality...")
        
        with open(DICT_PATH, 'r', encoding='utf-8') as f:
            dictionary = json.load(f)
        
        missing_signs = 0
        missing_meanings = 0
        missing_pages = 0
        missing_ids = 0
        duplicate_ids = set()
        seen_ids = set()
        
        page_range_issues = []
        
        for entry in dictionary:
            # Check required fields
            if not entry.get('sign'):
                missing_signs += 1
            if not entry.get('meaning'):
                missing_meanings += 1
            if not entry.get('page'):
                missing_pages += 1
            if not entry.get('id'):
                missing_ids += 1
            
            # Check for duplicate IDs
            entry_id = entry.get('id')
            if entry_id:
                if entry_id in seen_ids:
                    duplicate_ids.add(entry_id)
                seen_ids.add(entry_id)
            
            # Check page range (should be 1-292 based on PDF)
            page = entry.get('page')
            if page and (page < 1 or page > 292):
                page_range_issues.append({
                    'sign': entry.get('sign', 'unknown'),
                    'page': page
                })
        
        self.metrics['dictionary_quality'] = {
            'total_entries': len(dictionary),
            'missing_signs': missing_signs,
            'missing_meanings': missing_meanings,
            'missing_pages': missing_pages,
            'missing_ids': missing_ids,
            'duplicate_ids': len(duplicate_ids),
            'page_range_issues': len(page_range_issues)
        }
        
        if missing_signs > 0:
            self.issues['critical'].append(f"{missing_signs} entries missing 'sign' field")
        if missing_meanings > 0:
            self.issues['warnings'].append(f"{missing_meanings} entries missing 'meaning' field")
        if missing_pages > 0:
            self.issues['warnings'].append(f"{missing_pages} entries missing 'page' field")
        if duplicate_ids:
            self.issues['critical'].append(f"{len(duplicate_ids)} duplicate IDs found")
        if page_range_issues:
            self.issues['warnings'].extend([f"Page out of range: {p['sign']} on page {p['page']}" for p in page_range_issues[:5]])
        
        # Recommendations
        if missing_signs or missing_meanings:
            self.recommendations.append("Fill in missing sign names and meanings for complete dataset")
        if duplicate_ids:
            self.recommendations.append("Fix duplicate IDs - each entry should have a unique ID")
    
    def check_naming_consistency(self):
        """Check image file naming consistency."""
        print("Checking naming consistency...")
        
        image_files = sorted(IMAGES_DIR.glob("*.png")) + sorted(IMAGES_DIR.glob("*.jpeg")) + sorted(IMAGES_DIR.glob("*.jpg"))
        
        naming_issues = []
        valid_pattern = 0
        
        for img_path in image_files:
            name = img_path.name
            # Expected pattern: page_{number}_img_{number}.{ext}
            parts = name.replace('.png', '').replace('.jpeg', '').replace('.jpg', '').split('_')
            
            if len(parts) >= 3 and parts[0] == 'page' and parts[2] == 'img':
                try:
                    page_num = int(parts[1])
                    img_num = int(parts[3]) if len(parts) > 3 else 1
                    if 1 <= page_num <= 292:
                        valid_pattern += 1
                        continue
                except ValueError:
                    pass
            
            naming_issues.append(name)
        
        self.metrics['naming_consistency'] = {
            'total_files': len(image_files),
            'valid_naming': valid_pattern,
            'naming_issues': len(naming_issues),
            'consistency_percentage': (valid_pattern / len(image_files) * 100) if image_files else 0
        }
        
        if naming_issues:
            self.issues['warnings'].extend([f"Inconsistent naming: {name}" for name in naming_issues[:10]])
            if len(naming_issues) > 10:
                self.issues['warnings'].append(f"... and {len(naming_issues) - 10} more naming issues")
    
    def generate_quality_score(self):
        """Generate overall quality score (0-100)."""
        score = 100
        
        # Deduct for critical issues
        score -= len(self.issues.get('critical', [])) * 10
        score -= len(self.issues.get('warnings', [])) * 2
        
        # Deduct for quality metrics
        if self.metrics.get('image_quality', {}).get('corrupted', 0) > 0:
            score -= 20
        
        coverage = self.metrics.get('data_integrity', {}).get('coverage_percentage', 0)
        score -= max(0, (100 - coverage) / 2)  # Deduct up to 50 points for low coverage
        
        # Ensure score is between 0-100
        score = max(0, min(100, score))
        
        return score
    
    def run_all_checks(self):
        """Run all quality assurance checks."""
        print("=" * 60)
        print("Dataset Quality Assurance")
        print("=" * 60)
        print()
        
        self.check_image_quality()
        self.check_data_integrity()
        self.check_dictionary_quality()
        self.check_naming_consistency()
        
        quality_score = self.generate_quality_score()
        
        return quality_score
    
    def generate_report(self):
        """Generate comprehensive QA report."""
        quality_score = self.run_all_checks()
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'quality_score': quality_score,
            'quality_rating': self._get_rating(quality_score),
            'metrics': self.metrics,
            'issues': dict(self.issues),
            'recommendations': self.recommendations,
            'summary': self._generate_summary()
        }
        
        return report
    
    def _get_rating(self, score):
        """Get quality rating from score."""
        if score >= 90:
            return "Excellent"
        elif score >= 75:
            return "Good"
        elif score >= 60:
            return "Fair"
        elif score >= 40:
            return "Poor"
        else:
            return "Critical"
    
    def _generate_summary(self):
        """Generate human-readable summary."""
        summary = []
        summary.append(f"Overall Quality Score: {self.generate_quality_score()}/100")
        summary.append(f"Rating: {self._get_rating(self.generate_quality_score())}")
        summary.append("")
        summary.append("Key Metrics:")
        if 'image_quality' in self.metrics:
            iq = self.metrics['image_quality']
            summary.append(f"  - Images: {iq.get('total_images', 0)} total, {iq.get('corrupted', 0)} corrupted")
        if 'data_integrity' in self.metrics:
            di = self.metrics['data_integrity']
            summary.append(f"  - Dictionary coverage: {di.get('coverage_percentage', 0):.1f}%")
            summary.append(f"  - Pages covered: {di.get('pages_covered', 0)}")
        if 'dictionary_quality' in self.metrics:
            dq = self.metrics['dictionary_quality']
            summary.append(f"  - Dictionary entries: {dq.get('total_entries', 0)} total")
        summary.append("")
        summary.append(f"Issues: {len(self.issues.get('critical', []))} critical, {len(self.issues.get('warnings', []))} warnings")
        
        return "\n".join(summary)


def print_report(report):
    """Print formatted QA report to console."""
    print("\n" + "=" * 60)
    print("QUALITY ASSURANCE REPORT")
    print("=" * 60)
    print()
    print(report['summary'])
    print()
    
    if report['issues'].get('critical'):
        print("CRITICAL ISSUES:")
        print("-" * 60)
        for issue in report['issues']['critical'][:20]:
            print(f"  ✗ {issue}")
        if len(report['issues']['critical']) > 20:
            print(f"  ... and {len(report['issues']['critical']) - 20} more")
        print()
    
    if report['issues'].get('warnings'):
        print("WARNINGS:")
        print("-" * 60)
        for warning in report['issues']['warnings'][:15]:
            print(f"  ⚠ {warning}")
        if len(report['issues']['warnings']) > 15:
            print(f"  ... and {len(report['issues']['warnings']) - 15} more")
        print()
    
    if report['recommendations']:
        print("RECOMMENDATIONS:")
        print("-" * 60)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        print()


if __name__ == "__main__":
    qa = DatasetQA()
    report = qa.generate_report()
    
    # Save report
    QA_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(QA_REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print_report(report)
    
    print("=" * 60)
    print(f"✓ QA Report saved to: {QA_REPORT_PATH}")
    print("=" * 60)

