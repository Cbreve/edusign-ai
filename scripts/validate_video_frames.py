#!/usr/bin/env python3
"""
Validate extracted video frames for sign language dataset quality.
Ensures frames meet training quality standards.
"""
import json
import cv2
import numpy as np
from pathlib import Path
from collections import defaultdict
from PIL import Image

PROJECT_ROOT = Path(__file__).parent.parent
FRAMES_DIR = PROJECT_ROOT / "backend/app/data/raw/video_frames"
VALIDATED_DIR = PROJECT_ROOT / "backend/app/data/processed/validated_frames"
REJECTED_DIR = PROJECT_ROOT / "backend/app/data/raw/rejected_frames"
REPORT_PATH = PROJECT_ROOT / "backend/app/data/processed/frame_validation_report.json"

VALIDATED_DIR.mkdir(parents=True, exist_ok=True)
REJECTED_DIR.mkdir(parents=True, exist_ok=True)

class FrameValidator:
    """Validate frames for sign language training."""
    
    def __init__(self):
        self.stats = {
            'total': 0,
            'valid': 0,
            'rejected': 0,
            'issues': defaultdict(int)
        }
        self.rejected_frames = []
        self.valid_frames = []
    
    def validate_frame(self, frame_path: Path) -> dict:
        """
        Comprehensive frame validation.
        Returns validation result with score and issues.
        """
        result = {
            'frame': frame_path.name,
            'score': 100,
            'issues': [],
            'status': 'valid',
            'metrics': {}
        }
        
        try:
            img = cv2.imread(str(frame_path))
            if img is None:
                result['status'] = 'error'
                result['issues'].append('cannot_read')
                return result
            
            height, width = img.shape[:2]
            result['metrics']['resolution'] = f"{width}x{height}"
            result['metrics']['pixels'] = width * height
            
            # 1. Resolution check
            if width < 640 or height < 480:
                result['score'] -= 30
                result['issues'].append('low_resolution')
            elif width < 1280 or height < 720:
                result['score'] -= 10
                result['issues'].append('medium_resolution')
            
            # 2. Blur detection (Laplacian variance)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            result['metrics']['blur_score'] = float(blur_score)
            
            if blur_score < 50:
                result['score'] -= 25
                result['issues'].append('blurry')
            elif blur_score < 100:
                result['score'] -= 10
                result['issues'].append('slightly_blurry')
            
            # 3. Brightness check
            mean_brightness = np.mean(gray)
            result['metrics']['brightness'] = float(mean_brightness)
            
            if mean_brightness < 30:
                result['score'] -= 20
                result['issues'].append('too_dark')
            elif mean_brightness > 220:
                result['score'] -= 15
                result['issues'].append('overexposed')
            
            # 4. Contrast check
            contrast = np.std(gray)
            result['metrics']['contrast'] = float(contrast)
            
            if contrast < 20:
                result['score'] -= 20
                result['issues'].append('low_contrast')
            
            # 5. Aspect ratio (sign language videos are often 16:9 or 4:3)
            aspect_ratio = width / height if height > 0 else 0
            result['metrics']['aspect_ratio'] = float(aspect_ratio)
            
            if aspect_ratio < 0.5 or aspect_ratio > 3.0:
                result['score'] -= 15
                result['issues'].append('extreme_aspect_ratio')
            
            # 6. Check for signs of person in frame (motion detection)
            # Simple heuristic: variance in pixel values (person = more variance)
            pixel_variance = np.var(img)
            result['metrics']['pixel_variance'] = float(pixel_variance)
            
            if pixel_variance < 500:
                result['score'] -= 10
                result['issues'].append('low_variance_possible_no_person')
            
            # Determine status
            if result['score'] < 50:
                result['status'] = 'rejected'
            elif result['score'] < 70:
                result['status'] = 'needs_review'
            else:
                result['status'] = 'valid'
            
            return result
            
        except Exception as e:
            result['status'] = 'error'
            result['issues'].append(f'error: {str(e)}')
            return result
    
    def validate_all(self, frames_dir: Path):
        """Validate all frames in directory."""
        frame_files = list(frames_dir.glob("*.jpg")) + list(frames_dir.glob("*.jpeg")) + list(frames_dir.glob("*.png"))
        
        self.stats['total'] = len(frame_files)
        
        print(f"Validating {len(frame_files)} frames...")
        print()
        
        results = []
        
        for i, frame_path in enumerate(frame_files, 1):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(frame_files)}...")
            
            result = self.validate_frame(frame_path)
            results.append(result)
            
            if result['status'] == 'valid':
                self.stats['valid'] += 1
                self.valid_frames.append(frame_path)
            elif result['status'] == 'rejected':
                self.stats['rejected'] += 1
                self.rejected_frames.append(frame_path)
                for issue in result['issues']:
                    self.stats['issues'][issue] += 1
        
        return results
    
    def organize_frames(self, dry_run=True):
        """Organize frames into validated/rejected directories."""
        if dry_run:
            print("\n[DRY RUN] Would organize frames:")
            print(f"  Valid frames: {len(self.valid_frames)} → {VALIDATED_DIR}")
            print(f"  Rejected frames: {len(self.rejected_frames)} → {REJECTED_DIR}")
            return
        
        # Copy valid frames
        for frame_path in self.valid_frames:
            dest = VALIDATED_DIR / frame_path.name
            import shutil
            shutil.copy2(frame_path, dest)
        
        # Move rejected frames
        for frame_path in self.rejected_frames:
            dest = REJECTED_DIR / frame_path.name
            import shutil
            shutil.move(frame_path, dest)
        
        print(f"\n✓ Organized frames:")
        print(f"  {len(self.valid_frames)} → {VALIDATED_DIR}")
        print(f"  {len(self.rejected_frames)} → {REJECTED_DIR}")


def generate_report(results: list, validator: FrameValidator):
    """Generate validation report."""
    report = {
        'summary': {
            'total_frames': validator.stats['total'],
            'valid': validator.stats['valid'],
            'rejected': validator.stats['rejected'],
            'validation_rate': f"{(validator.stats['valid']/validator.stats['total']*100):.1f}%" if validator.stats['total'] > 0 else "0%"
        },
        'issues_breakdown': dict(validator.stats['issues']),
        'results': results[:100],  # Store first 100 for reference
        'rejected_frames': [r['frame'] for r in results if r['status'] == 'rejected'][:50]
    }
    
    return report


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate video frames for quality')
    parser.add_argument('--organize', action='store_true',
                        help='Organize frames into validated/rejected directories')
    parser.add_argument('--dry-run', action='store_true',
                        help='Dry run (don\'t move files)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Video Frame Validator")
    print("=" * 60)
    print()
    
    validator = FrameValidator()
    results = validator.validate_all(FRAMES_DIR)
    
    report = generate_report(results, validator)
    
    # Save report
    with open(REPORT_PATH, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total frames: {report['summary']['total_frames']}")
    print(f"✓ Valid: {report['summary']['valid']} ({report['summary']['validation_rate']})")
    print(f"✗ Rejected: {report['summary']['rejected']}")
    
    if report['issues_breakdown']:
        print("\nIssues found:")
        for issue, count in sorted(report['issues_breakdown'].items(), key=lambda x: x[1], reverse=True):
            print(f"  - {issue}: {count}")
    
    print(f"\n✓ Report saved to: {REPORT_PATH}")
    
    if args.organize:
        validator.organize_frames(dry_run=args.dry_run)
    
    print("=" * 60)

