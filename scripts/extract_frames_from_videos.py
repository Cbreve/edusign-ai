#!/usr/bin/env python3
"""
Extract frames from YouTube videos for sign language dataset.
Extracts high-quality frames with quality validation.
"""
import json
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

PROJECT_ROOT = Path(__file__).parent.parent
VIDEOS_DIR = PROJECT_ROOT / "backend/app/data/raw/youtube_videos"
FRAMES_DIR = PROJECT_ROOT / "backend/app/data/raw/video_frames"
METADATA_DIR = PROJECT_ROOT / "backend/app/data/raw/youtube_metadata"

FRAMES_DIR.mkdir(parents=True, exist_ok=True)

class FrameExtractor:
    """Extract and validate frames from videos."""
    
    def __init__(self, min_fps: float = 1.0, max_fps: float = 2.0):
        """
        Args:
            min_fps: Minimum frames per second to extract
            max_fps: Maximum frames per second to extract
        """
        self.min_fps = min_fps
        self.max_fps = max_fps
        self.stats = {
            'videos_processed': 0,
            'frames_extracted': 0,
            'frames_rejected': 0,
            'videos_failed': 0
        }
    
    def is_quality_frame(self, frame: np.ndarray) -> tuple[bool, str]:
        """
        Validate frame quality.
        Returns: (is_valid, reason)
        """
        if frame is None or frame.size == 0:
            return False, "empty_frame"
        
        height, width = frame.shape[:2]
        
        # Check minimum resolution
        if width < 320 or height < 240:
            return False, "too_small"
        
        # Check for blur (Laplacian variance)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if blur_score < 50:  # Threshold for blur detection
            return False, "blurry"
        
        # Check for black/blank frames
        mean_brightness = np.mean(gray)
        if mean_brightness < 10:  # Very dark
            return False, "too_dark"
        if mean_brightness > 250:  # Overexposed
            return False, "overexposed"
        
        # Check contrast (useful for sign detection)
        contrast = np.std(gray)
        if contrast < 15:  # Low contrast
            return False, "low_contrast"
        
        return True, "valid"
    
    def extract_frames(self, video_path: Path, output_dir: Path, 
                      target_fps: float = 1.5, max_frames: int = None) -> list:
        """
        Extract frames from video.
        
        Args:
            video_path: Path to video file
            output_dir: Directory to save frames
            target_fps: Target frames per second to extract
            max_frames: Maximum frames to extract (None = no limit)
        
        Returns:
            List of extracted frame filenames
        """
        if not video_path.exists():
            return []
        
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Calculate frame interval
        frame_interval = int(fps / target_fps) if fps > target_fps else 1
        
        output_dir.mkdir(parents=True, exist_ok=True)
        video_name = video_path.stem
        
        extracted_frames = []
        frame_count = 0
        saved_count = 0
        
        print(f"  Processing: {video_name}")
        print(f"    FPS: {fps:.2f}, Duration: {duration:.1f}s, Total frames: {total_frames}")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract every Nth frame based on target FPS
            if frame_count % frame_interval == 0:
                # Validate frame quality
                is_valid, reason = self.is_quality_frame(frame)
                
                if is_valid:
                    # Save frame
                    frame_filename = f"{video_name}_frame_{saved_count:06d}.jpg"
                    frame_path = output_dir / frame_filename
                    
                    # Save with good quality
                    cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    extracted_frames.append(frame_filename)
                    saved_count += 1
                    
                    if max_frames and saved_count >= max_frames:
                        break
                else:
                    self.stats['frames_rejected'] += 1
            
            frame_count += 1
        
        cap.release()
        
        self.stats['frames_extracted'] += saved_count
        
        print(f"    ✓ Extracted {saved_count} quality frames")
        if self.stats['frames_rejected'] > 0:
            print(f"    ✗ Rejected {self.stats['frames_rejected']} low-quality frames")
        
        return extracted_frames
    
    def process_all_videos(self, videos_dir: Path, target_fps: float = 1.5, 
                          max_frames_per_video: int = None):
        """Process all videos in directory."""
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(videos_dir.glob(f'*{ext}'))
        
        if not video_files:
            print(f"No videos found in {videos_dir}")
            return {}
        
        print(f"Found {len(video_files)} videos")
        print()
        
        video_metadata = {}
        
        for i, video_path in enumerate(video_files, 1):
            print(f"[{i}/{len(video_files)}] {video_path.name}")
            
            frames = self.extract_frames(video_path, FRAMES_DIR, target_fps, max_frames_per_video)
            
            if frames:
                video_metadata[video_path.stem] = {
                    'video_file': video_path.name,
                    'frames': frames,
                    'frame_count': len(frames),
                    'extracted_date': datetime.now().isoformat()
                }
                self.stats['videos_processed'] += 1
            else:
                self.stats['videos_failed'] += 1
        
        return video_metadata
    
    def save_metadata(self, metadata: dict):
        """Save frame extraction metadata."""
        metadata_file = FRAMES_DIR / 'frame_metadata.json'
        
        with open(metadata_file, 'w') as f:
            json.dump({
                'extraction_date': datetime.now().isoformat(),
                'stats': self.stats,
                'videos': metadata
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Metadata saved to {metadata_file}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract frames from YouTube videos')
    parser.add_argument('--fps', type=float, default=1.5,
                        help='Target frames per second to extract (default: 1.5)')
    parser.add_argument('--max-frames', type=int,
                        help='Maximum frames per video (default: no limit)')
    parser.add_argument('--video', type=str,
                        help='Process single video file (optional)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Video Frame Extractor")
    print("=" * 60)
    print()
    
    extractor = FrameExtractor()
    
    if args.video:
        # Process single video
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = VIDEOS_DIR / video_path
        
        frames = extractor.extract_frames(video_path, FRAMES_DIR, args.fps, args.max_frames)
        print(f"\n✓ Extracted {len(frames)} frames from {video_path.name}")
    else:
        # Process all videos
        metadata = extractor.process_all_videos(VIDEOS_DIR, args.fps, args.max_frames)
        extractor.save_metadata(metadata)
        
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        print(f"Videos processed: {extractor.stats['videos_processed']}")
        print(f"Frames extracted: {extractor.stats['frames_extracted']}")
        print(f"Frames rejected: {extractor.stats['frames_rejected']}")
        print(f"Videos failed: {extractor.stats['videos_failed']}")
    
    print("=" * 60)

