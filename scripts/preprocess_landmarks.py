#!/usr/bin/env python3
"""
Preprocess and cache MediaPipe landmarks for faster training.

This follows industry best practices (like Google/OpenAI) by:
1. Pre-extracting all landmarks once
2. Caching them to disk
3. Loading from cache during training (much faster)

Usage:
    python scripts/preprocess_landmarks.py
"""

import json
import logging
import sys
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import torch

# Import from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_edusign_gsl import MediaPipeLandmarkExtractor

PROJECT_ROOT = Path(__file__).parent.parent
FRAMES_DIR = PROJECT_ROOT / "backend/app/data/processed/validated_frames"
LANDMARKS_CACHE_DIR = PROJECT_ROOT / "backend/app/data/processed/landmarks_cache"
LANDMARKS_METADATA = LANDMARKS_CACHE_DIR / "landmarks_metadata.json"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LANDMARKS_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def preprocess_all_landmarks():
    """Pre-extract and cache all landmarks."""
    logger.info("="*60)
    logger.info("Preprocessing MediaPipe Landmarks")
    logger.info("="*60)
    
    # Initialize extractor
    logger.info("Initializing MediaPipe extractor...")
    extractor = MediaPipeLandmarkExtractor()
    
    # Get all frame files
    frame_files = sorted(list(FRAMES_DIR.glob("*.jpg")) + list(FRAMES_DIR.glob("*.jpeg")))
    logger.info(f"Found {len(frame_files)} frames to process")
    
    if len(frame_files) == 0:
        logger.error(f"No frames found in {FRAMES_DIR}")
        return False
    
    # Process frames
    metadata = {}
    processed = 0
    failed = 0
    
    logger.info("Extracting landmarks (this will take a while)...")
    for frame_path in tqdm(frame_files, desc="Processing frames"):
        try:
            # Load frame
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"Failed to load: {frame_path}")
                failed += 1
                continue
            
            # Extract landmarks
            landmarks = extractor.extract_landmarks(frame)
            
            if landmarks is None:
                # No landmarks detected - use zeros
                landmarks = np.zeros(225, dtype=np.float32)
            
            # Save landmarks to cache
            cache_file = LANDMARKS_CACHE_DIR / f"{frame_path.stem}.npy"
            np.save(cache_file, landmarks)
            
            metadata[str(frame_path.name)] = {
                'cache_file': cache_file.name,
                'has_landmarks': not np.allclose(landmarks, 0),
                'source': str(frame_path)
            }
            
            processed += 1
            
        except Exception as e:
            logger.error(f"Error processing {frame_path}: {e}")
            failed += 1
    
    # Save metadata
    with open(LANDMARKS_METADATA, 'w') as f:
        json.dump({
            'total_frames': len(frame_files),
            'processed': processed,
            'failed': failed,
            'cache_dir': str(LANDMARKS_CACHE_DIR),
            'frames': metadata
        }, f, indent=2)
    
    logger.info("="*60)
    logger.info("Preprocessing Complete!")
    logger.info(f"✓ Processed: {processed}")
    logger.info(f"✗ Failed: {failed}")
    logger.info(f"✓ Cache location: {LANDMARKS_CACHE_DIR}")
    logger.info("="*60)
    
    return processed > 0


if __name__ == "__main__":
    success = preprocess_all_landmarks()
    sys.exit(0 if success else 1)

