#!/usr/bin/env python3
"""
Batch test script for the trained GSL model.

Tests multiple frames and provides statistics.
"""

import sys
from pathlib import Path
from inference_edusign_gsl import GSLInference
import random

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_PATH = PROJECT_ROOT / "backend/app/models/edusign_gsl_finetuned.pth"
FRAMES_DIR = PROJECT_ROOT / "backend/app/data/processed/validated_frames"

def batch_test(num_samples: int = 10):
    """Test model on random sample of frames."""
    print("="*60)
    print("BATCH TESTING GSL MODEL")
    print("="*60)
    print()
    
    # Load model
    print("Loading model...")
    inference = GSLInference(MODEL_PATH)
    print("✓ Model loaded\n")
    
    # Get random sample of frames
    all_frames = list(FRAMES_DIR.glob("*.jpg"))
    if not all_frames:
        print("No frames found!")
        return
    
    test_frames = random.sample(all_frames, min(num_samples, len(all_frames)))
    
    print(f"Testing {len(test_frames)} random frames...\n")
    
    results = []
    for i, frame_path in enumerate(test_frames, 1):
        print(f"[{i}/{len(test_frames)}] Testing: {frame_path.name}")
        
        try:
            predictions = inference.predict_frame(frame_path, sequence_length=16)
            top_pred = predictions[0] if predictions else None
            
            if top_pred:
                results.append({
                    'frame': frame_path.name,
                    'sign': top_pred['sign'],
                    'confidence': top_pred['confidence'],
                    'meaning': top_pred['meaning']
                })
                print(f"  → {top_pred['sign']} ({top_pred['confidence']:.2f}%)")
            else:
                print("  → No prediction")
                results.append({
                    'frame': frame_path.name,
                    'sign': None,
                    'confidence': 0.0
                })
        except Exception as e:
            print(f"  → Error: {e}")
            results.append({
                'frame': frame_path.name,
                'sign': None,
                'confidence': 0.0,
                'error': str(e)
            })
        print()
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    successful = [r for r in results if r.get('sign')]
    avg_confidence = sum(r['confidence'] for r in successful) / len(successful) if successful else 0
    
    print(f"Tested: {len(results)} frames")
    print(f"Successful predictions: {len(successful)}/{len(results)} ({100*len(successful)/len(results):.1f}%)")
    print(f"Average confidence: {avg_confidence:.2f}%")
    print()
    
    print("Top predictions:")
    for r in results[:5]:
        if r.get('sign'):
            print(f"  • {r['sign']} ({r['confidence']:.2f}%) - {r['frame']}")
    
    return results

if __name__ == "__main__":
    num_samples = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    batch_test(num_samples)

