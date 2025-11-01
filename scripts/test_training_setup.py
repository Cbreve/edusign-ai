#!/usr/bin/env python3
"""
Test Training Setup Script

This script verifies that the training environment is properly configured
and that data is accessible before starting training.

Usage:
    python scripts/test_training_setup.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FRAMES_DIR = PROJECT_ROOT / "backend/app/data/processed/validated_frames"
DICTIONARY_PATH = PROJECT_ROOT / "backend/app/data/processed/gsl_dictionary.json"

def check_dependencies():
    """Check if required dependencies are installed."""
    print("="*60)
    print("Checking Dependencies")
    print("="*60)
    
    issues = []
    
    # Check PyTorch
    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"  CUDA available: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  CUDA not available (will use CPU)")
    except ImportError:
        print("✗ PyTorch: NOT INSTALLED")
        issues.append("Install PyTorch: pip install torch torchvision")
    
    # Check MediaPipe
    try:
        import mediapipe as mp
        print(f"✓ MediaPipe: {mp.__version__}")
    except ImportError:
        print("✗ MediaPipe: NOT INSTALLED")
        issues.append("Install MediaPipe: pip install mediapipe")
    
    # Check OpenCV
    try:
        import cv2
        print(f"✓ OpenCV: {cv2.__version__}")
    except ImportError:
        print("✗ OpenCV: NOT INSTALLED")
        issues.append("Install OpenCV: pip install opencv-python")
    
    # Check NumPy
    try:
        import numpy as np
        print(f"✓ NumPy: {np.__version__}")
    except ImportError:
        print("✗ NumPy: NOT INSTALLED")
        issues.append("Install NumPy: pip install numpy")
    
    print()
    return issues


def check_data():
    """Check if training data is available."""
    print("="*60)
    print("Checking Training Data")
    print("="*60)
    
    issues = []
    
    # Check dictionary
    if DICTIONARY_PATH.exists():
        import json
        with open(DICTIONARY_PATH, 'r', encoding='utf-8') as f:
            dictionary = json.load(f)
        print(f"✓ Dictionary: {len(dictionary)} entries")
        unique_signs = len(set(entry['sign'] for entry in dictionary))
        print(f"  Unique signs: {unique_signs}")
    else:
        print(f"✗ Dictionary: NOT FOUND at {DICTIONARY_PATH}")
        issues.append(f"Dictionary not found: {DICTIONARY_PATH}")
    
    # Check frames
    if FRAMES_DIR.exists():
        frame_files = list(FRAMES_DIR.glob("*.jpg")) + list(FRAMES_DIR.glob("*.jpeg"))
        print(f"✓ Validated frames: {len(frame_files)} files")
        
        if len(frame_files) == 0:
            print("  ⚠ Warning: No frame files found")
            issues.append("No validated frames found. Run frame extraction and validation first.")
    else:
        print(f"✗ Frames directory: NOT FOUND at {FRAMES_DIR}")
        issues.append(f"Frames directory not found: {FRAMES_DIR}")
    
    print()
    return issues


def check_model_structure():
    """Check if model directory structure exists."""
    print("="*60)
    print("Checking Model Structure")
    print("="*60)
    
    models_dir = PROJECT_ROOT / "backend/app/models"
    checkpoints_dir = models_dir / "checkpoints"
    
    if models_dir.exists():
        print(f"✓ Models directory: {models_dir}")
    else:
        print(f"✗ Models directory: NOT FOUND")
        models_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {models_dir}")
    
    if checkpoints_dir.exists():
        print(f"✓ Checkpoints directory: {checkpoints_dir}")
    else:
        print(f"✗ Checkpoints directory: NOT FOUND")
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Created: {checkpoints_dir}")
    
    print()
    return []


def test_mediapipe():
    """Test MediaPipe landmark extraction."""
    print("="*60)
    print("Testing MediaPipe")
    print("="*60)
    
    try:
        import mediapipe as mp
        import cv2
        import numpy as np
        
        # Initialize MediaPipe
        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose
        
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)
        pose = mp_pose.Pose(static_image_mode=True)
        
        # Try to load a test frame
        if FRAMES_DIR.exists():
            frame_files = list(FRAMES_DIR.glob("*.jpg")) + list(FRAMES_DIR.glob("*.jpeg"))
            if frame_files:
                test_frame_path = frame_files[0]
                frame = cv2.imread(str(test_frame_path))
                
                if frame is not None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(rgb_frame)
                    hands_results = hands.process(rgb_frame)
                    
                    landmarks_detected = pose_results.pose_landmarks is not None
                    hands_detected = hands_results.multi_hand_landmarks is not None
                    
                    print(f"✓ Test frame loaded: {test_frame_path.name}")
                    print(f"  Pose landmarks detected: {landmarks_detected}")
                    print(f"  Hand landmarks detected: {hands_detected}")
                else:
                    print(f"✗ Failed to load test frame: {test_frame_path}")
            else:
                print("⚠ No frames available for testing")
        
        hands.close()
        pose.close()
        print("✓ MediaPipe test completed")
        
    except Exception as e:
        print(f"✗ MediaPipe test failed: {e}")
        return [f"MediaPipe test failed: {e}"]
    
    print()
    return []


def main():
    print("\n" + "="*60)
    print("EduSign AI - Training Setup Test")
    print("="*60 + "\n")
    
    all_issues = []
    
    # Run checks
    all_issues.extend(check_dependencies())
    all_issues.extend(check_data())
    all_issues.extend(check_model_structure())
    all_issues.extend(test_mediapipe())
    
    # Summary
    print("="*60)
    print("Summary")
    print("="*60)
    
    if len(all_issues) == 0:
        print("✓ All checks passed! Ready to start training.")
        print("\nTo start training, run:")
        print("  python scripts/train_edusign_gsl.py")
        return 0
    else:
        print(f"✗ Found {len(all_issues)} issue(s):")
        for i, issue in enumerate(all_issues, 1):
            print(f"  {i}. {issue}")
        print("\nPlease fix the issues above before starting training.")
        return 1


if __name__ == "__main__":
    sys.exit(main())

