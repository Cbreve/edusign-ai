#!/usr/bin/env python3
"""
EduSign AI - GSL Sign Recognition Inference Script

This script loads a trained model and performs inference on video frames or sequences.
Useful for testing the trained model before integrating into the backend.

Usage:
    python scripts/inference_edusign_gsl.py --model backend/app/models/edusign_gsl_finetuned.pth --input path/to/frame.jpg
    python scripts/inference_edusign_gsl.py --model backend/app/models/edusign_gsl_finetuned.pth --video path/to/video.mp4
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import cv2

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Error: PyTorch not installed. Install with: pip install torch")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Error: MediaPipe not installed. Install with: pip install mediapipe")

# Import model and utilities from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_edusign_gsl import (
    MediaPipeLandmarkExtractor,
    SimpleI3D,
    DictionaryConnector
)

PROJECT_ROOT = Path(__file__).parent.parent
DICTIONARY_PATH = PROJECT_ROOT / "backend/app/data/processed/gsl_dictionary.json"
MODELS_DIR = PROJECT_ROOT / "backend/app/models"


class GSLInference:
    """Inference wrapper for trained GSL model."""
    
    def __init__(
        self,
        model_path: Path,
        dictionary_path: Path = DICTIONARY_PATH,
        device: str = 'auto'
    ):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to trained model checkpoint
            dictionary_path: Path to GSL dictionary JSON
            device: Device to run inference on (cuda/cpu/auto)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required")
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required")
        
        # Setup device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        # Load dictionary
        self.dictionary_connector = DictionaryConnector(dictionary_path)
        
        # Load sign mappings
        with open(dictionary_path, 'r', encoding='utf-8') as f:
            dictionary = json.load(f)
        
        unique_signs = sorted(list(set(entry['sign'] for entry in dictionary)))
        self.idx_to_sign = {idx: sign for idx, sign in enumerate(unique_signs)}
        self.sign_to_idx = {sign: idx for idx, sign in self.idx_to_sign.items()}
        self.num_classes = len(unique_signs)
        
        # Initialize model
        self.model = SimpleI3D(
            input_features=225,
            num_classes=self.num_classes,
            hidden_dim=512
        )
        
        # Load trained weights
        if model_path.exists():
            if model_path.suffix == '.pth':
                # Check if it's a checkpoint or just state dict
                checkpoint = torch.load(model_path, map_location=self.device)
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                raise ValueError(f"Unsupported model format: {model_path.suffix}")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize landmark extractor
        self.extractor = MediaPipeLandmarkExtractor()
        
        print(f"Model loaded: {len(self.idx_to_sign)} sign classes")
    
    def predict_frame(self, frame_path: Path, sequence_length: int = 16) -> List[Dict]:
        """
        Predict sign from a single frame (creates sequence by repeating frame).
        
        Args:
            frame_path: Path to input frame image
            sequence_length: Length of sequence (will repeat frame to create sequence)
            
        Returns:
            List of top predictions with sign, meaning, and confidence
        """
        frame = cv2.imread(str(frame_path))
        if frame is None:
            raise ValueError(f"Failed to load frame: {frame_path}")
        
        # Extract landmarks
        landmarks = self.extractor.extract_landmarks(frame)
        if landmarks is None:
            raise ValueError("No landmarks detected in frame")
        
        # Create sequence by repeating the frame
        landmarks_sequence = np.tile(landmarks, (sequence_length, 1))
        landmarks_tensor = torch.from_numpy(landmarks_sequence.astype(np.float32))
        landmarks_tensor = landmarks_tensor.unsqueeze(0).to(self.device)  # Add batch dimension
        
        # Predict
        with torch.no_grad():
            outputs = self.model(landmarks_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=5, dim=1)
        
        predictions = []
        for i in range(5):
            idx = top_indices[0][i].item()
            sign = self.idx_to_sign.get(idx, "Unknown")
            confidence = top_probs[0][i].item()
            meaning = self.dictionary_connector.get_meaning(sign)
            
            predictions.append({
                'sign': sign,
                'meaning': meaning,
                'confidence': confidence,
                'rank': i + 1
            })
        
        return predictions
    
    def predict_sequence(self, frame_paths: List[Path]) -> List[Dict]:
        """
        Predict sign from a sequence of frames.
        
        Args:
            frame_paths: List of paths to frame images
            
        Returns:
            List of top predictions
        """
        if len(frame_paths) == 0:
            raise ValueError("No frames provided")
        
        # Extract landmarks from all frames
        landmarks_sequence = []
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                print(f"Warning: Failed to load {frame_path}, skipping")
                continue
            
            landmarks = self.extractor.extract_landmarks(frame)
            if landmarks is None:
                # Use zeros if no landmarks detected
                landmarks = np.zeros(225)
            
            landmarks_sequence.append(landmarks)
        
        if len(landmarks_sequence) == 0:
            raise ValueError("No valid frames found")
        
        # Pad or truncate to consistent length
        target_length = 16
        if len(landmarks_sequence) < target_length:
            # Repeat last frame
            last_frame = landmarks_sequence[-1]
            while len(landmarks_sequence) < target_length:
                landmarks_sequence.append(last_frame)
        elif len(landmarks_sequence) > target_length:
            # Take evenly spaced frames
            indices = np.linspace(0, len(landmarks_sequence) - 1, target_length, dtype=int)
            landmarks_sequence = [landmarks_sequence[i] for i in indices]
        
        # Convert to tensor
        landmarks_array = np.array(landmarks_sequence, dtype=np.float32)
        landmarks_tensor = torch.from_numpy(landmarks_array).unsqueeze(0).to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(landmarks_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, k=5, dim=1)
        
        predictions = []
        for i in range(5):
            idx = top_indices[0][i].item()
            sign = self.idx_to_sign.get(idx, "Unknown")
            confidence = top_probs[0][i].item()
            meaning = self.dictionary_connector.get_meaning(sign)
            
            predictions.append({
                'sign': sign,
                'meaning': meaning,
                'confidence': confidence,
                'rank': i + 1
            })
        
        return predictions
    
    def predict_video(self, video_path: Path, fps: float = 1.5, max_frames: int = 50) -> Dict:
        """
        Extract frames from video and predict sign.
        
        Args:
            video_path: Path to input video
            fps: Frames per second to extract
            max_frames: Maximum number of frames to extract
            
        Returns:
            Dictionary with predictions and metadata
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        fps_video = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps_video / fps)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        frame_paths = []
        frame_count = 0
        extracted_count = 0
        
        # Create temp directory for frames
        temp_dir = PROJECT_ROOT / "temp_frames"
        temp_dir.mkdir(exist_ok=True)
        
        try:
            while extracted_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    temp_frame_path = temp_dir / f"temp_frame_{extracted_count:05d}.jpg"
                    cv2.imwrite(str(temp_frame_path), frame)
                    frame_paths.append(temp_frame_path)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            
            # Predict from extracted frames
            if len(frame_paths) > 0:
                predictions = self.predict_sequence(frame_paths)
                
                # Cleanup temp files
                for temp_path in frame_paths:
                    temp_path.unlink()
                temp_dir.rmdir()
                
                return {
                    'predictions': predictions,
                    'frames_extracted': len(frame_paths),
                    'video_fps': fps_video,
                    'total_video_frames': total_frames
                }
            else:
                raise ValueError("No frames extracted from video")
        
        except Exception as e:
            # Cleanup on error
            for temp_path in frame_paths:
                if temp_path.exists():
                    temp_path.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()
            raise e


def main():
    parser = argparse.ArgumentParser(description='GSL Sign Recognition Inference')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input', type=str, help='Path to input image or video')
    parser.add_argument('--video', type=str, help='Path to input video file')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--top-k', type=int, default=5, help='Number of top predictions to show')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    
    print(f"Loading model from: {model_path}")
    inference = GSLInference(model_path, device=args.device)
    
    # Run inference
    if args.video:
        video_path = Path(args.video)
        if not video_path.is_absolute():
            video_path = PROJECT_ROOT / video_path
        
        print(f"\nProcessing video: {video_path}")
        result = inference.predict_video(video_path)
        
        print("\n" + "="*60)
        print("PREDICTIONS")
        print("="*60)
        for pred in result['predictions'][:args.top_k]:
            print(f"\nRank {pred['rank']}: {pred['sign']}")
            print(f"  Meaning: {pred['meaning']}")
            print(f"  Confidence: {pred['confidence']:.2%}")
        print("\n" + "="*60)
        
    elif args.input:
        input_path = Path(args.input)
        if not input_path.is_absolute():
            input_path = PROJECT_ROOT / input_path
        
        if input_path.is_file():
            if input_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                print(f"\nProcessing frame: {input_path}")
                predictions = inference.predict_frame(input_path)
            else:
                print(f"\nProcessing video: {input_path}")
                result = inference.predict_video(input_path)
                predictions = result['predictions']
        else:
            # Assume it's a directory with frames
            frame_files = sorted(list(input_path.glob("*.jpg")) + list(input_path.glob("*.jpeg")))
            print(f"\nProcessing {len(frame_files)} frames from: {input_path}")
            predictions = inference.predict_sequence(frame_files[:16])  # Limit to 16 frames
        
        print("\n" + "="*60)
        print("PREDICTIONS")
        print("="*60)
        for pred in predictions[:args.top_k]:
            print(f"\nRank {pred['rank']}: {pred['sign']}")
            print(f"  Meaning: {pred['meaning']}")
            print(f"  Confidence: {pred['confidence']:.2%}")
        print("\n" + "="*60)
    
    else:
        print("Error: Provide either --input or --video")
        parser.print_help()


if __name__ == "__main__":
    main()

