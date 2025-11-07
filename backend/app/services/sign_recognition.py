"""
Sign Recognition Service for EduSign AI

This service loads the trained GSL sign recognition model and provides
inference capabilities for the FastAPI backend.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
import cv2

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available. Sign recognition will be disabled.")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logging.warning("MediaPipe not available. Sign recognition will be disabled.")

logger = logging.getLogger(__name__)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "backend/app/models/edusign_gsl_finetuned.pth"
DICTIONARY_PATH = PROJECT_ROOT / "backend/app/data/processed/gsl_dictionary.json"


class SignRecognitionService:
    """
    Service for GSL sign recognition using trained model.
    """
    
    def __init__(self):
        self.model = None
        self.extractor = None
        self.dictionary = None
        self.idx_to_sign = {}
        self.sign_to_idx = {}
        self.device = None
        self._initialized = False
    
    def initialize(self):
        """Initialize the sign recognition service."""
        if not TORCH_AVAILABLE or not MEDIAPIPE_AVAILABLE:
            logger.error("Required dependencies not available. Cannot initialize sign recognition.")
            return False
        
        try:
            # Setup device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Load dictionary
            if DICTIONARY_PATH.exists():
                with open(DICTIONARY_PATH, 'r', encoding='utf-8') as f:
                    self.dictionary = json.load(f)
                
                unique_signs = sorted(list(set(entry['sign'] for entry in self.dictionary)))
                self.idx_to_sign = {idx: sign for idx, sign in enumerate(unique_signs)}
                self.sign_to_idx = {sign: idx for idx, sign in self.idx_to_sign.items()}
                logger.info(f"Loaded dictionary with {len(self.idx_to_sign)} unique signs")
            else:
                logger.error(f"Dictionary not found: {DICTIONARY_PATH}")
                return False
            
            # Load model
            if MODEL_PATH.exists():
                # Import model architectures
                import sys
                sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
                from train_edusign_gsl import SimpleI3D, MediaPipeLandmarkExtractor
                
                # Try to import FullI3D
                try:
                    from scripts.i3d_architecture import FullI3D, HybridI3D
                    I3D_AVAILABLE = True
                except ImportError:
                    I3D_AVAILABLE = False
                    FullI3D = None
                    HybridI3D = None
                
                num_classes = len(self.idx_to_sign)
                
                # Load checkpoint to detect architecture
                checkpoint = torch.load(MODEL_PATH, map_location=self.device)
                state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
                
                # Detect architecture from checkpoint keys
                has_conv3d = any('conv3d' in k.lower() or '3d' in k.lower() or 'i3d' in k.lower() for k in state_dict.keys())
                has_lstm = any('lstm' in k.lower() for k in state_dict.keys())
                
                # Determine architecture
                if has_conv3d and I3D_AVAILABLE:
                    # FullI3D or HybridI3D
                    if has_lstm:
                        # HybridI3D
                        logger.info("Detected HybridI3D architecture")
                        self.model = HybridI3D(
                            input_features=225,
                            num_classes=num_classes,
                            base_channels=64
                        )
                    else:
                        # FullI3D
                        logger.info("Detected FullI3D architecture")
                        self.model = FullI3D(
                            input_features=225,
                            num_classes=num_classes,
                            base_channels=64,
                            depth_factor=1.0
                        )
                else:
                    # SimpleI3D (default/fallback)
                    logger.info("Detected SimpleI3D architecture (or I3D not available)")
                    self.model = SimpleI3D(input_features=225, num_classes=num_classes, hidden_dim=512)
                
                # Load weights
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                
                self.model.to(self.device)
                self.model.eval()
                
                # Initialize landmark extractor
                self.extractor = MediaPipeLandmarkExtractor()
                
                self._initialized = True
                logger.info("Sign recognition service initialized successfully")
                return True
            else:
                logger.warning(f"Model not found: {MODEL_PATH}. Service will use placeholder.")
                return False
        
        except Exception as e:
            logger.error(f"Failed to initialize sign recognition service: {e}", exc_info=True)
            return False
    
    def is_initialized(self) -> bool:
        """Check if service is initialized."""
        return self._initialized
    
    def recognize_from_frame(self, frame: np.ndarray, top_k: int = 5) -> List[Dict]:
        """
        Recognize sign from a single frame.
        
        Args:
            frame: Input frame as numpy array (BGR format)
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions with sign, meaning, and confidence
        """
        if not self._initialized:
            return self._placeholder_response()
        
        try:
            # Extract landmarks
            landmarks = self.extractor.extract_landmarks(frame)
            if landmarks is None:
                return [{
                    'sign': 'Unknown',
                    'meaning': 'No landmarks detected',
                    'confidence': 0.0
                }]
            
            # Create sequence by repeating frame
            sequence_length = 16
            landmarks_sequence = np.tile(landmarks, (sequence_length, 1))
            landmarks_tensor = torch.from_numpy(landmarks_sequence.astype(np.float32))
            landmarks_tensor = landmarks_tensor.unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(landmarks_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.idx_to_sign)), dim=1)
            
            predictions = []
            for i in range(top_k):
                idx = top_indices[0][i].item()
                sign = self.idx_to_sign.get(idx, "Unknown")
                confidence = float(top_probs[0][i].item())
                
                # Get meaning from dictionary
                meaning = "No meaning found"
                if self.dictionary:
                    for entry in self.dictionary:
                        if entry['sign'] == sign:
                            meaning = entry.get('meaning', 'No meaning found')
                            break
                
                predictions.append({
                    'sign': sign,
                    'meaning': meaning,
                    'confidence': confidence
                })
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error during recognition: {e}", exc_info=True)
            return [{
                'sign': 'Error',
                'meaning': str(e),
                'confidence': 0.0
            }]
    
    def recognize_from_frames(self, frames: List[np.ndarray], top_k: int = 5) -> List[Dict]:
        """
        Recognize sign from a sequence of frames.
        
        Args:
            frames: List of frames as numpy arrays
            top_k: Number of top predictions to return
            
        Returns:
            List of predictions
        """
        if not self._initialized:
            return self._placeholder_response()
        
        try:
            # Extract landmarks from all frames
            landmarks_sequence = []
            for frame in frames:
                landmarks = self.extractor.extract_landmarks(frame)
                if landmarks is None:
                    landmarks = np.zeros(225)
                landmarks_sequence.append(landmarks)
            
            if len(landmarks_sequence) == 0:
                return [{
                    'sign': 'Unknown',
                    'meaning': 'No valid frames',
                    'confidence': 0.0
                }]
            
            # Pad or truncate to sequence length
            target_length = 16
            if len(landmarks_sequence) < target_length:
                last_frame = landmarks_sequence[-1]
                while len(landmarks_sequence) < target_length:
                    landmarks_sequence.append(last_frame)
            elif len(landmarks_sequence) > target_length:
                indices = np.linspace(0, len(landmarks_sequence) - 1, target_length, dtype=int)
                landmarks_sequence = [landmarks_sequence[i] for i in indices]
            
            # Convert to tensor
            landmarks_array = np.array(landmarks_sequence, dtype=np.float32)
            landmarks_tensor = torch.from_numpy(landmarks_array).unsqueeze(0).to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(landmarks_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                top_probs, top_indices = torch.topk(probabilities, k=min(top_k, len(self.idx_to_sign)), dim=1)
            
            predictions = []
            for i in range(top_k):
                idx = top_indices[0][i].item()
                sign = self.idx_to_sign.get(idx, "Unknown")
                confidence = float(top_probs[0][i].item())
                
                # Get meaning
                meaning = "No meaning found"
                for entry in self.dictionary:
                    if entry['sign'] == sign:
                        meaning = entry.get('meaning', 'No meaning found')
                        break
                
                predictions.append({
                    'sign': sign,
                    'meaning': meaning,
                    'confidence': confidence
                })
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error during recognition: {e}", exc_info=True)
            return [{
                'sign': 'Error',
                'meaning': str(e),
                'confidence': 0.0
            }]
    
    def _placeholder_response(self) -> List[Dict]:
        """Return placeholder response when model is not available."""
        return [{
            'sign': 'Model Not Available',
            'meaning': 'Sign recognition model has not been trained yet. Please train the model first.',
            'confidence': 0.0
        }]


# Global service instance
_sign_recognition_service = None


def get_sign_recognition_service() -> SignRecognitionService:
    """Get or create the global sign recognition service instance."""
    global _sign_recognition_service
    
    if _sign_recognition_service is None:
        _sign_recognition_service = SignRecognitionService()
        _sign_recognition_service.initialize()
    
    return _sign_recognition_service
