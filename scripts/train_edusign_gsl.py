#!/usr/bin/env python3
"""
EduSign AI - GSL Sign Recognition Model Training Script

This script fine-tunes a pretrained WLASL (I3D) model on Ghanaian Sign Language (GSL) data.
It extracts MediaPipe landmarks from video frames and trains a sign recognition model.

Usage:
    python scripts/train_edusign_gsl.py --epochs 50 --batch-size 16
    python scripts/train_edusign_gsl.py --resume backend/app/models/checkpoint_epoch_10.pth
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, TYPE_CHECKING
from datetime import datetime
import warnings
from collections import Counter
import random

import numpy as np
import cv2
from PIL import Image

# ML Framework imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    from torchvision import transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not installed. Install with: pip install torch torchvision")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("Warning: MediaPipe not installed. Install with: pip install mediapipe")

# Import our custom modules
LandmarkAugmenter = None
get_augmenter = None
FocalLoss = None
LabelSmoothingCrossEntropy = None
CombinedLoss = None
calculate_class_weights = None
get_loss_function = None
FullI3D = None
HybridI3D = None
get_i3d_model = None

try:
    from scripts.data_augmentation import LandmarkAugmenter, get_augmenter
    from scripts.loss_functions import (
        FocalLoss, LabelSmoothingCrossEntropy, CombinedLoss,
        calculate_class_weights, get_loss_function
    )
    from scripts.i3d_architecture import FullI3D, HybridI3D, get_i3d_model
    I3D_AVAILABLE = True
    AUGMENTATION_AVAILABLE = True
except ImportError as e:
    AUGMENTATION_AVAILABLE = False
    I3D_AVAILABLE = False
    print(f"Warning: Custom modules not found. Using basic training. Error: {e}")
    # Create dummy types for type hints
    from typing import TYPE_CHECKING
    if TYPE_CHECKING:
        from typing import Any
        LandmarkAugmenter = Any

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "backend/app/data"
RAW_VIDEOS_DIR = DATA_DIR / "raw/youtube_videos"
FRAMES_DIR = DATA_DIR / "processed/validated_frames"
LANDMARKS_CACHE_DIR = DATA_DIR / "processed/landmarks_cache"
LANDMARKS_METADATA = LANDMARKS_CACHE_DIR / "landmarks_metadata.json"
DICTIONARY_PATH = DATA_DIR / "processed/gsl_dictionary.json"
MODELS_DIR = PROJECT_ROOT / "backend/app/models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
TRAINED_MODEL_PATH = MODELS_DIR / "edusign_gsl_finetuned.pth"

# Ensure directories exist
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / "training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class MediaPipeLandmarkExtractor:
    """
    Extract pose and hand landmarks from frames using MediaPipe.
    
    MediaPipe provides:
    - Pose landmarks (33 points)
    - Left hand landmarks (21 points)
    - Right hand landmarks (21 points)
    Total: 75 landmarks × (x, y, z) = 225 features per frame
    """
    
    def __init__(self):
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe is required. Install with: pip install mediapipe")
        
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize MediaPipe models
        self.hands = self.mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        self.pose = self.mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5
        )
        
        logger.info("MediaPipe landmark extractor initialized")
    
    def extract_landmarks(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract pose and hand landmarks from a single frame.
        
        Args:
            frame: Input frame as numpy array (BGR format from OpenCV)
            
        Returns:
            Combined landmark features as numpy array (225 features)
            or None if no landmarks detected
        """
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract pose landmarks
        pose_results = self.pose.process(rgb_frame)
        pose_landmarks = np.zeros(33 * 3)  # 33 pose points × (x, y, z)
        
        if pose_results.pose_landmarks:
            for i, landmark in enumerate(pose_results.pose_landmarks.landmark):
                idx = i * 3
                pose_landmarks[idx] = landmark.x
                pose_landmarks[idx + 1] = landmark.y
                pose_landmarks[idx + 2] = landmark.z
        
        # Extract hand landmarks
        hands_results = self.hands.process(rgb_frame)
        
        # Left hand landmarks (21 points × 3)
        left_hand_landmarks = np.zeros(21 * 3)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Determine if left or right hand
                # MediaPipe doesn't directly tell us, so we use x-coordinate heuristic
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                is_left = np.mean(x_coords) < 0.5  # Left side of frame typically < 0.5
                
                for i, landmark in enumerate(hand_landmarks.landmark):
                    idx = i * 3
                    if is_left:
                        left_hand_landmarks[idx] = landmark.x
                        left_hand_landmarks[idx + 1] = landmark.y
                        left_hand_landmarks[idx + 2] = landmark.z
        
        # Right hand landmarks (21 points × 3)
        right_hand_landmarks = np.zeros(21 * 3)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                is_right = np.mean(x_coords) >= 0.5
                
                for i, landmark in enumerate(hand_landmarks.landmark):
                    idx = i * 3
                    if is_right:
                        right_hand_landmarks[idx] = landmark.x
                        right_hand_landmarks[idx + 1] = landmark.y
                        right_hand_landmarks[idx + 2] = landmark.z
        
        # Combine all landmarks
        combined_landmarks = np.concatenate([
            pose_landmarks,      # 99 features
            left_hand_landmarks,  # 63 features
            right_hand_landmarks  # 63 features
        ])  # Total: 225 features
        
        # Check if we have any valid landmarks
        if np.sum(np.abs(combined_landmarks)) < 1e-6:
            return None  # No landmarks detected
        
        return combined_landmarks
    
    def __del__(self):
        """Cleanup MediaPipe resources."""
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'pose'):
            self.pose.close()


class SimpleI3D(nn.Module):
    """
    Improved I3D-like architecture for sign recognition.
    
    Enhanced with better regularization and feature extraction.
    """
    
    def __init__(
        self,
        input_features: int = 225,
        num_classes: int = 100,
        hidden_dim: int = 512,
        dropout_rate: float = 0.3,
        lstm_layers: int = 2,
        use_batch_norm: bool = True
    ):
        super(SimpleI3D, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        self.use_batch_norm = use_batch_norm
        
        # Enhanced feature extraction with batch normalization
        layers = []
        layers.append(nn.Linear(input_features, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        self.feature_extractor = nn.Sequential(*layers)
        
        # Temporal modeling (LSTM for sequence understanding)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout_rate if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, features)
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, features = x.shape
        
        # Extract features for each frame
        # Reshape to (batch * seq_len, features) for linear layers
        x_reshaped = x.view(-1, features)
        features_out = self.feature_extractor(x_reshaped)
        # Reshape back to (batch, seq_len, hidden_dim)
        features_out = features_out.view(batch_size, seq_len, -1)
        
        # Temporal modeling with LSTM
        lstm_out, (h_n, c_n) = self.lstm(features_out)
        
        # Use the last hidden state
        final_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)
        
        # Classification
        logits = self.classifier(final_hidden)
        
        return logits


class GSLDataset(Dataset):
    """
    Dataset for GSL sign recognition with augmentation support.
    
    Loads video frames, extracts MediaPipe landmarks, and pairs them with sign labels.
    Supports data augmentation and oversampling for rare classes.
    """
    
    def __init__(
        self,
        frames_dir: Path,
        dictionary_path: Path,
        sequence_length: int = 16,
        transform: Optional[transforms.Compose] = None,
        extractor: Optional[MediaPipeLandmarkExtractor] = None,
        augmenter=None,  # Optional[LandmarkAugmenter]
        use_augmentation: bool = False,
        oversample_rare: bool = False,
        min_samples_per_class: int = 5
    ):
        self.frames_dir = Path(frames_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        self.use_augmentation = use_augmentation
        self.augmenter = augmenter
        
        # Load dictionary for label mapping
        with open(dictionary_path, 'r', encoding='utf-8') as f:
            self.dictionary = json.load(f)
        
        # Create sign-to-index mapping
        unique_signs = sorted(list(set(entry['sign'] for entry in self.dictionary)))
        self.sign_to_idx = {sign: idx for idx, sign in enumerate(unique_signs)}
        self.idx_to_sign = {idx: sign for sign, idx in self.sign_to_idx.items()}
        self.num_classes = len(unique_signs)
        
        logger.info(f"Loaded dictionary with {len(self.dictionary)} entries")
        logger.info(f"Found {self.num_classes} unique signs")
        
        # Extract landmarks for all frames (cached)
        self.extractor = extractor or MediaPipeLandmarkExtractor()
        
        # Build dataset: group frames by video/sign
        self.samples = self._build_samples()
        
        # Oversample rare classes if requested
        if oversample_rare:
            self.samples = self._oversample_rare_classes(min_samples_per_class)
        
        logger.info(f"Dataset initialized with {len(self.samples)} sequences")
        
        # Log class distribution
        self._log_class_distribution()
    
    def _log_class_distribution(self):
        """Log distribution of classes in dataset."""
        class_counts = Counter(sample['label'] for sample in self.samples)
        logger.info(f"Class distribution:")
        logger.info(f"  Min samples per class: {min(class_counts.values())}")
        logger.info(f"  Max samples per class: {max(class_counts.values())}")
        logger.info(f"  Mean samples per class: {sum(class_counts.values()) / len(class_counts):.1f}")
        rare_count = sum(1 for c in class_counts.values() if c < 5)
        logger.info(f"  Classes with <5 samples: {rare_count}")
    
    def _oversample_rare_classes(self, min_samples: int) -> List[Dict]:
        """
        Oversample rare classes to balance dataset.
        
        Args:
            min_samples: Minimum samples per class after oversampling
            
        Returns:
            Augmented sample list
        """
        class_counts = Counter(sample['label'] for sample in self.samples)
        class_samples = {label: [] for label in class_counts.keys()}
        
        # Group samples by class
        for sample in self.samples:
            class_samples[sample['label']].append(sample)
        
        # Oversample
        augmented_samples = []
        for label, samples in class_samples.items():
            augmented_samples.extend(samples)
            
            # If class has fewer than min_samples, duplicate samples
            if len(samples) < min_samples:
                num_needed = min_samples - len(samples)
                for _ in range(num_needed):
                    # Randomly select a sample to duplicate
                    augmented_samples.append(random.choice(samples))
        
        logger.info(f"Oversampling: {len(self.samples)} -> {len(augmented_samples)} samples")
        return augmented_samples
    
    def _build_samples(self) -> List[Dict]:
        """
        Build dataset samples from frames.
        
        Groups frames by video and creates sequences for training.
        """
        samples = []
        
        # Get all frame files
        frame_files = sorted(list(self.frames_dir.glob("*.jpg")))
        
        if not frame_files:
            logger.warning(f"No frames found in {self.frames_dir}")
            return samples
        
        logger.info(f"Processing {len(frame_files)} frames...")
        
        # Group frames by video (assume naming: VIDEO_frame_NNN.jpg)
        video_groups = {}
        for frame_path in frame_files:
            # Extract video name from filename
            parts = frame_path.stem.split('_frame_')
            if len(parts) >= 2:
                video_name = '_frame_'.join(parts[:-1])
                frame_num = int(parts[-1])
                
                if video_name not in video_groups:
                    video_groups[video_name] = []
                video_groups[video_name].append((frame_num, frame_path))
        
        # Create sequences from grouped frames
        for video_name, frames in video_groups.items():
            frames.sort(key=lambda x: x[0])  # Sort by frame number
            
            # Try to infer sign from video name
            sign_label = self._infer_sign_from_video_name(video_name)
            
            if sign_label and sign_label in self.sign_to_idx:
                label_idx = self.sign_to_idx[sign_label]
                
                # Create sequences of specified length
                for i in range(0, len(frames) - self.sequence_length + 1, self.sequence_length):
                    sequence_frames = frames[i:i + self.sequence_length]
                    samples.append({
                        'frames': [f[1] for f in sequence_frames],
                        'label': label_idx,
                        'sign': sign_label,
                        'video': video_name
                    })
        
        return samples
    
    def _infer_sign_from_video_name(self, video_name: str) -> Optional[str]:
        """
        Try to infer sign label from video filename.
        
        This is a simple heuristic - in production, use proper annotation.
        """
        video_name_upper = video_name.upper()
        
        # Check dictionary for matching signs
        for entry in self.dictionary:
            sign_upper = entry['sign'].upper()
            if sign_upper in video_name_upper or video_name_upper in sign_upper:
                return entry['sign']
        
        return None
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Get a sample from the dataset.
        
        Uses cached landmarks if available (much faster), otherwise extracts on-the-fly.
        Applies augmentation if enabled.
        
        Returns:
            (landmarks_sequence, label, sign_name)
        """
        sample = self.samples[idx]
        
        # Load landmarks (from cache if available, otherwise extract)
        landmarks_sequence = []
        
        # Check if cache exists
        use_cache = LANDMARKS_CACHE_DIR.exists() and LANDMARKS_METADATA.exists()
        
        for frame_path in sample['frames']:
            landmarks = None
            
            if use_cache:
                # Try to load from cache
                cache_file = LANDMARKS_CACHE_DIR / f"{frame_path.stem}.npy"
                if cache_file.exists():
                    try:
                        landmarks = np.load(cache_file)
                    except Exception as e:
                        logger.debug(f"Failed to load cache for {frame_path.stem}: {e}")
            
            # Fallback to on-the-fly extraction if cache not available
            if landmarks is None:
                frame = cv2.imread(str(frame_path))
                if frame is None:
                    landmarks = np.zeros(225, dtype=np.float32)
                else:
                    extracted = self.extractor.extract_landmarks(frame)
                    landmarks = extracted if extracted is not None else np.zeros(225, dtype=np.float32)
            
            landmarks_sequence.append(landmarks)
        
        # Convert to numpy array
        landmarks_array = np.array(landmarks_sequence, dtype=np.float32)
        
        # Apply augmentation if enabled (only during training)
        if self.use_augmentation and self.augmenter is not None:
            landmarks_array = self.augmenter.augment_sequence(
                landmarks_array,
                label=sample['label']
            )
        
        # Convert to tensor
        landmarks_tensor = torch.from_numpy(landmarks_array)
        
        return landmarks_tensor, sample['label'], sample['sign']


def load_pretrained_model(
    model_path: Path,
    num_classes: int,
    freeze_backbone: bool = False
) -> nn.Module:
    """
    Load pretrained WLASL model and adapt for GSL fine-tuning.
    
    Args:
        model_path: Path to pretrained model checkpoint
        num_classes: Number of GSL sign classes
        freeze_backbone: If True, freeze feature extraction layers
        
    Returns:
        PyTorch model ready for fine-tuning
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required. Install with: pip install torch torchvision")
    
    if not model_path.exists():
        raise FileNotFoundError(f"Pretrained model not found: {model_path}")
    
    logger.info(f"Loading pretrained model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Determine model structure from checkpoint
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Create new model with GSL number of classes
    # Try to determine architecture from pretrained model if possible
    # For now, use SimpleI3D for compatibility
    model = SimpleI3D(
        input_features=225,
        num_classes=num_classes,
        hidden_dim=512,
        dropout_rate=0.3,
        lstm_layers=2,
        use_batch_norm=True
    )
    
    # Try to load compatible weights (skip incompatible layers)
    model_dict = model.state_dict()
    pretrained_dict = {}
    skipped_layers = []
    
    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                pretrained_dict[k] = v
            else:
                skipped_layers.append(f"{k} (shape mismatch: {model_dict[k].shape} vs {v.shape})")
        else:
            skipped_layers.append(f"{k} (not in model)")
    
    # Load compatible weights
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)
    
    loaded_count = len(pretrained_dict)
    logger.info(f"Loaded {loaded_count} compatible layers from pretrained model")
    if skipped_layers:
        logger.info(f"Skipped {len(skipped_layers)} incompatible layers (expected for classifier head)")
    
    # Freeze backbone if requested (keep classifier trainable)
    if freeze_backbone:
        for name, param in model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
        logger.info("Frozen feature extraction layers (only classifier will train)")
    
    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    use_mixup: bool = False,
    mixup_alpha: float = 0.2
) -> Dict[str, float]:
    """
    Train for one epoch with optional Mixup augmentation.
    
    Args:
        model: Model to train
        dataloader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        use_mixup: Enable Mixup augmentation
        mixup_alpha: Mixup alpha parameter
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (landmarks, labels, signs) in enumerate(dataloader):
        landmarks = landmarks.to(device)
        labels = labels.to(device)
        
        # Mixup augmentation (if enabled)
        if use_mixup and AUGMENTATION_AVAILABLE and np.random.random() < 0.5:
            # Randomly shuffle batch for mixup
            indices = torch.randperm(landmarks.size(0))
            landmarks_mixed = landmarks[indices]
            labels_mixed = labels[indices]
            
            # Generate lambda from beta distribution
            lam = np.random.beta(mixup_alpha, mixup_alpha)
            
            # Mix landmarks and labels
            mixed_landmarks = lam * landmarks + (1 - lam) * landmarks_mixed
            mixed_labels = (labels, labels_mixed, lam, 1 - lam)
            
            optimizer.zero_grad()
            outputs = model(mixed_landmarks)
            
            # Mixup loss: weighted combination
            loss1 = criterion(outputs, mixed_labels[0])
            loss2 = criterion(outputs, mixed_labels[1])
            loss = lam * loss1 + (1 - lam) * loss2
        else:
            # Standard training
            optimizer.zero_grad()
            outputs = model(landmarks)
            loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx + 1}/{len(dataloader)}, "
                f"Loss: {loss.item():.4f}, Acc: {100 * correct / total:.2f}%"
            )
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return {'loss': epoch_loss, 'accuracy': epoch_acc}


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for landmarks, labels, signs in dataloader:
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            
            outputs = model(landmarks)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return {'loss': epoch_loss, 'accuracy': epoch_acc}


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    optimizer: optim.Optimizer,
    metrics: Dict,
    filepath: Path
):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }
    torch.save(checkpoint, filepath)
    logger.info(f"Checkpoint saved to {filepath}")


def load_checkpoint(model: nn.Module, optimizer: optim.Optimizer, filepath: Path):
    """Load training checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    logger.info(f"Checkpoint loaded from {filepath}, resuming from epoch {epoch + 1}")
    return epoch


class DictionaryConnector:
    """
    Connect model predictions to GSL dictionary for English meanings.
    """
    
    def __init__(self, dictionary_path: Path):
        with open(dictionary_path, 'r', encoding='utf-8') as f:
            self.dictionary = json.load(f)
        
        # Create sign-to-entries mapping
        self.sign_map = {}
        for entry in self.dictionary:
            sign = entry['sign']
            if sign not in self.sign_map:
                self.sign_map[sign] = []
            self.sign_map[sign].append(entry)
    
    def get_meaning(self, sign: str) -> str:
        """Get English meaning for a sign."""
        if sign in self.sign_map:
            # Return first entry's meaning
            return self.sign_map[sign][0].get('meaning', 'No meaning found')
        return "Unknown sign"
    
    def predict_with_meaning(
        self,
        model: nn.Module,
        landmarks: torch.Tensor,
        idx_to_sign: Dict[int, str],
        device: torch.device,
        top_k: int = 1
    ) -> List[Dict]:
        """
        Get model prediction with English meaning.
        
        Returns:
            List of top-k predictions with sign, meaning, and confidence
        """
        model.eval()
        with torch.no_grad():
            landmarks = landmarks.unsqueeze(0).to(device)
            outputs = model(landmarks)
            probabilities = torch.softmax(outputs, dim=1)
            
            top_probs, top_indices = torch.topk(probabilities, k=top_k, dim=1)
            
            predictions = []
            for i in range(top_k):
                idx = top_indices[0][i].item()
                sign = idx_to_sign.get(idx, "Unknown")
                confidence = top_probs[0][i].item()
                meaning = self.get_meaning(sign)
                
                predictions.append({
                    'sign': sign,
                    'meaning': meaning,
                    'confidence': confidence
                })
            
            return predictions


def main():
    parser = argparse.ArgumentParser(description='Train GSL Sign Recognition Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--sequence-length', type=int, default=16, help='Frame sequence length')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--pretrained-model', type=str, default=None, help='Path to pretrained WLASL model')
    parser.add_argument('--freeze-backbone', action='store_true', help='Freeze feature extraction layers (fine-tune only classifier)')
    parser.add_argument('--fine-tune-lr', type=float, default=None, help='Learning rate for fine-tuning (if different from main LR)')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    
    # Augmentation arguments
    parser.add_argument('--augment', action='store_true', default=True, help='Enable data augmentation')
    parser.add_argument('--augment-noise', type=float, default=0.02, help='Landmark noise std')
    parser.add_argument('--oversample', action='store_true', default=True, help='Oversample rare classes')
    parser.add_argument('--min-samples', type=int, default=5, help='Minimum samples per class after oversampling')
    
    # Loss function arguments
    parser.add_argument('--loss', type=str, default='focal', choices=['ce', 'focal', 'smooth', 'combined'], help='Loss function type')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='Focal loss gamma parameter')
    parser.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing factor')
    parser.add_argument('--class-weights', action='store_true', default=True, help='Use class-weighted loss')
    
    # Training technique arguments
    parser.add_argument('--mixup', action='store_true', default=False, help='Enable Mixup augmentation')
    parser.add_argument('--mixup-alpha', type=float, default=0.2, help='Mixup alpha parameter')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine', 'plateau'], help='LR scheduler type')
    
    # Architecture arguments
    parser.add_argument('--architecture', type=str, default='simple', 
                       choices=['simple', 'i3d', 'hybrid'],
                       help='Model architecture: simple (SimpleI3D), i3d (FullI3D), hybrid (HybridI3D)')
    parser.add_argument('--base-channels', type=int, default=64, help='Base channels for I3D (multiplied for deeper layers)')
    parser.add_argument('--depth-factor', type=float, default=1.0, help='Depth multiplier for I3D channels (1.0=standard, 0.5=lighter, 2.0=deeper)')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    
    # Check dependencies
    if not TORCH_AVAILABLE:
        logger.error("PyTorch is required. Install with: pip install torch torchvision")
        sys.exit(1)
    
    if not MEDIAPIPE_AVAILABLE:
        logger.error("MediaPipe is required. Install with: pip install mediapipe")
        sys.exit(1)
    
    # Initialize landmark extractor
    logger.info("Initializing MediaPipe landmark extractor...")
    extractor = MediaPipeLandmarkExtractor()
    
    # Setup augmentation
    augmenter = None
    if args.augment and AUGMENTATION_AVAILABLE:
        augmenter = get_augmenter(
            noise_std=args.augment_noise,
            enable_temporal=True,
            enable_spatial=True,
            enable_mixup=False  # Mixup handled separately in training loop
        )
        logger.info("Data augmentation enabled")
    else:
        logger.info("Data augmentation disabled")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = GSLDataset(
        frames_dir=FRAMES_DIR,
        dictionary_path=DICTIONARY_PATH,
        sequence_length=args.sequence_length,
        extractor=extractor,
        augmenter=augmenter,
        use_augmentation=args.augment and AUGMENTATION_AVAILABLE,
        oversample_rare=args.oversample,
        min_samples_per_class=args.min_samples
    )
    
    if len(dataset) == 0:
        logger.error("No training samples found. Check your data paths.")
        sys.exit(1)
    
    # Split dataset (stratified split would be better, but random works for now)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_indices, val_indices = torch.utils.data.random_split(
        range(len(dataset)), [train_size, val_size]
    )
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(dataset, train_indices.indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices.indices)
    
    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Load or initialize model
    if args.pretrained_model:
        logger.info("="*60)
        logger.info("FINE-TUNING MODE: Loading pretrained model")
        logger.info("="*60)
        model = load_pretrained_model(
            Path(args.pretrained_model),
            num_classes=dataset.num_classes,
            freeze_backbone=args.freeze_backbone
        )
    else:
        logger.info("="*60)
        logger.info(f"TRAINING FROM SCRATCH - Architecture: {args.architecture}")
        logger.info("="*60)
        
        if args.architecture == 'i3d' and I3D_AVAILABLE:
            model = FullI3D(
                input_features=225,
                num_classes=dataset.num_classes,
                base_channels=args.base_channels,
                depth_factor=args.depth_factor,
                use_batch_norm=True,
                dropout_rate=0.3,
                sequence_length=args.sequence_length
            )
            logger.info(f"Using FullI3D architecture (base_channels={args.base_channels}, depth_factor={args.depth_factor})")
        elif args.architecture == 'hybrid' and I3D_AVAILABLE:
            model = HybridI3D(
                input_features=225,
                num_classes=dataset.num_classes,
                base_channels=args.base_channels,
                lstm_hidden=512,
                use_batch_norm=True,
                dropout_rate=0.3,
                sequence_length=args.sequence_length
            )
            logger.info(f"Using HybridI3D architecture (base_channels={args.base_channels})")
        else:
            if args.architecture != 'simple':
                logger.warning(f"Requested {args.architecture} architecture but I3D not available. Using SimpleI3D.")
            model = SimpleI3D(
                input_features=225,
                num_classes=dataset.num_classes,
                hidden_dim=512,
                dropout_rate=0.3,
                lstm_layers=2,
                use_batch_norm=True
            )
            logger.info("Using SimpleI3D architecture")
    
    model = model.to(device)
    logger.info(f"Model initialized with {dataset.num_classes} classes")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Calculate class weights if requested
    class_weights = None
    if args.class_weights and AUGMENTATION_AVAILABLE:
        logger.info("Calculating class weights...")
        # Calculate from full dataset
        class_weights = calculate_class_weights(dataset)
        class_weights = class_weights.to(device)
        logger.info(f"Class weights calculated (min: {class_weights.min():.2f}, max: {class_weights.max():.2f})")
    
    # Setup loss function
    if AUGMENTATION_AVAILABLE:
        criterion = get_loss_function(
            loss_type=args.loss,
            class_weights=class_weights,
            focal_gamma=args.focal_gamma,
            label_smoothing=args.label_smoothing if args.loss in ['smooth', 'combined'] else 0.0,
            device=str(device)
        )
        logger.info(f"Using {args.loss} loss function")
    else:
        if class_weights is not None:
            criterion = nn.CrossEntropyLoss(weight=class_weights)
        else:
            criterion = nn.CrossEntropyLoss()
        logger.info("Using standard CrossEntropy loss")
    
    # Setup optimizer with fine-tuning learning rate if specified
    train_lr = args.fine_tune_lr if (args.pretrained_model and args.fine_tune_lr) else args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=train_lr, weight_decay=1e-5)
    
    # Setup learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        logger.info("Using CosineAnnealingLR scheduler")
    elif args.scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
        logger.info("Using ReduceLROnPlateau scheduler")
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        logger.info("Using StepLR scheduler")
    
    if args.pretrained_model:
        logger.info(f"Fine-tuning learning rate: {train_lr} (from scratch: {args.learning_rate})")
    else:
        logger.info(f"Initial learning rate: {train_lr}")
    
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, Path(args.resume))
    
    # Training loop
    logger.info("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, args.epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{args.epochs}")
        logger.info(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch + 1,
            use_mixup=args.mixup,
            mixup_alpha=args.mixup_alpha
        )
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        
        # Update learning rate
        if args.scheduler == 'plateau':
            scheduler.step(val_metrics['accuracy'])
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Current LR: {current_lr:.6f}")
        
        # Save checkpoint
        checkpoint_path = CHECKPOINTS_DIR / f"checkpoint_epoch_{epoch + 1}.pth"
        save_checkpoint(model, epoch + 1, optimizer, val_metrics, checkpoint_path)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), TRAINED_MODEL_PATH)
            logger.info(f"✓ New best model saved (Val Acc: {best_val_acc:.2f}%)")
    
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Model saved to: {TRAINED_MODEL_PATH}")
    logger.info("="*60)
    
    # Test dictionary connection
    logger.info("\nTesting dictionary connection...")
    connector = DictionaryConnector(DICTIONARY_PATH)
    
    # Example prediction
    if len(val_subset) > 0:
        sample_landmarks, sample_label, sample_sign = val_subset[0]
        predictions = connector.predict_with_meaning(
            model,
            sample_landmarks,
            dataset.idx_to_sign,
            device,
            top_k=3
        )
        
        logger.info("\nExample prediction:")
        for pred in predictions:
            logger.info(
                f"  Sign: {pred['sign']}, "
                f"Meaning: {pred['meaning']}, "
                f"Confidence: {pred['confidence']:.2%}"
            )


if __name__ == "__main__":
    main()

