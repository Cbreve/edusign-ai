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
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import warnings

import numpy as np
import cv2
from PIL import Image

# ML Framework imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
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
    Simplified I3D-like architecture for sign recognition.
    
    This is a simplified version of the Inflated 3D ConvNet (I3D) architecture
    commonly used in WLASL. For production, use a pretrained I3D model.
    """
    
    def __init__(self, input_features: int = 225, num_classes: int = 100, hidden_dim: int = 512):
        super(SimpleI3D, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Temporal modeling (LSTM for sequence understanding)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, num_classes)
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
        x_reshaped = x.view(-1, features)
        features_out = self.feature_extractor(x_reshaped)
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
    Dataset for GSL sign recognition.
    
    Loads video frames, extracts MediaPipe landmarks, and pairs them with sign labels.
    """
    
    def __init__(
        self,
        frames_dir: Path,
        dictionary_path: Path,
        sequence_length: int = 16,
        transform: Optional[transforms.Compose] = None,
        extractor: Optional[MediaPipeLandmarkExtractor] = None
    ):
        self.frames_dir = Path(frames_dir)
        self.sequence_length = sequence_length
        self.transform = transform
        
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
        
        logger.info(f"Dataset initialized with {len(self.samples)} sequences")
    
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
        
        # Convert to tensor
        landmarks_array = np.array(landmarks_sequence, dtype=np.float32)
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
    model = SimpleI3D(input_features=225, num_classes=num_classes, hidden_dim=512)
    
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
    epoch: int
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (landmarks, labels, signs) in enumerate(dataloader):
        landmarks = landmarks.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(landmarks)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
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
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = GSLDataset(
        frames_dir=FRAMES_DIR,
        dictionary_path=DICTIONARY_PATH,
        sequence_length=args.sequence_length,
        extractor=extractor
    )
    
    if len(dataset) == 0:
        logger.error("No training samples found. Check your data paths.")
        sys.exit(1)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
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
        logger.info("TRAINING FROM SCRATCH")
        logger.info("="*60)
        model = SimpleI3D(
            input_features=225,
            num_classes=dataset.num_classes,
            hidden_dim=512
        )
    
    model = model.to(device)
    logger.info(f"Model initialized with {dataset.num_classes} classes")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")
    
    # Setup training with fine-tuning learning rate if specified
    criterion = nn.CrossEntropyLoss()
    train_lr = args.fine_tune_lr if (args.pretrained_model and args.fine_tune_lr) else args.learning_rate
    optimizer = optim.Adam(model.parameters(), lr=train_lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    if args.pretrained_model:
        logger.info(f"Fine-tuning learning rate: {train_lr} (from scratch: {args.learning_rate})")
    
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
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device, epoch + 1)
        logger.info(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        
        # Save checkpoint
        checkpoint_path = CHECKPOINTS_DIR / f"checkpoint_epoch_{epoch + 1}.pth"
        save_checkpoint(model, epoch + 1, optimizer, val_metrics, checkpoint_path)
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            torch.save(model.state_dict(), TRAINED_MODEL_PATH)
            logger.info(f"✓ New best model saved (Val Acc: {best_val_acc:.2f}%)")
        
        scheduler.step()
    
    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    logger.info(f"Model saved to: {TRAINED_MODEL_PATH}")
    logger.info("="*60)
    
    # Test dictionary connection
    logger.info("\nTesting dictionary connection...")
    connector = DictionaryConnector(DICTIONARY_PATH)
    
    # Example prediction
    if len(val_dataset) > 0:
        sample_landmarks, sample_label, sample_sign = val_dataset[0]
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

