#!/usr/bin/env python3
"""
Data Augmentation for GSL Sign Recognition

Implements various augmentation techniques for landmark sequences:
- Landmark noise injection
- Temporal augmentation (speed variation, frame dropping)
- Spatial augmentation (mirroring, rotation simulation)
- Sequence length variation
"""

import numpy as np
import random
from typing import Optional, Tuple
import torch


class LandmarkAugmenter:
    """
    Augmentation module for MediaPipe landmark sequences.
    
    Applies various augmentation techniques to improve model generalization.
    """
    
    def __init__(
        self,
        noise_std: float = 0.02,
        temporal_jitter: bool = True,
        spatial_augment: bool = True,
        mirror_prob: float = 0.3,
        temporal_speed_range: Tuple[float, float] = (0.8, 1.2),
        enable_mixup: bool = False,
        mixup_alpha: float = 0.2
    ):
        """
        Initialize augmenter.
        
        Args:
            noise_std: Standard deviation for landmark noise (as fraction of range)
            temporal_jitter: Enable temporal augmentations
            spatial_augment: Enable spatial augmentations (mirroring)
            mirror_prob: Probability of mirroring sequence
            temporal_speed_range: Range for speed variation (slow-fast multiplier)
            enable_mixup: Enable Mixup augmentation (requires paired data)
            mixup_alpha: Mixup alpha parameter
        """
        self.noise_std = noise_std
        self.temporal_jitter = temporal_jitter
        self.spatial_augment = spatial_augment
        self.mirror_prob = mirror_prob
        self.temporal_speed_range = temporal_speed_range
        self.enable_mixup = enable_mixup
        self.mixup_alpha = mixup_alpha
    
    def add_noise(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Add Gaussian noise to landmarks.
        
        Args:
            landmarks: Landmark array of shape (225,) or (seq_len, 225)
            
        Returns:
            Augmented landmarks with noise
        """
        noise = np.random.normal(0, self.noise_std, landmarks.shape).astype(np.float32)
        # Normalize noise to landmark coordinate range (0-1 for x,y, -1-1 for z)
        augmented = landmarks + noise
        
        # Clip to valid ranges
        if len(landmarks.shape) == 1:
            # Single frame: shape (225,)
            augmented[:99] = np.clip(augmented[:99], 0, 1)  # Pose x,y
            augmented[99:198] = np.clip(augmented[99:198], 0, 1)  # Left hand x,y
            augmented[198:] = np.clip(augmented[198:], 0, 1)  # Right hand x,y
            # z coordinates are less constrained
        else:
            # Sequence: shape (seq_len, 225)
            augmented[:, :99] = np.clip(augmented[:, :99], 0, 1)
            augmented[:, 99:198] = np.clip(augmented[:, 99:198], 0, 1)
            augmented[:, 198:] = np.clip(augmented[:, 198:], 0, 1)
        
        return augmented
    
    def temporal_speed_variation(self, landmarks_seq: np.ndarray) -> np.ndarray:
        """
        Apply temporal speed variation (slow motion / fast forward).
        
        Args:
            landmarks_seq: Sequence of landmarks (seq_len, 225)
            
        Returns:
            Augmented sequence with varying frame rate
        """
        if not self.temporal_jitter:
            return landmarks_seq
        
        speed_factor = np.random.uniform(*self.temporal_speed_range)
        seq_len = len(landmarks_seq)
        
        if speed_factor < 1.0:
            # Slow down: duplicate frames
            target_len = int(seq_len / speed_factor)
            indices = np.linspace(0, seq_len - 1, target_len, dtype=int)
            augmented = landmarks_seq[indices]
        else:
            # Speed up: skip frames
            target_len = int(seq_len / speed_factor)
            if target_len < 1:
                target_len = 1
            indices = np.linspace(0, seq_len - 1, target_len, dtype=int)
            augmented = landmarks_seq[indices]
        
        # Ensure output length matches input
        if len(augmented) != seq_len:
            # Pad or trim to original length
            if len(augmented) < seq_len:
                # Repeat last frame
                padding = np.repeat(augmented[-1:], seq_len - len(augmented), axis=0)
                augmented = np.vstack([augmented, padding])
            else:
                # Take evenly spaced frames
                indices = np.linspace(0, len(augmented) - 1, seq_len, dtype=int)
                augmented = augmented[indices]
        
        return augmented
    
    def random_frame_drop(self, landmarks_seq: np.ndarray, drop_prob: float = 0.1) -> np.ndarray:
        """
        Randomly drop frames from sequence.
        
        Args:
            landmarks_seq: Sequence of landmarks (seq_len, 225)
            drop_prob: Probability of dropping each frame
            
        Returns:
            Augmented sequence with some frames dropped
        """
        if not self.temporal_jitter or drop_prob <= 0:
            return landmarks_seq
        
        seq_len = len(landmarks_seq)
        keep_mask = np.random.random(seq_len) > drop_prob
        
        if keep_mask.sum() == 0:
            # Keep at least one frame
            keep_mask[np.random.randint(seq_len)] = True
        
        augmented = landmarks_seq[keep_mask]
        
        # Pad back to original length
        if len(augmented) < seq_len:
            # Repeat last frame
            padding = np.repeat(augmented[-1:], seq_len - len(augmented), axis=0)
            augmented = np.vstack([augmented, padding])
        
        return augmented
    
    def mirror_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Mirror landmarks horizontally (swap left/right).
        
        Args:
            landmarks: Landmark array (225,) or (seq_len, 225)
            
        Returns:
            Mirrored landmarks
        """
        augmented = landmarks.copy()
        
        # MediaPipe landmarks are in normalized coordinates (0-1)
        # Pose landmarks (33 points): indices 0-98 (x, y, z for each)
        # Left hand (21 points): indices 99-161
        # Right hand (21 points): indices 162-224
        
        if len(landmarks.shape) == 1:
            # Single frame
            # Mirror x coordinates (0 -> 1, 1 -> 0)
            augmented[0::3] = 1.0 - augmented[0::3]  # All x coordinates
            
            # Swap left and right hands
            left_hand = augmented[99:162].copy()
            right_hand = augmented[162:225].copy()
            augmented[99:162] = right_hand
            augmented[162:225] = left_hand
        else:
            # Sequence
            # Mirror x coordinates
            augmented[:, 0::3] = 1.0 - augmented[:, 0::3]
            
            # Swap hands for each frame
            left_hand = augmented[:, 99:162].copy()
            right_hand = augmented[:, 162:225].copy()
            augmented[:, 99:162] = right_hand
            augmented[:, 162:225] = left_hand
        
        return augmented
    
    def augment_sequence(
        self,
        landmarks_seq: np.ndarray,
        label: Optional[int] = None,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Apply random augmentations to landmark sequence.
        
        Args:
            landmarks_seq: Input sequence (seq_len, 225)
            label: Class label (for mixup if enabled)
            deterministic: If True, apply all augmentations (for testing)
            
        Returns:
            Augmented sequence
        """
        if deterministic:
            # Apply all augmentations
            augmented = landmarks_seq.copy()
            augmented = self.add_noise(augmented)
            augmented = self.temporal_speed_variation(augmented)
            if self.spatial_augment:
                augmented = self.mirror_landmarks(augmented)
            return augmented
        
        augmented = landmarks_seq.copy()
        
        # Random noise (always apply with probability)
        if np.random.random() < 0.8:  # 80% chance
            augmented = self.add_noise(augmented)
        
        # Temporal augmentations
        if self.temporal_jitter:
            if np.random.random() < 0.5:  # 50% chance
                augmented = self.temporal_speed_variation(augmented)
            
            if np.random.random() < 0.3:  # 30% chance
                augmented = self.random_frame_drop(augmented, drop_prob=0.1)
        
        # Spatial augmentation (mirroring)
        if self.spatial_augment and np.random.random() < self.mirror_prob:
            augmented = self.mirror_landmarks(augmented)
        
        return augmented
    
    def mixup(
        self,
        landmarks_seq1: np.ndarray,
        label1: int,
        landmarks_seq2: np.ndarray,
        label2: int
    ) -> Tuple[np.ndarray, Tuple[int, int], Tuple[float, float]]:
        """
        Apply Mixup augmentation between two sequences.
        
        Args:
            landmarks_seq1: First sequence (seq_len, 225)
            label1: Label for first sequence
            landmarks_seq2: Second sequence (seq_len, 225)
            label2: Label for second sequence
            
        Returns:
            Mixed sequence, tuple of labels, tuple of mixup weights
        """
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Ensure sequences have same length
        if len(landmarks_seq1) != len(landmarks_seq2):
            target_len = max(len(landmarks_seq1), len(landmarks_seq2))
            if len(landmarks_seq1) < target_len:
                padding = np.repeat(landmarks_seq1[-1:], target_len - len(landmarks_seq1), axis=0)
                landmarks_seq1 = np.vstack([landmarks_seq1, padding])
            if len(landmarks_seq2) < target_len:
                padding = np.repeat(landmarks_seq2[-1:], target_len - len(landmarks_seq2), axis=0)
                landmarks_seq2 = np.vstack([landmarks_seq2, padding])
        
        mixed_seq = lam * landmarks_seq1 + (1 - lam) * landmarks_seq2
        
        return mixed_seq, (label1, label2), (lam, 1 - lam)


def get_augmenter(
    noise_std: float = 0.02,
    enable_temporal: bool = True,
    enable_spatial: bool = True,
    enable_mixup: bool = False
) -> LandmarkAugmenter:
    """
    Factory function to create augmenter with common configurations.
    
    Args:
        noise_std: Landmark noise standard deviation
        enable_temporal: Enable temporal augmentations
        enable_spatial: Enable spatial augmentations
        enable_mixup: Enable Mixup augmentation
        
    Returns:
        Configured LandmarkAugmenter instance
    """
    return LandmarkAugmenter(
        noise_std=noise_std,
        temporal_jitter=enable_temporal,
        spatial_augment=enable_spatial,
        mirror_prob=0.3,
        temporal_speed_range=(0.8, 1.2),
        enable_mixup=enable_mixup,
        mixup_alpha=0.2
    )

