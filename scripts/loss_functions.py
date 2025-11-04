#!/usr/bin/env python3
"""
Loss Functions for Sign Recognition Training

Includes:
- Focal Loss for imbalanced datasets
- Class-Weighted CrossEntropy
- Label Smoothing CrossEntropy
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    
    Focal loss focuses learning on hard examples by down-weighting easy examples.
    Paper: https://arxiv.org/abs/1708.02002
    """
    
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0, reduction: str = 'mean'):
        """
        Initialize Focal Loss.
        
        Args:
            alpha: Weighting factor for each class (1D tensor). If None, uniform weights.
            gamma: Focusing parameter (higher = more focus on hard examples)
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.
        
        Args:
            inputs: Logits of shape (N, C)
            targets: Class indices of shape (N,)
            
        Returns:
            Scalar loss (or tensor if reduction='none')
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingCrossEntropy(nn.Module):
    """
    Cross Entropy Loss with Label Smoothing.
    
    Prevents overconfidence by smoothing target labels.
    """
    
    def __init__(self, smoothing: float = 0.1, reduction: str = 'mean'):
        """
        Initialize Label Smoothing CrossEntropy.
        
        Args:
            smoothing: Smoothing factor (0 = no smoothing, 1 = uniform)
            reduction: 'mean', 'sum', or 'none'
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label-smoothed cross entropy.
        
        Args:
            inputs: Logits of shape (N, C)
            targets: Class indices of shape (N,)
            
        Returns:
            Scalar loss
        """
        log_probs = F.log_softmax(inputs, dim=1)
        num_classes = inputs.size(1)
        
        # Create smooth labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # Compute KL divergence
        kl_div = F.kl_div(log_probs, true_dist, reduction='none').sum(dim=1)
        
        if self.reduction == 'mean':
            return kl_div.mean()
        elif self.reduction == 'sum':
            return kl_div.sum()
        else:
            return kl_div


class CombinedLoss(nn.Module):
    """
    Combined loss: Focal Loss + Label Smoothing.
    
    Best of both worlds for imbalanced datasets with overconfidence issues.
    """
    
    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        smoothing: float = 0.1,
        focal_weight: float = 1.0,
        smoothing_weight: float = 0.5
    ):
        """
        Initialize combined loss.
        
        Args:
            alpha: Class weights for focal loss
            gamma: Focal loss gamma parameter
            smoothing: Label smoothing factor
            focal_weight: Weight for focal loss component
            smoothing_weight: Weight for smoothing component
        """
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.smooth_loss = LabelSmoothingCrossEntropy(smoothing=smoothing)
        self.focal_weight = focal_weight
        self.smoothing_weight = smoothing_weight
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss."""
        focal = self.focal_loss(inputs, targets)
        smooth = self.smooth_loss(inputs, targets)
        return self.focal_weight * focal + self.smoothing_weight * smooth


def calculate_class_weights(dataset) -> torch.Tensor:
    """
    Calculate class weights inversely proportional to frequency.
    
    Args:
        dataset: Dataset with samples containing labels
        
    Returns:
        Tensor of weights for each class
    """
    from collections import Counter
    
    # Count class frequencies
    labels = []
    for i in range(len(dataset)):
        _, label, _ = dataset[i]
        labels.append(label)
    
    class_counts = Counter(labels)
    # Use dataset.num_classes instead of len(class_counts)
    # because labels are indices that can be any value up to num_classes-1
    num_classes = dataset.num_classes if hasattr(dataset, 'num_classes') else len(class_counts)
    
    # Calculate weights: total_samples / (num_classes * class_count)
    total_samples = sum(class_counts.values())
    weights = torch.ones(num_classes)  # Initialize with 1.0 for all classes
    
    # Only update weights for classes that appear in the dataset
    for class_idx, count in class_counts.items():
        if count > 0:
            weights[class_idx] = total_samples / (num_classes * count)
        else:
            weights[class_idx] = 1.0
    
    # Normalize
    if weights.sum() > 0:
        weights = weights / weights.mean()
    
    return weights


def get_loss_function(
    loss_type: str = 'ce',
    class_weights: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    label_smoothing: float = 0.0,
    device: str = 'cpu'
) -> nn.Module:
    """
    Factory function to create loss function.
    
    Args:
        loss_type: 'ce', 'focal', 'smooth', or 'combined'
        class_weights: Optional class weights tensor
        focal_gamma: Gamma for focal loss
        label_smoothing: Smoothing factor (0 = no smoothing)
        device: Device to move weights to
        
    Returns:
        Loss function module
    """
    if class_weights is not None:
        class_weights = class_weights.to(device)
    
    if loss_type == 'ce':
        if class_weights is not None:
            return nn.CrossEntropyLoss(weight=class_weights)
        return nn.CrossEntropyLoss()
    
    elif loss_type == 'focal':
        return FocalLoss(alpha=class_weights, gamma=focal_gamma)
    
    elif loss_type == 'smooth':
        return LabelSmoothingCrossEntropy(smoothing=label_smoothing)
    
    elif loss_type == 'combined':
        return CombinedLoss(
            alpha=class_weights,
            gamma=focal_gamma,
            smoothing=label_smoothing
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

