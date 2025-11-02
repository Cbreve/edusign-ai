#!/usr/bin/env python3
"""
Full I3D (Inflated 3D ConvNet) Architecture for Sign Recognition

This module implements a proper I3D architecture adapted for landmark-based sign recognition.
I3D uses 3D convolutions to capture spatiotemporal features effectively.

Paper: "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
https://arxiv.org/abs/1705.07750
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class I3DBlock(nn.Module):
    """
    Basic I3D block with 3D convolution, batch norm, and ReLU.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        super(I3DBlock, self).__init__()
        
        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(kernel_size, kernel_size, kernel_size),
            stride=(stride, stride, stride),
            padding=(padding, padding, padding),
            bias=not use_batch_norm
        )
        
        self.bn = nn.BatchNorm3d(out_channels) if use_batch_norm else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout3d(dropout_rate) if dropout_rate > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class I3DLayer(nn.Module):
    """
    I3D layer with multiple blocks and optional pooling.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int = 2,
        pool: bool = True,
        pool_stride: int = 2,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ):
        super(I3DLayer, self).__init__()
        
        blocks = []
        for i in range(num_blocks):
            stride = 2 if (i == 0 and pool) else 1
            blocks.append(
                I3DBlock(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    use_batch_norm=use_batch_norm,
                    dropout_rate=dropout_rate if i == num_blocks - 1 else 0.0
                )
            )
        
        self.blocks = nn.Sequential(*blocks)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(x)


class FullI3D(nn.Module):
    """
    Full I3D architecture adapted for landmark-based sign recognition.
    
    Adapts landmark sequences (225 features per frame) to 3D convolutional input
    by reshaping and using 1x1x1 convolutions to create channels.
    
    Architecture:
    1. Input projection (landmarks -> 3D tensor)
    2. Multiple I3D layers with increasing channels
    3. Global average pooling
    4. Classification head
    """
    
    def __init__(
        self,
        input_features: int = 225,
        num_classes: int = 100,
        base_channels: int = 64,
        depth_factor: float = 1.0,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.3,
        sequence_length: int = 16
    ):
        """
        Initialize Full I3D model.
        
        Args:
            input_features: Number of landmark features per frame (225 for MediaPipe)
            num_classes: Number of sign classes
            base_channels: Base number of channels (will be multiplied by depth_factor)
            depth_factor: Multiplier for channel depth (1.0 = standard, 0.5 = lighter, 2.0 = deeper)
            use_batch_norm: Use batch normalization
            dropout_rate: Dropout rate for regularization
            sequence_length: Length of input sequences
        """
        super(FullI3D, self).__init__()
        
        self.input_features = input_features
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        
        # Project landmarks to 3D tensor format
        # Strategy: Reshape landmarks to a spatial grid (e.g., 15x15 for 225 features)
        # Then use 1x1x1 convs to create channels
        
        # Find reasonable spatial dimensions (should multiply to input_features)
        # For 225 features, we can use 15x15 = 225
        spatial_size = int(input_features ** 0.5)
        if spatial_size * spatial_size != input_features:
            # Find closest square
            spatial_size = int(input_features ** 0.5) + 1
            # We'll pad/trim as needed
        
        self.spatial_size = spatial_size
        self.padded_features = spatial_size * spatial_size
        
        # Input projection: reshape landmarks to spatial format
        # (batch, seq_len, features) -> (batch, seq_len, spatial_size, spatial_size)
        self.input_projection = nn.Sequential(
            nn.Linear(input_features, self.padded_features),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Convert to 3D format: (batch, 1, seq_len, spatial_size, spatial_size)
        # Then use 1x1x1 conv to create initial channels
        self.input_conv = nn.Conv3d(
            in_channels=1,
            out_channels=int(base_channels * depth_factor),
            kernel_size=(1, 1, 1),
            stride=1,
            padding=0
        )
        
        # I3D layers with increasing channels
        channels = [
            int(base_channels * depth_factor),
            int(base_channels * 2 * depth_factor),
            int(base_channels * 4 * depth_factor),
            int(base_channels * 8 * depth_factor)
        ]
        
        self.layer1 = I3DLayer(
            channels[0], channels[1],
            num_blocks=2,
            pool=True,
            use_batch_norm=use_batch_norm,
            dropout_rate=0.0
        )
        
        self.layer2 = I3DLayer(
            channels[1], channels[2],
            num_blocks=2,
            pool=True,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate * 0.5
        )
        
        self.layer3 = I3DLayer(
            channels[2], channels[3],
            num_blocks=3,
            pool=True,
            use_batch_norm=use_batch_norm,
            dropout_rate=dropout_rate
        )
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(channels[3], channels[3] // 2),
            nn.BatchNorm1d(channels[3] // 2) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(channels[3] // 2, channels[3] // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(channels[3] // 4, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_features)
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        batch_size, seq_len, features = x.shape
        
        # Project to padded features
        x = self.input_projection(x)  # (batch, seq_len, padded_features)
        
        # Reshape to spatial format: (batch, seq_len, spatial_size, spatial_size)
        x = x.view(batch_size, seq_len, self.spatial_size, self.spatial_size)
        
        # Convert to 3D format: (batch, channels=1, seq_len, spatial_size, spatial_size)
        x = x.unsqueeze(1)
        
        # Initial convolution to create channels
        x = self.input_conv(x)  # (batch, base_channels, seq_len, spatial_size, spatial_size)
        
        # Apply I3D layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Global average pooling
        x = self.global_pool(x)  # (batch, channels, 1, 1, 1)
        
        # Flatten
        x = x.view(batch_size, -1)  # (batch, channels)
        
        # Classification
        logits = self.classifier(x)
        
        return logits


class HybridI3D(nn.Module):
    """
    Hybrid I3D architecture combining 3D convolutions with LSTM.
    
    Uses 3D convolutions for spatiotemporal feature extraction,
    then LSTM for temporal sequence modeling.
    """
    
    def __init__(
        self,
        input_features: int = 225,
        num_classes: int = 100,
        base_channels: int = 64,
        lstm_hidden: int = 256,
        use_batch_norm: bool = True,
        dropout_rate: float = 0.3,
        sequence_length: int = 16
    ):
        super(HybridI3D, self).__init__()
        
        self.input_features = input_features
        self.sequence_length = sequence_length
        
        # Use simplified I3D as feature extractor
        spatial_size = int(input_features ** 0.5) + 1
        self.spatial_size = spatial_size
        self.padded_features = spatial_size * spatial_size
        
        # Input projection
        self.input_projection = nn.Sequential(
            nn.Linear(input_features, self.padded_features),
            nn.ReLU()
        )
        
        # Reduced I3D layers (feature extraction only)
        self.i3d_conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=base_channels,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1)
        )
        
        self.i3d_bn1 = nn.BatchNorm3d(base_channels) if use_batch_norm else nn.Identity()
        
        self.i3d_conv2 = nn.Conv3d(
            in_channels=base_channels,
            out_channels=base_channels * 2,
            kernel_size=(3, 3, 3),
            stride=(2, 2, 2),
            padding=(1, 1, 1)
        )
        
        self.i3d_bn2 = nn.BatchNorm3d(base_channels * 2) if use_batch_norm else nn.Identity()
        
        # Calculate LSTM input size after I3D feature extraction
        # This depends on spatial_size and pooling
        # Approximate: after 2x2x2 pooling, spatial reduces by factor of 2
        reduced_spatial = spatial_size // 2
        reduced_seq = sequence_length // 2
        lstm_input_size = (base_channels * 2) * reduced_spatial * reduced_spatial
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=dropout_rate if 2 > 1 else 0
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.BatchNorm1d(lstm_hidden) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(lstm_hidden, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, features = x.shape
        
        # Project to spatial format
        x = self.input_projection(x)
        x = x.view(batch_size, seq_len, self.spatial_size, self.spatial_size)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # I3D feature extraction
        x = F.relu(self.i3d_bn1(self.i3d_conv1(x)))
        x = F.relu(self.i3d_bn2(self.i3d_conv2(x)))
        
        # Reshape for LSTM: (batch, seq, features)
        # After pooling, sequence length and spatial dimensions are reduced
        reduced_seq = x.size(2)
        x = x.permute(0, 2, 1, 3, 4)  # (batch, seq, channels, h, w)
        x = x.contiguous().view(batch_size, reduced_seq, -1)
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state
        final_hidden = lstm_out[:, -1, :]
        
        # Classify
        logits = self.classifier(final_hidden)
        
        return logits


def get_i3d_model(
    model_type: str = 'full',
    input_features: int = 225,
    num_classes: int = 100,
    **kwargs
) -> nn.Module:
    """
    Factory function to create I3D model.
    
    Args:
        model_type: 'full' for FullI3D, 'hybrid' for HybridI3D
        input_features: Number of input features per frame
        num_classes: Number of classes
        **kwargs: Additional arguments for model initialization
        
    Returns:
        I3D model instance
    """
    if model_type == 'full':
        return FullI3D(
            input_features=input_features,
            num_classes=num_classes,
            **kwargs
        )
    elif model_type == 'hybrid':
        return HybridI3D(
            input_features=input_features,
            num_classes=num_classes,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'full' or 'hybrid'")


# For backward compatibility and easy comparison
I3D = FullI3D

