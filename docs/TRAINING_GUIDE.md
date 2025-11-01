# GSL Sign Recognition Model Training Guide

This guide explains how to train the EduSign AI GSL (Ghanaian Sign Language) recognition model using a pretrained WLASL base model.

## Overview

The training pipeline:
1. Loads pretrained WLASL model (I3D architecture)
2. Preprocesses GSL video frames from validated dataset
3. Extracts pose and hand landmarks using MediaPipe
4. Fine-tunes the model on GSL-specific data
5. Saves the trained model and connects predictions to dictionary

## Prerequisites

### Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Key dependencies:
- **PyTorch** (≥2.0.0) - Deep learning framework
- **MediaPipe** (≥0.10.0) - Landmark extraction
- **OpenCV** - Video/frame processing
- **NumPy** - Numerical operations

### Data Requirements

1. **Validated Video Frames**: `backend/app/data/processed/validated_frames/`
   - High-quality frames extracted from YouTube videos
   - ~8,980 validated frames ready for training

2. **GSL Dictionary**: `backend/app/data/processed/gsl_dictionary.json`
   - Sign-to-meaning mappings
   - Used for label mapping and prediction interpretation

## Getting Pretrained WLASL Models

WLASL models are typically available as:

### Option 1: Official WLASL Repository
```bash
# Clone WLASL repository
git clone https://github.com/dxli94/WLASL.git
cd WLASL

# Download pretrained I3D model
# Follow their instructions for model download
```

### Option 2: PyTorch Hub (if available)
```python
import torch
model = torch.hub.load('pytorch/vision', 'i3d_r50', pretrained=True)
```

### Option 3: Use Simplified Model (Current Implementation)
The current script uses a simplified I3D-like architecture that can be trained from scratch or fine-tuned. For production, replace with actual pretrained WLASL model.

## Training Steps

### 1. Basic Training

Train with default settings:
```bash
python scripts/train_edusign_gsl.py
```

### 2. Custom Training Parameters

```bash
python scripts/train_edusign_gsl.py \
    --epochs 100 \
    --batch-size 32 \
    --learning-rate 0.0001 \
    --sequence-length 16
```

### 3. Resume from Checkpoint

Resume training from a saved checkpoint:
```bash
python scripts/train_edusign_gsl.py \
    --resume backend/app/models/checkpoints/checkpoint_epoch_20.pth \
    --epochs 100
```

### 4. Use Pretrained WLASL Model

```bash
python scripts/train_edusign_gsl.py \
    --pretrained-model path/to/wlasl_i3d.pth \
    --epochs 50
```

## Training Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--epochs` | 50 | Number of training epochs |
| `--batch-size` | 16 | Batch size for training |
| `--learning-rate` | 0.001 | Learning rate |
| `--sequence-length` | 16 | Number of frames per sequence |
| `--resume` | None | Path to checkpoint to resume from |
| `--pretrained-model` | None | Path to pretrained WLASL model |
| `--device` | auto | Device (cuda/cpu/auto) |

## Data Pipeline

### Frame Processing Flow

```
Video Frames (JPG)
    ↓
MediaPipe Extraction
    ↓
Pose Landmarks (33 points × 3 = 99 features)
Left Hand Landmarks (21 points × 3 = 63 features)
Right Hand Landmarks (21 points × 3 = 63 features)
    ↓
Combined Features (225 features per frame)
    ↓
Sequence Formation (16 frames = 1 sample)
    ↓
Model Training
```

### Dataset Structure

The dataset automatically:
- Groups frames by video source
- Creates sequences of specified length
- Maps signs to labels using dictionary
- Splits into train/validation (80/20)

## Model Architecture

### Simplified I3D (Current Implementation)

```
Input: (batch_size, sequence_length, 225 features)
    ↓
Feature Extractor (Linear layers)
    ↓
LSTM (Bidirectional, 2 layers)
    ↓
Classification Head
    ↓
Output: (batch_size, num_classes)
```

### For Production

Replace with actual I3D architecture:
- Inflated 3D Convolutional Networks
- Pretrained on WLASL (2000 signs)
- Fine-tuned on GSL data

## Training Outputs

### Saved Files

1. **Best Model**: `backend/app/models/edusign_gsl_finetuned.pth`
   - Best model based on validation accuracy
   - Use for inference

2. **Checkpoints**: `backend/app/models/checkpoints/checkpoint_epoch_N.pth`
   - Saved every epoch
   - Contains model state, optimizer state, metrics

3. **Training Log**: `training.log`
   - Detailed training logs
   - Loss and accuracy per epoch/batch

### Model Usage

```python
import torch
from scripts.train_edusign_gsl import DictionaryConnector, SimpleI3D

# Load trained model
model = SimpleI3D(input_features=225, num_classes=100, hidden_dim=512)
model.load_state_dict(torch.load('backend/app/models/edusign_gsl_finetuned.pth'))
model.eval()

# Load dictionary connector
connector = DictionaryConnector('backend/app/data/processed/gsl_dictionary.json')

# Make prediction (example)
# predictions = connector.predict_with_meaning(model, landmarks, idx_to_sign, device)
```

## Dictionary Integration

The model automatically connects predictions to `gsl_dictionary.json`:

```python
# Example output
[
    {
        'sign': 'HELLO',
        'meaning': 'Raise the hand and move it forward.',
        'confidence': 0.95
    },
    ...
]
```

## Monitoring Training

### Training Metrics

- **Loss**: Cross-entropy loss (lower is better)
- **Accuracy**: Classification accuracy (higher is better)
- **Learning Rate**: Automatically scheduled (reduces every 10 epochs)

### Expected Training Time

- **Hardware**: GPU recommended (CUDA)
- **Time per epoch**: ~5-30 minutes (depends on dataset size)
- **Total time**: Varies with epochs

## Troubleshooting

### Common Issues

1. **No landmarks detected**
   - Ensure frames contain visible person/signer
   - Check frame quality (resolution, brightness)

2. **Out of memory**
   - Reduce batch size: `--batch-size 8`
   - Reduce sequence length: `--sequence-length 8`

3. **Poor accuracy**
   - Increase epochs: `--epochs 100`
   - Lower learning rate: `--learning-rate 0.0001`
   - Add more training data

4. **MediaPipe errors**
   - Ensure MediaPipe is installed: `pip install mediapipe`
   - Check frame format (should be BGR from OpenCV)

## Next Steps

1. **Annotation**: Add proper frame-level labels for better supervision
2. **Data Augmentation**: Add rotation, scaling, brightness variations
3. **Pretrained Model**: Integrate actual WLASL I3D model
4. **Evaluation**: Create test set and measure per-sign accuracy
5. **Inference API**: Integrate model into FastAPI backend

## References

- [WLASL Paper](https://arxiv.org/abs/1910.11006)
- [I3D Architecture](https://arxiv.org/abs/1705.07750)
- [MediaPipe Documentation](https://google.github.io/mediapipe/)

