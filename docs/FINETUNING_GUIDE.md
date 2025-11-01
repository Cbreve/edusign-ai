# Fine-Tuning Guide - WLASL Pretrained Model

This guide explains how to fine-tune a pretrained WLASL model on GSL data.

## Prerequisites

1. **Pretrained WLASL Model**: You need a pretrained WLASL model checkpoint
2. **GSL Dataset**: Already prepared (8,980 validated frames)
3. **Dependencies**: All installed (PyTorch, MediaPipe, etc.)

## Getting Pretrained WLASL Models

### Option 1: Official WLASL Repository

```bash
# Clone the WLASL repository
git clone https://github.com/dxli94/WLASL.git
cd WLASL

# Follow their README for model download
# Models are typically available on their releases page or Google Drive
```

### Option 2: PyTorch Hub I3D Models

You can use pretrained I3D models from PyTorch Hub:

```python
import torch

# Load pretrained I3D
model = torch.hub.load('pytorch/vision', 'i3d_r50', pretrained=True)
# Save for use in training
torch.save(model.state_dict(), 'pretrained_i3d.pth')
```

### Option 3: Download from Research Papers

- WLASL paper: https://arxiv.org/abs/1910.11006
- Check paper repository for model links
- Common locations: Google Drive, Zenodo, Hugging Face

### Option 4: Train a Small Baseline First

If you can't find WLASL models immediately, you can:
1. Train a small model from scratch for a few epochs
2. Use that as a "pretrained" model
3. Fine-tune it further with more data

## Fine-Tuning Process

### Step 1: Place Pretrained Model

Place your pretrained model file in the models directory:

```bash
# Example
cp wlasl_i3d.pth backend/app/models/pretrained_wlasl.pth
```

### Step 2: Start Fine-Tuning

```bash
# Basic fine-tuning (all layers trainable)
python scripts/train_edusign_gsl.py \
    --pretrained-model backend/app/models/pretrained_wlasl.pth \
    --epochs 50 \
    --batch-size 16 \
    --fine-tune-lr 0.0001

# Fine-tuning with frozen backbone (only classifier trains)
python scripts/train_edusign_gsl.py \
    --pretrained-model backend/app/models/pretrained_wlasl.pth \
    --freeze-backbone \
    --epochs 30 \
    --batch-size 16 \
    --fine-tune-lr 0.001
```

## Fine-Tuning Options

### Full Fine-Tuning (Recommended)
```bash
--pretrained-model <path> --fine-tune-lr 0.0001
```
- All layers are trainable
- Lower learning rate (0.0001) to preserve pretrained knowledge
- Best for larger datasets

### Partial Fine-Tuning
```bash
--pretrained-model <path> --freeze-backbone --fine-tune-lr 0.001
```
- Only classifier head trains
- Feature extraction frozen
- Faster, requires less GPU memory
- Good for small datasets

### Progressive Fine-Tuning
1. Start with frozen backbone: `--freeze-backbone --epochs 10`
2. Unfreeze and fine-tune: `--epochs 40` (no freeze flag)
3. Lower learning rate: `--fine-tune-lr 0.00001`

## What Happens During Fine-Tuning

1. **Model Loading**: Pretrained weights are loaded
2. **Classifier Adaptation**: Last layer replaced for GSL classes (1,485 signs)
3. **Weight Transfer**: Compatible layers keep pretrained weights
4. **Training**: Model learns GSL-specific patterns

## Expected Improvements

Fine-tuning typically provides:
- ✅ Faster convergence (fewer epochs needed)
- ✅ Better accuracy (leverages WLASL knowledge)
- ✅ More stable training (better initialization)
- ✅ Works well with limited data

## Troubleshooting

### Model Not Loading

**Error**: `FileNotFoundError: Pretrained model not found`

**Solution**: Check file path is correct:
```bash
ls -lh backend/app/models/pretrained_wlasl.pth
```

### Shape Mismatch Warnings

**Warning**: `Skipped layers due to shape mismatch`

**Expected**: This is normal! The classifier head will have different output size (1485 GSL classes vs WLASL classes).

### Out of Memory

**Solution**: Use smaller batch size or freeze backbone:
```bash
--batch-size 8 --freeze-backbone
```

## Next Steps

1. Obtain pretrained WLASL model
2. Place in `backend/app/models/`
3. Run fine-tuning command
4. Monitor training progress
5. Evaluate on validation set

## Resources

- [WLASL GitHub](https://github.com/dxli94/WLASL)
- [WLASL Paper](https://arxiv.org/abs/1910.11006)
- [I3D Architecture](https://arxiv.org/abs/1705.07750)

