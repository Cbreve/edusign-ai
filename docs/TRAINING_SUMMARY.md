# Training Summary

## Model Performance

- **Best Validation Accuracy**: 95.45%
- **Final Training Accuracy**: 92.26%
- **Model Size**: 47MB
- **Classes**: 1,485 GSL signs

## Training Details

- **Base Model**: WLASL I3D (pretrained on 2000 ASL signs)
- **Dataset**: 549 sequences (439 train, 110 validation)
- **Frames**: 8,980 validated frames
- **Training Time**: ~1 minute (with cached landmarks)
- **Epochs**: 50
- **Batch Size**: 16
- **Learning Rate**: 0.0001 (fine-tuning)

## Model Location

- **Final Model**: `backend/app/models/edusign_gsl_finetuned.pth`
- **Checkpoints**: `backend/app/models/checkpoints/checkpoint_epoch_N.pth`

## Usage

See `TRAINING_GUIDE.md` and `FINETUNING_GUIDE.md` for detailed instructions.

