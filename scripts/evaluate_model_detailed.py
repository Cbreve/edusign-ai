#!/usr/bin/env python3
"""
Detailed Model Evaluation Script

Provides comprehensive evaluation metrics including:
- Overall accuracy
- Per-class accuracy
- Top-5 accuracy
- Confusion matrix
- Worst performing classes
- Best performing classes
"""

import argparse
import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Import from training script
sys.path.insert(0, str(Path(__file__).parent))
from train_edusign_gsl import (
    SimpleI3D, GSLDataset, MediaPipeLandmarkExtractor,
    FRAMES_DIR, DICTIONARY_PATH, MODELS_DIR
)


def load_model(model_path: Path, num_classes: int, device: torch.device) -> nn.Module:
    """Load trained model."""
    model = SimpleI3D(
        input_features=225,
        num_classes=num_classes,
        hidden_dim=512,
        dropout_rate=0.3,
        lstm_layers=2,
        use_batch_norm=True
    )
    
    checkpoint = torch.load(model_path, map_location='cpu')
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model


def evaluate_detailed(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    idx_to_sign: Dict[int, str],
    top_k: int = 5
) -> Dict:
    """
    Perform detailed evaluation.
    
    Returns:
        Dictionary with comprehensive metrics
    """
    model.eval()
    
    all_preds = []
    all_labels = []
    all_top_k_preds = []
    all_probs = []
    
    correct = 0
    top_k_correct = 0
    total = 0
    
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for landmarks, labels, signs in dataloader:
            landmarks = landmarks.to(device)
            labels = labels.to(device)
            
            outputs = model(landmarks)
            probs = torch.softmax(outputs, dim=1)
            
            # Top-1 predictions
            _, predicted = torch.max(outputs.data, 1)
            
            # Top-k predictions
            top_k_probs, top_k_indices = torch.topk(probs, k=min(top_k, outputs.size(1)), dim=1)
            
            # Statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Top-k accuracy
            for i in range(labels.size(0)):
                if labels[i] in top_k_indices[i]:
                    top_k_correct += 1
            
            # Per-class statistics
            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                class_total[label] += 1
                if pred == label:
                    class_correct[label] += 1
            
            # Store predictions for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_top_k_preds.append(top_k_indices.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100 * correct / total
    top_k_accuracy = 100 * top_k_correct / total
    
    # Per-class accuracy
    per_class_acc = {}
    for class_idx in class_total:
        if class_total[class_idx] > 0:
            per_class_acc[class_idx] = 100 * class_correct[class_idx] / class_total[class_idx]
        else:
            per_class_acc[class_idx] = 0.0
    
    # Sort classes by accuracy
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1])
    
    return {
        'overall_accuracy': accuracy,
        'top_k_accuracy': top_k_accuracy,
        'per_class_accuracy': per_class_acc,
        'class_counts': dict(class_total),
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'sorted_classes': sorted_classes
    }


def generate_confusion_matrix(
    labels: np.ndarray,
    predictions: np.ndarray,
    idx_to_sign: Dict[int, str],
    output_path: Path,
    top_n: int = 50
):
    """
    Generate and save confusion matrix for top N classes.
    
    Args:
        labels: True labels
        predictions: Predicted labels
        idx_to_sign: Mapping from index to sign name
        output_path: Path to save confusion matrix
        top_n: Number of top classes to include (by frequency)
    """
    # Get most frequent classes
    from collections import Counter
    label_counts = Counter(labels)
    top_classes = [cls for cls, _ in label_counts.most_common(top_n)]
    
    # Filter to top classes
    mask = np.isin(labels, top_classes)
    filtered_labels = labels[mask]
    filtered_preds = predictions[mask]
    
    # Create confusion matrix
    cm = confusion_matrix(filtered_labels, filtered_preds, labels=top_classes)
    
    # Normalize
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(20, 16))
    
    # Plot
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=[idx_to_sign.get(i, f'Class_{i}')[:20] for i in top_classes],
        yticklabels=[idx_to_sign.get(i, f'Class_{i}')[:20] for i in top_classes],
        cbar_kws={'label': 'Normalized Accuracy'}
    )
    
    plt.title(f'Confusion Matrix (Top {top_n} Classes)', fontsize=16)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrix saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Detailed Model Evaluation')
    parser.add_argument('--model', type=str, required=True, help='Path to model file')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device (cuda/cpu/auto)')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--top-k', type=int, default=5, help='Top-k accuracy')
    parser.add_argument('--confusion-top-n', type=int, default=50, help='Top N classes for confusion matrix')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    extractor = MediaPipeLandmarkExtractor()
    dataset = GSLDataset(
        frames_dir=FRAMES_DIR,
        dictionary_path=DICTIONARY_PATH,
        sequence_length=16,
        extractor=extractor,
        use_augmentation=False  # No augmentation for evaluation
    )
    
    if len(dataset) == 0:
        print("Error: No data found")
        return
    
    # Create full dataset loader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = load_model(Path(args.model), dataset.num_classes, device)
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_detailed(
        model,
        dataloader,
        device,
        dataset.idx_to_sign,
        top_k=args.top_k
    )
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Top-{args.top_k} Accuracy: {results['top_k_accuracy']:.2f}%")
    print(f"Total Samples: {len(results['labels'])}")
    
    # Per-class statistics
    sorted_classes = results['sorted_classes']
    
    print("\n" + "-"*60)
    print("WORST PERFORMING CLASSES (Bottom 10)")
    print("-"*60)
    for class_idx, acc in sorted_classes[:10]:
        sign = dataset.idx_to_sign.get(class_idx, f'Class_{class_idx}')
        count = results['class_counts'].get(class_idx, 0)
        print(f"  {sign:30s} | Acc: {acc:5.1f}% | Samples: {count:3d}")
    
    print("\n" + "-"*60)
    print("BEST PERFORMING CLASSES (Top 10)")
    print("-"*60)
    for class_idx, acc in sorted_classes[-10:][::-1]:
        sign = dataset.idx_to_sign.get(class_idx, f'Class_{class_idx}')
        count = results['class_counts'].get(class_idx, 0)
        print(f"  {sign:30s} | Acc: {acc:5.1f}% | Samples: {count:3d}")
    
    # Save detailed results
    results_file = output_dir / 'detailed_results.json'
    with open(results_file, 'w') as f:
        json.dump({
            'overall_accuracy': results['overall_accuracy'],
            'top_k_accuracy': results['top_k_accuracy'],
            'per_class_accuracy': {str(k): v for k, v in results['per_class_accuracy'].items()},
            'class_counts': {str(k): v for k, v in results['class_counts'].items()},
            'worst_classes': [
                {'class_idx': int(k), 'sign': dataset.idx_to_sign.get(k, f'Class_{k}'), 'accuracy': float(v)}
                for k, v in sorted_classes[:20]
            ],
            'best_classes': [
                {'class_idx': int(k), 'sign': dataset.idx_to_sign.get(k, f'Class_{k}'), 'accuracy': float(v)}
                for k, v in sorted_classes[-20:][::-1]
            ]
        }, f, indent=2)
    
    print(f"\nDetailed results saved to {results_file}")
    
    # Generate confusion matrix
    if args.confusion_top_n > 0:
        print("\nGenerating confusion matrix...")
        cm_path = output_dir / 'confusion_matrix.png'
        generate_confusion_matrix(
            results['labels'],
            results['predictions'],
            dataset.idx_to_sign,
            cm_path,
            top_n=args.confusion_top_n
        )
    
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()

