# ResNet-UNet for Semantic Segmentation on PASCAL VOC

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance semantic segmentation model combining ResNet34 encoder with U-Net decoder architecture, achieving **79.38% mIoU** on PASCAL VOC 2012 validation set.

## Key Features

- **Hybrid Architecture**: ResNet34 pretrained encoder + custom U-Net decoder with skip connections
- **Efficient Training**: Converges in ~55 epochs with ReduceLROnPlateau scheduler
- **Production Ready**: Includes data augmentation, model checkpointing, and comprehensive visualization
- **Memory Optimized**: Batch normalization and efficient decoder blocks for stable training
- **Reproducible**: Fixed seeds for consistent results across runs

## Performance Metrics

| Metric | Score |
|--------|-------|
| Best Validation mIoU | **79.38%** (Epoch 55) |
| Final Training Loss | 0.2075 |
| Model Parameters | 26.7M |
| Training Time | ~100 epochs on single GPU |

## Architecture Overview

```
Input (3×256×256)
    ↓
┌─────────────────┐
│ ResNet34 Encoder│  (Pretrained on ImageNet)
│  - Layer 1: 64  │
│  - Layer 2: 128 │
│  - Layer 3: 256 │
│  - Layer 4: 512 │
└─────────────────┘
    ↓
┌─────────────────┐
│   Bridge (512)  │
└─────────────────┘
    ↓
┌─────────────────┐
│  U-Net Decoder  │  (Skip connections from encoder)
│  - Up1: 256     │  ← Concatenate with encoder4
│  - Up2: 128     │  ← Concatenate with encoder3
│  - Up3: 64      │  ← Concatenate with encoder2
│  - Up4: 32      │  ← Concatenate with encoder1
└─────────────────┘
    ↓
Output (21×256×256)  [21 classes]
```

## Quick Start

### Prerequisites

```bash
pip install torch torchvision matplotlib numpy pillow tqdm
```

### Training

```python
python train.py
```

**Key Hyperparameters:**
- Batch Size: 8
- Learning Rate: 1e-4 (with ReduceLROnPlateau)
- Optimizer: Adam (weight decay: 1e-5)
- Image Size: 256×256
- Epochs: 100

### Inference

```python
# Load trained model
model = ResNetUNet(num_classes=21).to(device)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Run visualization
visualize_results(model, val_loader)
```

## Project Structure

```
├── train.py                 # Main training script
├── best_model.pth          # Best checkpoint (79.38% mIoU)
├── diverse_results.png     # Visualization outputs
└── data/                   # PASCAL VOC dataset (auto-downloaded)
```

##  Customization Guide

### 1. Change Dataset

Replace `VOCDataset` with your custom dataset:

```python
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        # Your implementation
        pass
    
    def __getitem__(self, idx):
        return image, mask  # mask shape: [H, W] with class indices
```

### 2. Modify Architecture

**Change Encoder:**
```python
# In ResNetUNet.__init__()
resnet = models.resnet50(pretrained=True)  # Use ResNet50
# Or use other backbones:
# efficientnet, mobilenet, etc.
```

**Adjust Number of Classes:**
```python
model = ResNetUNet(num_classes=YOUR_NUM_CLASSES)
```

### 3. Training Configuration

```python
# In main execution block
BATCH_SIZE = 16          # Increase if GPU memory allows
LR = 5e-5                # Lower for fine-tuning
TARGET_SIZE = 512        # Higher resolution (requires more memory)
num_epochs = 50          # Reduce for faster experimentation
```

### 4. Data Augmentation

Modify `VOCDataset.get_transform()`:

```python
transforms_list = [
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomResizedCrop(target_size, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(0.5),
    # ... existing transforms
]
```

## Training Progress

The model shows clear convergence with minimal overfitting:

```
Epoch [1/100]   - Loss: 2.5461 - Val IoU: 0.4472
Epoch [10/100]  - Loss: 1.0291 - Val IoU: 0.7261
Epoch [27/100]  - Loss: 0.6010 - Val IoU: 0.7635
Epoch [40/100]  - Loss: 0.3687 - Val IoU: 0.7903
Epoch [55/100]  - Loss: 0.2380 - Val IoU: 0.7938 ← Best
Epoch [100/100] - Loss: 0.2075 - Val IoU: 0.7914
```

**Key Observations:**
- Rapid initial improvement (Epochs 1-30)
- Plateau detection triggers LR reduction
- Stable performance after Epoch 55

## Visualization Features

The `visualize_results()` function generates a 3×5 grid showing:
- **Row 1**: Original RGB images
- **Row 2**: Ground truth segmentation masks
- **Row 3**: Model predictions

Color coding follows PASCAL VOC standard palette (21 classes + border).

## Troubleshooting

### Out of Memory Error
```python
# Reduce batch size
BATCH_SIZE = 4

# Or reduce image size
TARGET_SIZE = 128
```

### Model Not Converging
```python
# Try different learning rate
LR = 5e-5

# Add stronger augmentation
# Increase weight decay
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
```

### Poor Performance on Custom Dataset
- Ensure mask values are in range [0, num_classes-1]
- Set correct `ignore_index` in loss function
- Verify data normalization matches pretrained weights

## Technical Details

### Loss Function
```python
CrossEntropyLoss(ignore_index=21)  # Ignores border pixels (class 255 → 21)
```

### Evaluation Metric
Per-class Intersection over Union (IoU), averaged across all classes:
```
IoU = |Prediction ∩ Ground Truth| / |Prediction ∪ Ground Truth|
mIoU = mean(IoU_class_0, IoU_class_1, ..., IoU_class_20)
```

### Skip Connections
Decoder blocks concatenate upsampled features with corresponding encoder features, preserving spatial details lost during downsampling.

## Potential Improvements

1. **Attention Mechanisms**: Add CBAM or spatial attention modules
2. **Deeper Encoders**: Try ResNet101 or EfficientNet-B4
3. **Advanced Augmentation**: Implement MixUp or CutMix
4. **Post-processing**: Apply CRF (Conditional Random Fields) for smoother boundaries
5. **Multi-scale Training**: Train on different resolutions
6. **Ensemble Models**: Combine predictions from multiple checkpoints

## Citation

If you use this implementation, please cite:

```bibtex
@misc{resnet_unet_voc,
  author = {Your Name},
  title = {ResNet-UNet for Semantic Segmentation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/resnet-unet}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request


---

⭐ **Star this repository if you find it helpful!**
