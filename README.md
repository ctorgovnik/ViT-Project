# Vision Transformer and ResNet Implementation

## Introduction
This repository implements Vision Transformer (ViT) and ResNet models for image classification, focusing on their performance through pretraining and finetuning. We explore how these architectures scale with different dataset sizes, using both custom ViT models and pretrained variants. The project reproduces the findings from "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al., while also comparing its performance with ResNet architectures.

## Chosen Result
We focus on reproducing the image classification performance of ViT on CIFAR-10 dataset, pretrained on CIFAR-100 dataset, comparing it with ResNet18 as a baseline. While the original paper used large-scale datasets like ImageNet-21k for pretraining, we demonstrate the model's behavior on smaller datasets, showing how ViT's performance scales with data size.

## GitHub Contents
```
.
├── code/
│   ├── main.py           # Main training script
│   ├── config.py         # Configuration settings
│   ├── test.py          # Testing utilities
│   ├── model/
│   │   └── vit.py       # Vision Transformer implementation
│   ├── train/
│   │   ├── base_trainer.py    # Base training class
│   │   ├── resnet.py          # ResNet training implementation
│   │   ├── finetuner.py       # Finetuning functionality
│   │   └── pretrainer.py      # Pretraining functionality
│   └── utils/
│       ├── data_loader.py     # Data loading utilities
│       ├── dataset.py         # Dataset implementations
│       ├── augmentations.py   # Data augmentation pipeline
│       └── loss.py           # Loss function implementations
├── data/                 # Dataset storage
├── checkpoints/          # Saved model checkpoints
├── results/             # Training results and metrics
├── report/              # Project documentation
└── poster/              # Project presentation materials
```

## Re-implementation Details
- **Models**: 
  - Custom ViT (~1M parameters) pretrained on CIFAR-100
  - Mini ViT (deit-tiny-patch16-224, ~5M parameters) pretrained on ImageNet-1K
  - Base ViT (~86.5M parameters) pretrained on ImageNet-1K
  - ResNet18 (~11.6M parameters) pretrained on both CIFAR-100 and ImageNet-1K
- **Datasets**: 
  - CIFAR-10 for finetuning and evaluation
  - CIFAR-100 for small-scale pretraining
  - ImageNet-1K for large-scale pretraining (using pretrained models)
- **Tools**: PyTorch, torchvision, HuggingFace
- **Evaluation**: Classification accuracy on CIFAR-10 after finetuning
- **Modifications**: 
  - Scaled down architecture for smaller datasets
  - Comprehensive data augmentation pipeline
  - Flexible configuration system
  - Mixed approach using both custom and pretrained models

## Reproduction Steps
1. Clone the repository:
```bash
git clone https://github.com/ctorgovnik/ViT-Project.git
cd ViT-Project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run training:
```bash
# Pretraining ViT on CIFAR-100
python -m code.main -m pretrain -b 128 -e 100 -i 32 -nc 100 -dir data/cifar100 --model_name pretrained_vit

# Pretraining ViT on ImageNet-100
python -m code.main -m pretrain -b 256 -e 200 -i 224 -nc 100 -dir data/imagenet-100 --model_name pretrained_vit

# Finetuning ViT on CIFAR-10
python -m code.main -m finetune -b 128 -e 100 -i 32 -nc 10 --mlp_dim 1024 -dir data/cifar10 -ckpt checkpoints/pretrained_vit.pth --model_name finetune_vit

# Pretraining ResNet on CIFAR-100
python -m code.main -m resnet -b 128 -e 200 -i 32 -nc 100 -dir data/cifar100 --model_name pretrained_resnet
```

### Command Line Options
| Option | Description | Default | Example |
|--------|-------------|---------|---------|
| `-m, --mode` | Training mode (pretrain/finetune/resnet) | pretrain | `-m pretrain` |
| `-b, --batch_size` | Batch size for training | 32 | `-b 128` |
| `-e, --epochs` | Number of training epochs | 100 | `-e 200` |
| `-i, --image_size` | Input image size | 32 | `-i 224` |
| `-nc, --num_classes` | Number of output classes | 10 | `-nc 100` |
| `-dir, --data_dir` | Dataset directory | data | `-dir data/cifar100` |
| `-ps, --patch_size` | ViT patch size | 16 | `-ps 16` |
| `-dim, --dim` | ViT embedding dimension | 128 | `-dim 768` |
| `-hd, --heads` | Number of attention heads | 4 | `-hd 12` |
| `-d, --depth` | Number of transformer layers | 12 | `-d 12` |
| `-md, --mlp_dim` | MLP dimension in transformer | 512 | `-md 3072` |
| `-do, --dropout` | Dropout rate | 0.1 | `-do 0.1` |
| `--optimizer` | Optimizer name | AdamW | `--optimizer AdamW` |
| `--lr` | Learning rate | 1e-3 | `--lr 1e-4` |
| `--weight_decay` | Weight decay | 0.1 | `--weight_decay 0.01` |
| `-ckpt, --checkpoint` | Checkpoint path for finetuning | checkpoints/pretrained_vit.pth | `-ckpt path/to/checkpoint.pth` |
| `-mn, --model_name` | Name for saving model | pretrained_vit | `--model_name my_model` |

**Requirements**:
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM
- 30GB+ disk space for datasets

## Results/Insights
Our implementation results demonstrate the scaling behavior of ViT models with different pretraining datasets:

| Model | # Parameters | Pretraining Dataset | CIFAR-10 Accuracy |
|-------|-------------|---------------------|-------------------|
| ViT (OC) | 1M | CIFAR-100 | 65.00% |
| ResNet-18 | 11.6M | CIFAR-100 | 72.36% |
| Mini ViT | 5M | ImageNet-1K | 87.33% |
| Base ViT | 86.5M | ImageNet-1K | 94.64% |
| ResNet-18 | 11.6M | ImageNet-1K | 79.73% |

The results show that while ResNet outperforms ViT on smaller datasets (CIFAR-100), ViT models significantly improve with larger pretraining datasets (ImageNet-1K). Notably, the Base ViT achieves the highest accuracy despite having more parameters, while the Mini ViT outperforms ResNet-18 even with fewer parameters when both are pretrained on ImageNet-1K.

## Conclusion
This implementation successfully demonstrates the feasibility of using transformer architectures for image classification tasks. The comparison between ViT and ResNet provides valuable insights into the trade-offs between these architectures, particularly in the context of smaller datasets.

## References
1. Ashish Vaswani et al. "Attention is All You Need". In: Advances in Neural Information Processing Systems. Vol. 30. Curran Associates, Inc., 2017.
2. Alexey Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale". In: International Conference on Learning Representations (ICLR). arXiv:2010.11929. 2021.
3. Alex Krizhevsky. Learning multiple layers of features from tiny images. Tech. rep. University of Toronto, 2009. url: https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf.
4. Facebook AI. deit-tiny-patch16-224 on HuggingFace. https://huggingface.co/facebook/deit-tiny-patch16-224. 2024.
5. PyTorch Team. Torchvision Models. https://pytorch.org/vision/main/models.html. 2024.

## Acknowledgements
This project was developed as part of the Deep Learning course at Cornell University. Special thanks to the course instructors and teaching assistants for their guidance and support.



