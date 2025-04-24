from utils.data_loader import load_data
import torch

def test_cifar10_loading():
    print("Testing CIFAR-10 data loading...")
    
    # Load CIFAR-10
    train_loader, val_loader = load_data(batch_size=64)
    
    # Get a batch of data
    images, labels = next(iter(train_loader))
    
    print(f"\nDataset loaded successfully!")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"\nImage batch shape: {images.shape}")
    print(f"Label batch shape: {labels.shape}")
    print(f"\nImage value range: [{images.min():.3f}, {images.max():.3f}]")
    print(f"Unique labels in batch: {torch.unique(labels).tolist()}")

if __name__ == "__main__":
    test_cifar10_loading() 