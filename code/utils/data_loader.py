import os
import numpy as np
import pickle
from torch.utils.data import DataLoader
from .dataset import ImageDataset, HFDatasetWrapper, simple_collate
from torchvision.datasets import ImageNet
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import CIFAR100
from torchvision import transforms
from datasets import load_dataset
import torch
import ssl

# Disable SSL verification for CIFAR-100 download
ssl._create_default_https_context = ssl._create_unverified_context

def get_imagenet1000_dataloaders(config, imagenet_path = "data/ImageNet-2012", shuffle=True):
    
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std =[0.229, 0.224, 0.225],
    # )

    # train_tf = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    # val_tf = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    # train_ds = ImageNet(root=imagenet_path, split="train", transform=train_tf)
    # val_ds   = ImageNet(root=imagenet_path, split="val",   transform=val_tf)

    ds = load_dataset("ILSVRC/imagenet-1k")
    train_ds = ds["train"]
    val_ds = ds["validation"]

    train_ds.with_format(type="torch", columns=["image", "label"])
    val_ds.with_format(type="torch", columns=["image", "label"])
    
    print(train_ds[0])
    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_and_transform,      # << override default here
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_and_transform,
    )

    return train_loader, val_loader

def get_imagenet100_dataloaders(config, imagenet_path = "data/imagenet100", shuffle=True):
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406],
    #     std =[0.229, 0.224, 0.225],
    # )

    # train_tf = transforms.Compose([
    #     transforms.RandomResizedCrop(224),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    # val_tf = transforms.Compose([
    #     transforms.Resize(256),
    #     transforms.CenterCrop(224),
    #     transforms.ToTensor(),
    #     normalize,
    # ])

    # train_ds = ImageFolder(root=os.path.join(imagenet_path, "train"), transform=train_tf)
    # val_ds   = ImageFolder(root=os.path.join(imagenet_path, "val"),   transform=val_tf)

    ds = load_dataset("clane9/imagenet-100")
    # train_ds = HFDatasetWrapper(ds["train"], image_size=160)
    # val_ds = HFDatasetWrapper(ds["validation"], image_size=160)
    train_ds = ds["train"]
    val_ds = ds["validation"]
    
    train_ds.with_format(type="torch", columns=["image", "label"])
    val_ds.with_format(type="torch", columns=["image", "label"])

    print(train_ds[0])

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_and_transform,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        collate_fn=collate_and_transform,
    )

    return train_loader, val_loader

def get_cifar10_dataloaders(config, shuffle=True):
    """Load either CIFAR-10 or ImageNet dataset."""

    data = load_cifar10_batches(config.data_dir)
    
    
    # Reshape data to (N, H, W, C)
    train_data = data['train_data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = data['test_data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    
    train_dataset = ImageDataset(train_data, data['train_labels'], is_cifar10=True)
    test_dataset = ImageDataset(test_data, data['test_labels'], is_cifar10=True)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )
    return train_loader, test_loader

def get_cifar100_dataloaders(config, data_dir="data/cifar100", shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761],
    )

    train_tf = transforms.Compose([
        # Essential augmentations
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        
        # Moderate color augmentations
        transforms.ColorJitter(
            brightness=0.2,  # Reduced from 0.4
            contrast=0.2,    # Reduced from 0.4
            saturation=0.2,  # Reduced from 0.4
            hue=0.1
        ),
        
        # Simple geometric augmentations
        transforms.RandomRotation(10),  # Reduced from 15
        
        # Convert to tensor and normalize
        transforms.ToTensor(),
        normalize,
        
        # Random erasing with moderate settings
        transforms.RandomErasing(p=0.2)  # Reduced from 0.5
    ])

    val_tf = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    print("loading cifar100")
    train_ds = CIFAR100(
        root=data_dir,
        train=True,
        download=True,
        transform=train_tf
    )

    val_ds = CIFAR100(
        root=data_dir,
        train=False,
        download=True,
        transform=val_tf
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.train.batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader
    


def load_cifar10_batches(data_dir):
    """Load and combine all CIFAR-10 batches."""
    # Load training batches
    train_data = []
    train_labels = []
    
    for i in range(1, 6):  # Load data_batch_1 to data_batch_5
        batch_file = os.path.join(data_dir, f'data_batch_{i}')
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            # Convert to numpy arrays
            data = np.array(batch[b'data'])
            labels = np.array(batch[b'labels'])
            train_data.append(data)
            train_labels.append(labels)
    
    # Combine all training batches
    train_data = np.vstack(train_data)  # Shape: (50000, 3072)
    train_labels = np.concatenate(train_labels)  # Shape: (50000,)
    
    # Load test batch
    test_file = os.path.join(data_dir, 'test_batch')
    with open(test_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        test_data = np.array(batch[b'data'])  # Shape: (10000, 3072)
        test_labels = np.array(batch[b'labels'])  # Shape: (10000,)
    
    # Load label names
    meta_file = os.path.join(data_dir, 'batches.meta')
    with open(meta_file, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
        label_names = [name.decode('utf-8') for name in meta[b'label_names']]

    return {
        'train_data': train_data,
        'train_labels': train_labels,
        'test_data': test_data,
        'test_labels': test_labels,
        'label_names': label_names
    }

# use for imagenet and cifar100
def gather_image_paths(data_dir, mode):
    image_paths = {}
    if mode == "train":
        data_dir_train = os.path.join("data", data_dir, "train")
        data_dir_val = os.path.join("data", data_dir, "val")

        for root, dirs, files in os.walk(data_dir_train):
            for file in files:
                if file.endswith(".JPEG"):
                    image_paths['train'].append(os.path.join(root, file))

        for root, dirs, files in os.walk(data_dir_val):
            for file in files:
                if file.endswith(".JPEG"):
                    image_paths['val'].append(os.path.join(root, file))
                
    if mode == "test":
        for root, dirs, files in os.walk(os.path.join("data", data_dir)):
            for file in files:
                if file.endswith(".JPEG"):
                    image_paths['test'].append(os.path.join(root, file))

    return image_paths

def collate_and_transform(batch):
    def ensure_rgb(img):
        return img.convert("RGB")
    transform = transforms.Compose([
        ensure_rgb,  # Convert to RGB if not already
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Add normalization if needed
    ])
    images = [transform(item['image']) for item in batch]
    labels = [item['label'] for item in batch]
    return torch.stack(images), torch.tensor(labels)

