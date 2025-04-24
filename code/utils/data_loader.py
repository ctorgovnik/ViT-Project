import os
import numpy as np
import pickle
from torch.utils.data import DataLoader
from .dataset import ImageDataset

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

def load_data(batch_size, shuffle=True, mode="train", data_dir="data", dataset="cifar10"):
    """Load either CIFAR-10 or ImageNet dataset."""
    if dataset.lower() == "cifar10":
        # CIFAR-10
        data = load_cifar10_batches(os.path.join(data_dir, "cifar10"))
       
        # Reshape data to (N, H, W, C)
        train_data = data['train_data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        test_data = data['test_data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        
        train_dataset = ImageDataset(train_data, data['train_labels'], is_cifar10=True)
        test_dataset = ImageDataset(test_data, data['test_labels'], is_cifar10=True)
        
        if mode == "train":
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4,
                pin_memory=True
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4,
                pin_memory=True
            )
            return train_loader, test_loader
        else:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=4,
                pin_memory=True
            )
            return test_loader
    else:
        # ImageNet
        image_paths = gather_image_paths(data_dir, mode)

        if mode == "train":
            train_dataset = ImageDataset(image_paths['train'])
            val_dataset = ImageDataset(image_paths['val'])

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

            return train_loader, val_loader
        
        elif mode == "test":
            test_dataset = ImageDataset(image_paths['test'])
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
            return test_loader


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

