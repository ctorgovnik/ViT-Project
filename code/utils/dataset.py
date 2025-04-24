import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, data, labels=None, transform=None, is_cifar10=False):
        """
        Args:
            data: Either a list of image paths or a numpy array of images
            labels: Labels for the images
            transform: Optional transform to be applied
            is_cifar10: Whether the data is from CIFAR-10
        """
        self.data = data
        self.labels = labels
        self.is_cifar10 = is_cifar10
        self.transform = transform or self._get_default_transform()
        
        if not is_cifar10:
            # For ImageNet, get class labels from directory structure
            self.class_to_idx = {}
            self.labels = []
            for path in data:
                class_name = path.split('/')[-2]
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)
                self.labels.append(self.class_to_idx[class_name])
    
    def _get_default_transform(self):
        if self.is_cifar10:
            return transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                  (0.2023, 0.1994, 0.2010))
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                  std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.is_cifar10:
            # For CIFAR-10, data is already a numpy array
            img = self.data[idx]
            # Convert numpy array to PIL Image
            img = Image.fromarray(img)
        else:
            # For ImageNet, data is a list of paths
            img = Image.open(self.data[idx]).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        label = self.labels[idx]
        return img, label

    def get_image_size(self):
        """Returns the size of images in the dataset (assumes square images)"""
        if self.is_cifar10:
            return 32  
        else:
            return 224  
    
    def get_num_classes(self):
        """Returns the number of classes in the dataset"""
        if self.is_cifar10:
            return 10
        else:
            return len(self.class_to_idx) if hasattr(self, 'class_to_idx') else 1000
    
    def get_dataset_size(self):
        """Returns the total number of images in the dataset"""
        return len(self.data)
    
    def get_class_names(self):
        """Returns the names of the classes"""
        if self.is_cifar10:
            return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
        else:
            return list(self.class_to_idx.keys()) if hasattr(self, 'class_to_idx') else None
