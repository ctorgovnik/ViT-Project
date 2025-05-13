# import torch
# import torchvision.transforms as transforms
# from torchvision.transforms import functional as F
# import random
# import numpy as np
# from PIL import Image, ImageEnhance, ImageOps

# class AutoAugment:
#     def __init__(self, policy='cifar10'):
#         self.policy = policy
#         self.transforms = self._get_policy()

#     def _get_policy(self):
#         if self.policy == 'cifar10':
#             return [
#                 (transforms.RandomHorizontalFlip(), 0.5),
#                 (transforms.RandomCrop(32, padding=4), 1.0),
#                 (transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), 0.8),
#                 (transforms.RandomRotation(15), 0.3),
#                 (transforms.RandomAffine(0, translate=(0.1, 0.1)), 0.3),
#                 (transforms.RandomPerspective(distortion_scale=0.2), 0.3),
#             ]
#         return []

#     def __call__(self, img):
#         for transform, p in self.transforms:
#             if random.random() < p:
#                 img = transform(img)
#         return img

# class CutMix:
#     def __init__(self, alpha=1.0):
#         self.alpha = alpha

#     def __call__(self, batch):
#         images, targets = batch
#         batch_size = images.size(0)
        
#         # Generate random bounding box
#         lam = np.random.beta(self.alpha, self.alpha)
#         rand_index = torch.randperm(batch_size)
        
#         # Get random box coordinates
#         bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        
#         # Apply cutmix
#         images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        
#         # Adjust lambda to exactly match pixel ratio
#         lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
#         targets_a, targets_b = targets, targets[rand_index]
#         return images, targets_a, targets_b, lam

#     def _rand_bbox(self, size, lam):
#         W = size[2]
#         H = size[3]
#         cut_rat = np.sqrt(1. - lam)
#         cut_w = np.int(W * cut_rat)
#         cut_h = np.int(H * cut_rat)

#         # uniform
#         cx = np.random.randint(W)
#         cy = np.random.randint(H)

#         bbx1 = np.clip(cx - cut_w // 2, 0, W)
#         bby1 = np.clip(cy - cut_h // 2, 0, H)
#         bbx2 = np.clip(cx + cut_w // 2, 0, W)
#         bby2 = np.clip(cy + cut_h // 2, 0, H)

#         return bbx1, bby1, bbx2, bby2

# class RandomErasing:
#     def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
#         self.p = p
#         self.scale = scale
#         self.ratio = ratio

#     def __call__(self, img):
#         if random.random() < self.p:
#             # Get image dimensions
#             img_c, img_h, img_w = img.shape
            
#             # Calculate area
#             area = img_h * img_w
            
#             # Calculate target area
#             target_area = random.uniform(*self.scale) * area
#             aspect_ratio = random.uniform(*self.ratio)
            
#             # Calculate dimensions
#             h = int(round((target_area * aspect_ratio) ** 0.5))
#             w = int(round((target_area / aspect_ratio) ** 0.5))
            
#             if h < img_h and w < img_w:
#                 # Get random position
#                 top = random.randint(0, img_h - h)
#                 left = random.randint(0, img_w - w)
                
#                 # Erase
#                 img[:, top:top + h, left:left + w] = torch.randn((img_c, h, w))
        
#         return img

# class Mixup:
#     def __init__(self, alpha=0.2):
#         self.alpha = alpha

#     def __call__(self, batch):
#         images, targets = batch
#         batch_size = images.size(0)
        
#         # Generate mixing coefficients
#         lam = np.random.beta(self.alpha, self.alpha)
#         rand_index = torch.randperm(batch_size)
        
#         # Mix images
#         mixed_images = lam * images + (1 - lam) * images[rand_index]
        
#         # Mix targets
#         targets_a, targets_b = targets, targets[rand_index]
        
#         return mixed_images, targets_a, targets_b, lam

# def get_train_transforms():
#     return transforms.Compose([
#         # Basic augmentations
#         transforms.RandomCrop(32, padding=4),
#         transforms.RandomHorizontalFlip(p=0.5),
#         transforms.RandomVerticalFlip(p=0.2),
        
#         # Color augmentations
#         transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
#         transforms.RandomAutocontrast(p=0.5),
#         transforms.RandomEqualize(p=0.5),
#         transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
        
#         # Geometric augmentations
#         transforms.RandomRotation(15),
#         transforms.RandomAffine(0, translate=(0.1, 0.1)),
#         transforms.RandomPerspective(distortion_scale=0.2),
        
#         # Advanced augmentations
#         transforms.RandomSolarize(threshold=192.0, p=0.2),
#         transforms.RandomPosterize(bits=4, p=0.2),
#         transforms.RandomGrayscale(p=0.2),
        
#         # Convert to tensor and normalize
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        
#         # Random erasing
#         transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3))
#     ])

# def get_val_transforms():
#     return transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
#     ]) 