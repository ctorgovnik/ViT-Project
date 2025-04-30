import torch
from code.train.base_trainer import BaseTrainer


class PreTrainer(BaseTrainer):
    
    @classmethod
    def from_config(cls, config):
        from code.model.vit import ViT
        from code.utils.data_loader import get_imagenet100_dataloaders, get_imagenet1000_dataloaders, get_cifar10_dataloaders
        
        model = ViT(**config.model.model_dump())

        # create optimizer
        optimizer_class = getattr(torch.optim, config.optimizer.name)
        optimizer = optimizer_class(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay
        )

        criterion = config.criterion
        device = config.device
        output_dir = config.output_dir
        if config.data_dir == "data/imagenet-100":
            train_loader, val_loader = get_imagenet100_dataloaders(config, imagenet_path=config.data_dir, shuffle=config.train.shuffle)
        elif config.data_dir == "data/ImageNet-2012":
            train_loader, val_loader = get_imagenet1000_dataloaders(config, imagenet_path=config.data_dir, shuffle=config.train.shuffle)
        elif config.data_dir == "data/cifar10":
            train_loader, val_loader = get_cifar10_dataloaders(config, shuffle=config.train.shuffle)
        else:
            raise ValueError(f"Invalid dataset: {config.data_dir}")

        return cls(model, optimizer, criterion, device, output_dir, train_loader, val_loader, config.train.epochs, config)