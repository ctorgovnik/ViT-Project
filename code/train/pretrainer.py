import torch
from code.train.base_trainer import BaseTrainer


class PreTrainer(BaseTrainer):
    
    @classmethod
    def from_config(cls, config):
        from code.model.vit import ViT
        from code.utils.data_loader import get_cifar10_dataloaders # TODO: change to imagenet
        
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
        train_loader, val_loader = get_cifar10_dataloaders(config, config.train.shuffle)

        return cls(model, optimizer, criterion, device, output_dir, train_loader, val_loader, config.train.epochs)