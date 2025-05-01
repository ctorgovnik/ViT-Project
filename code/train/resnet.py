from code.train.base_trainer import BaseTrainer
from torchvision import models
import torch
from code.utils.data_loader import get_cifar10_dataloaders, get_cifar100_dataloaders

class ResNetTrainer(BaseTrainer):

    @classmethod
    def from_config(cls, config):

        resnet = models.resnet18(pretrained=False)

        num_features = resnet.fc.in_features
        resnet.fc = torch.nn.Linear(num_features, config.model.num_classes)
        
        optimizer_class = getattr(torch.optim, config.optimizer.name)
        optimizer = optimizer_class(
            resnet.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay
        )

        scheduler_class = getattr(torch.optim.lr_scheduler, config.scheduler.name)
        scheduler = scheduler_class(
            optimizer,
            T_max=config.scheduler.T_max,
            eta_min=config.scheduler.eta_min
        )

        if config.data_dir == "data/cifar10":
            train_loader, val_loader = get_cifar10_dataloaders(config, config.train.shuffle)
        elif config.data_dir == "data/cifar100":
            train_loader, val_loader = get_cifar100_dataloaders(config)
        else:
            raise ValueError(f"Invalid dataset: {config.data_dir}")

        criterion = config.criterion
        device = config.device
        output_dir = config.output_dir

        return cls(resnet, optimizer, criterion, scheduler, device, output_dir, 
                  train_loader, val_loader, config.train.epochs, config)