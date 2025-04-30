from code.train.base_trainer import BaseTrainer
import torch

class FineTuner(BaseTrainer):

    @classmethod
    def from_config(cls, config):
        from code.model.vit import ViT
        from code.utils.data_loader import get_cifar10_dataloaders, get_cifar100_dataloaders

        # load pretrained weights
        checkpoint = torch.load(config.finetune_cfg.checkpoint)
        model = ViT(**checkpoint["model_config"]) # dump saved model config
        model.load_state_dict(checkpoint["model_state_dict"])


        # replace head
        model.reset_classification_head(num_classes=config.model.num_classes)

        # create optimizer
        optimizer_class = getattr(torch.optim, config.optimizer.name)
        optimizer = optimizer_class(
            model.parameters(),
            lr=config.optimizer.lr,
            weight_decay=config.optimizer.weight_decay
        )

        scheduler_class = getattr(torch.optim.lr_scheduler, config.scheduler.name)
        scheduler = scheduler_class(
            optimizer,
            T_max=config.scheduler.T_max,
            eta_min=config.scheduler.eta_min
        )
        criterion = config.criterion
        device = config.device
        output_dir = config.output_dir
        if config.data_dir == "data/cifar10":
            train_loader, val_loader = get_cifar10_dataloaders(config, config.train.shuffle)
        elif config.data_dir == "data/cifar100":
            train_loader, val_loader = get_cifar100_dataloaders(config, config.train.shuffle)
        else:
            raise ValueError(f"Invalid dataset: {config.data_dir}")

        return cls(model, optimizer, criterion, scheduler, device, output_dir, train_loader, val_loader, config.train.epochs, config)
