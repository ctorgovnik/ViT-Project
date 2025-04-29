from code.train.base_trainer import BaseTrainer
import torch

class FineTuner(BaseTrainer):

    @classmethod
    def from_config(cls, config):
        from code.model.vit import ViT
        from code.utils.data_loader import get_cifar10_dataloaders

        # load pretrained weights
        model = ViT(**config.model.model_dump())
        checkpoint = torch.load(config.checkpoint["model_state_dict"])
        model.load_state_dict(checkpoint)

        # replace head
        model.reset_classification_head(num_classes=config.model.num_classes)

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
