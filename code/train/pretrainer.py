
from code.train.base_trainer import BaseTrainer

class PreTrainer(BaseTrainer):
    
    @classmethod
    def from_config(cls, config):
        from code.model.vit import ViT
        from code.data.imagenet import get_imagenet_dataloaders
        
        model = ViT(**config.model.model_dump())
        optimizer = config.optimizer
        criterion = config.criterion
        device = config.device
        output_dir = config.output_dir
        train_loader, val_loader = get_imagenet_dataloaders(config.train.batch_size, config.train.shuffle)

        return cls(model, optimizer, criterion, device, output_dir, train_loader, val_loader)