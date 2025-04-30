import argparse
import torch
from pydantic import BaseModel
from typing import Optional
import torch.nn as nn

class ModelConfig(BaseModel):
    patch_size:   int = 16
    image_size:   int = 32 # 32x32 image for cifar10, 224x224 image for imagenet
    dim:          int = 768
    depth:        int = 12
    mlp_dim:      int = 3072
    heads:        int = 12
    num_classes:  int = 1000
    dropout:      float = 0.1

class TrainConfig(BaseModel):
    epochs:     int = 100
    batch_size: int = 4096
    shuffle:    bool = True

class OptimConfig(BaseModel):
    name: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 0.1

class FinetuneConfig(BaseModel):
    checkpoint: Optional[str] = None

class SchedulerConfig(BaseModel):
    name: str = "CosineAnnealingLR"
    T_max: int = 200  # Your total epochs
    eta_min: float = 1e-6


class Config(BaseModel):
    model_config = {
        "arbitrary_types_allowed": True
    }
    
    model: ModelConfig
    model_name:   str = "pretrained_vit"
    train: TrainConfig
    scheduler: SchedulerConfig
    finetune_cfg: Optional[FinetuneConfig] = None
    device: str = "cpu"
    # checkpoint: str = "./results/checkpoint.pth"
    data_dir: str = "./data"
    mode: str = "pretrain"
    optimizer: OptimConfig
    criterion: torch.nn.Module
    output_dir: str = "results"
    @classmethod
    def parse_args(cls, args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-b', "--batch_size", type=int, default=32)
        parser.add_argument('-s', "--shuffle", type=bool, default=True)
        parser.add_argument('-m', "--mode", type=str, default="pretrain")
        parser.add_argument('-dir', "--data_dir", type=str, default="data")
        parser.add_argument('-ds', "--dataset", type=str, default="cifar10")
        parser.add_argument('-ps', "--patch_size", type=int, default=16)
        parser.add_argument('-dim', "--dim", type=int, default=128)
        parser.add_argument('-hd', "--heads", type=int, default=4)
        parser.add_argument('-d', "--depth", type=int, default=12)
        parser.add_argument('-md', "--mlp_dim", type=int, default=512)
        parser.add_argument('-do', "--dropout", type=float, default=0.1)
        parser.add_argument('-e', "--epochs", type=int, default=100)
        parser.add_argument('-od', "--output_dir", type=str, default="results")
        parser.add_argument('--optimizer', type=str, default="AdamW")
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--weight_decay', type=float, default=0.1)
        parser.add_argument('--criterion', type=str, default="CrossEntropyLoss")
        parser.add_argument('-i', '--image_size', type=int, default=32)
        parser.add_argument('-nc', "--num_classes", type=int, default=10)
        parser.add_argument('-mn', "--model_name", type=str, default="pretrained_vit")
        parser.add_argument('-ckpt', "--checkpoint", type=str, default="checkpoints/pretrained_vit.pth")

        args = parser.parse_args(args)
        
        criterion_class = getattr(torch.nn, args.criterion)
        criterion = criterion_class()
        
        return cls(
            model=ModelConfig(
                patch_size=args.patch_size,
                image_size=args.image_size,
                dim=args.dim,
                depth=args.depth,
                mlp_dim=args.mlp_dim,
                heads=args.heads,
                num_classes=args.num_classes,
                dropout=args.dropout,
            ),
            scheduler=SchedulerConfig(
                name=args.scheduler,
                T_max=args.epochs,
                eta_min=args.lr * 0.01
            ),
            train=TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                shuffle=args.shuffle
            ),
            optimizer=OptimConfig(
                name=args.optimizer,
                lr=args.lr,
                weight_decay=args.weight_decay
            ),
            finetune_cfg=FinetuneConfig(
                checkpoint=args.checkpoint
            ),
            device="cpu",
            # checkpoint=f"{args.output_dir}/checkpoint.pth",
            data_dir=args.data_dir,
            mode=args.mode,
            criterion=criterion,
            output_dir=args.output_dir,
            model_name=args.model_name
        )

    @classmethod
    def from_config(cls, config):
        # load pretrained weights
        checkpoint = torch.load(config.finetune_cfg.checkpoint)
        print("Loaded checkpoint with config:", checkpoint["model_config"])  # Debug print
        
        model = ViT(**checkpoint["model_config"])
        print("Model dim:", model.dim)  # Debug print
        
        model.load_state_dict(checkpoint["model_state_dict"])
        model.reset_classification_head(num_classes=config.model.num_classes)


class ViT(nn.Module):
    def __init__(self, image_size, patch_size, dim, depth, heads, mlp_dim, num_classes, dropout=0.1):
        super().__init__()
        self.dim = dim  # Store dim as instance variable
        # ... rest of initialization ...

    def reset_classification_head(self, num_classes):
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),  # Now we can access self.dim
            nn.Linear(self.dim, num_classes)
        )

    