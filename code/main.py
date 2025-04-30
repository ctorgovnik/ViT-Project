import sys
from code.model.vit import ViT
import torch
from code.train.finetuner import FineTuner
from code.train.pretrainer import PreTrainer
from code.config import Config

"""
sample command:
Pretrain on Imagenet100:
python -m code.main -m pretrain -b 128 -e 100 -i 224 -nc 100 -dir data/imagenet-100 --model_name pretrained_vit

Pretrain on ImageNet-1000:
python -m code.main -m pretrain -b 128 -e 100 -i 224 -nc 1000 -dir data/ImageNet-2012

Finetune on CIFAR10:
python -m code.main -m finetune -b 128 -e 100 -i 32 -nc 10 -dir data/cifar10 -ckpt checkpoints/pretrained_vit.pth --model_name finetune_vit
"""

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}\n')
    
    config = Config.parse_args(args)
    
    if config.mode == "finetune":
        trainer = FineTuner.from_config(config)
    else:
        trainer = PreTrainer.from_config(config)
    
    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])