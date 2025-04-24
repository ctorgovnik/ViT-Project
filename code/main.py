import sys
from config import Config
from model.vit import ViT
from train import train
import torch
from train.finetuner import Finetuner
from train.pretrainer import Pretrainer

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}\n')
    
    config = Config(args)
    
    if config.mode == "finetune":
        trainer = Finetuner.from_config(config)
    else:
        trainer = Pretrainer.from_config(config)
    
    trainer.train()


if __name__ == "__main__":
    main(sys.argv[1:])