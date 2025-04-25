import sys
from code.model.vit import ViT
import torch
from code.train.finetuner import FineTuner
from code.train.pretrainer import PreTrainer
from code.config import Config

"""
sample command:
Pretrain on CIFAR10:
python -m code.main -m pretrain -b 128 -e 100 -i 32
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