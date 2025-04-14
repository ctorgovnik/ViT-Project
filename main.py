import sys
from config import Config
from utils.data_loader import load_data
from model.vit import ViT
from train import train
import torch

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}\n')
    
    config = Config(args)
    print(config)

    if config.mode == "train":  
        train_loader, val_loader = load_data(config.batch_size, config.shuffle, config.mode, config.data_dir, config.dataset)
        
        num_classes = train_loader.dataset.get_num_classes()
        image_size = train_loader.dataset.get_image_size()

        model = ViT(image_size, config.patch_size, num_classes, config.dim, config.depth, config.heads, config.mlp_dim, config.dropout)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
        criterion = torch.nn.CrossEntropyLoss()
        train(model, train_loader, val_loader, optimizer, scheduler, criterion, config.epochs, device)

    elif config.mode == "test":
        test_loader = load_data(config.batch_size, config.shuffle, config.mode, config.data_dir, config.dataset)
        # TODO: Test the model

if __name__ == "__main__":
    main(sys.argv[1:])