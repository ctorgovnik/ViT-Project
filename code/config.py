import argparse

from pydantic import BaseModel

class ModelConfig(BaseModel):
    patch_size:   int = 16
    image_size:   int = 224
    embed_dim:    int = 768
    depth:        int = 12
    mlp_dim:      int = 3072
    num_heads:    int = 12
    num_classes:  int = 1000
    dropout:      float = 0.1

class OptimConfig(BaseModel):
    name:     str   = "AdamW"
    lr:       float = 1e-3
    weight_decay: float = 0.1

class TrainConfig(BaseModel):
    epochs:     int = 100
    batch_size: int = 4096
    shuffle:    bool = True
class Config(BaseModel):
    model: ModelConfig
    optim: OptimConfig
    train: TrainConfig
    device: str = "cpu"
    checkpoint: str = "./results/checkpoint.pth"
    data_dir: str = "./data"
    mode: str = "pretrain"

    @classmethod
    def from_args(cls, args):
        params = cls._parse_args(args)
        return cls(**params)

    def _parse_args(self, args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-b', "--batch_size", type=int, default=32)
        parser.add_argument('-s', "--shuffle", type=bool, default=True)
        parser.add_argument('-m', "--mode", type=str, default="pretrain")
        parser.add_argument('-dir', "--data_dir", type=str, default="data")
        parser.add_argument('-ds', "--dataset", type=str, default="cifar10")
        parser.add_argument('-ps', "--patch_size", type=int, default=16)
        parser.add_argument('-dim', "--dim", type=int, default=128)
        parser.add_argument('-d', "--depth", type=int, default=12)
        parser.add_argument('-hd', "--heads", type=int, default=4)
        parser.add_argument('-md', "--mlp_dim", type=int, default=512)
        parser.add_argument('-do', "--dropout", type=float, default=0.1)
        parser.add_argument('-lr', "--learning_rate", type=float, default=0.001)
        parser.add_argument('-e', "--epochs", type=int, default=10)
        parser.add_argument('-od', "--output_dir", type=str, default="results")

        return parser.parse_args()


    