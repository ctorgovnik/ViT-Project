import argparse


class Config:

    def __init__(self, args):
        params = self._parse_args(args)
        self.batch_size = params.batch_size
        self.shuffle = params.shuffle
        self.mode = params.mode
        self.data_dir = params.data_dir
        self.dataset = params.dataset
        self.patch_size = params.patch_size
        self.dim = params.dim
        self.depth = params.depth
        self.heads = params.heads
        self.mlp_dim = params.mlp_dim
        self.dropout = params.dropout
        self.learning_rate = params.learning_rate
        self.epochs = params.epochs

    def _parse_args(self, args):
        parser = argparse.ArgumentParser()
        parser.add_argument('-b', "--batch_size", type=int, default=32)
        parser.add_argument('-s', "--shuffle", type=bool, default=True)
        parser.add_argument('-m', "--mode", type=str, default="train")
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
        
        return parser.parse_args()