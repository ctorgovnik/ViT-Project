import os
import torch

class BaseTrainer:

    def __init__(self, model, optimizer, criterion, device, output_dir, train_loader, val_loader, num_epochs):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.output_dir = output_dir
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.best_val_loss = float('inf')
        self.best_val_acc = 0
    
    def train(self):
        for epoch in range(self.num_epochs):
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate()
            print(f"Epoch {epoch} - Train loss: {train_loss}, Train accuracy: {train_acc}")
            if val_acc > self.best_val_acc:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, train_loss, train_acc, val_loss, val_acc)
                print(f"Saved checkpoint for epoch {epoch} with validation loss: {val_loss}, validation accuracy: {val_acc}")   
            else:
                print(f"Epoch {epoch} - Validation loss: {val_loss}, Validation accuracy: {val_acc}")
    
    def _train_epoch(self):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_loss = train_loss / len(self.train_loader)
        train_acc = 100. * correct / total
        return train_loss, train_acc
    
    def _validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
        val_loss = val_loss / len(self.val_loader)
        val_acc = 100. * correct / total
        return val_loss, val_acc
                
    def _save_checkpoint(self, epoch, train_loss, train_acc, val_loss, val_acc):
        checkpoint_path = os.path.join(self.output_dir, f"checkpoint_epoch_{epoch}.pth")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "loss": {
                "train": train_loss,
                "val": val_loss
            },
            "accuracy": {
                "train": train_acc,
                "val": val_acc
            }
        }, checkpoint_path)
    
        