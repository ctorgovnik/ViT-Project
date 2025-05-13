import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1, reduction='mean'):
        super().__init__()
        self.smoothing = smoothing
        self.reduction = reduction

    def forward(self, x, target):
        n_classes = x.size(-1)
        
        # Convert target to one-hot
        with torch.no_grad():
            target_one_hot = torch.zeros_like(x)
            target_one_hot.scatter_(1, target.unsqueeze(1), 1)
            
            # Apply label smoothing
            target_one_hot = target_one_hot * (1 - self.smoothing) + self.smoothing / n_classes
        
        # Compute cross entropy
        log_prob = F.log_softmax(x, dim=-1)
        loss = -(target_one_hot * log_prob).sum(dim=-1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss 