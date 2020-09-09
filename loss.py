import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2, device='cuda'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs,
                                                      targets,
                                                      reduction='none',
                                                      weight=torch.tensor([.303, .303, .303, .091]))
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.view(-1))
        pt = torch.exp(-BCE_loss).view(-1)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss.view(-1)
        return F_loss.mean()
