import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=.25, gamma=2, device='cuda'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha]).to(device)
        self.gamma = gamma
        self.device = device

    def forward(self, inputs, targets):
        # BCE_loss = F.binary_cross_entropy_with_logits(inputs,
        #                                               targets.type(torch.FloatTensor).to(self.device),
        #                                               reduction='none',
        #                                               weight=torch.tensor([.303, .303, .303, .091]).to(self.device))
        loss = None

        for batch_x, batch_y in zip(inputs, targets):
            negative = list()
            for idx in range(batch_x.shape[0]):
                x, y = batch_x[idx], batch_y[idx]
                if torch.equal(y, torch.tensor([0, 0, 0, 1], device=self.device)) is True:
                    negative.append(idx)
                    continue

                _loss = F.cross_entropy(x.unsqueeze(0), torch.tensor([torch.argmax(y, dim=-1)], device=self.device))
                loss = loss + _loss if loss is not None else _loss

            neg_ids = np.random.choice(negative, min(negative.__len__(), batch_x.shape[0] - negative.__len__()))
            for idx in neg_ids:
                x, y = batch_x[idx], batch_y[idx]
                _loss = F.cross_entropy(x.unsqueeze(0), torch.tensor([torch.argmax(y, dim=-1)], device=self.device))
                loss = loss + _loss if loss is not None else _loss

        return loss.mean()
        # targets = targets.type(torch.long)
        # at = self.alpha.gather(0, targets.view(-1))
        # pt = torch.exp(-BCE_loss).view(-1)
        # F_loss = at * (1 - pt) ** self.gamma * BCE_loss.view(-1)
        # return F_loss.mean()
