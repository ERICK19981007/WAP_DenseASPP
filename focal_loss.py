from torch import nn
import torch.nn.functional as F
import torch


class Focal_loss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25):
        super(Focal_loss, self).__init__()
        self.gamma = 2
        self.alpha = 0.25
        self.elipson = 0.000001

    def forward(self, x, y):
        aa = - self.alpha * (1 - x) ** self.gamma * y * torch.log(x)
        bb = - (1 - self.alpha) * x ** self.gamma * (1 - y) * torch.log(1 - x)
        loss = aa + bb
        return loss
