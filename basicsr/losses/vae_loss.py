import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F
from basicsr.losses.basic_loss import WeightedTVLoss
from basicsr.utils.registry import LOSS_REGISTRY
@LOSS_REGISTRY.register()
class VAELoss(nn.Module):
    def __init__(self,loss_weight=1,reduction='mean'):
        super(VAELoss, self).__init__()
        self.cri_l2 = nn.MSELoss(reduction=reduction)
        self.loss_weight = loss_weight
    def forward(self,output,mu,var,gt):
        #这个是正常使用的loss
        re_loss = self.cri_l2(output,gt)
        kl_loss = 0.5 * torch.sum(-1 - var + mu.pow(2) + var.exp())

        return self.loss_weight*(re_loss+0.00005*kl_loss)
