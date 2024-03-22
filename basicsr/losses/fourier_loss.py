import math
import torch
from torch import autograd as autograd
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY
@LOSS_REGISTRY.register()
class FFTLoss(nn.Module):
    def __init__(self,loss_weight=0.05,reduction='mean'):
        super(FFTLoss, self).__init__()
        self.cri_l1 = nn.L1Loss(reduction=reduction)
        self.loss_weight = loss_weight
    def forward(self,pred,target,y1_phase,y1):
        gt_fft = torch.fft.rfft2(target, dim=(-2, -1))
        gt_amp = torch.abs(gt_fft)
        gt_phase = torch.angle(gt_fft)
        label_fft = torch.stack((gt_fft.real, gt_fft.imag), -1)
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        l_fft = self.cri_l1(pred_fft, label_fft)
        l_phase = self.cri_l1(y1_phase, gt_phase)
        # l_inv = self.cri_l1(y1,target)
        return self.loss_weight*(l_fft+l_phase)
@LOSS_REGISTRY.register()
class FFTV10Loss(nn.Module):
    def __init__(self,loss_weight=0.05,reduction='mean'):
        super(FFTV10Loss, self).__init__()
        self.cri_l1 = nn.L1Loss(reduction=reduction)
        self.loss_weight = loss_weight
    def forward(self,pred,target,y1_phase,y1_amp):
        gt_fft = torch.fft.rfft2(target, dim=(-2, -1))
        gt_amp = torch.abs(gt_fft)
        gt_phase = torch.angle(gt_fft)
        label_fft = torch.stack((gt_fft.real, gt_fft.imag), -1)
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        l_fft = self.cri_l1(pred_fft, label_fft)
        l_phase = self.cri_l1(y1_phase, gt_phase)
        l_amp = self.cri_l1(y1_amp,gt_amp)
        # l_inv = self.cri_l1(y1,target)
        return self.loss_weight*(l_fft+l_phase+l_amp)
@LOSS_REGISTRY.register()
class FFTV10Y1Loss(nn.Module):
    def __init__(self,loss_weight=0.05,reduction='mean'):
        super(FFTV10Y1Loss, self).__init__()
        self.cri_l1 = nn.L1Loss(reduction=reduction)
        self.loss_weight = loss_weight
    def forward(self,pred,target,y1_phase,y1_amp,y1):
        gt_fft = torch.fft.rfft2(target, dim=(-2, -1))
        gt_amp = torch.abs(gt_fft)
        gt_phase = torch.angle(gt_fft)
        label_fft = torch.stack((gt_fft.real, gt_fft.imag), -1)
        pred_fft = torch.fft.rfft2(pred, dim=(-2, -1))
        pred_fft = torch.stack((pred_fft.real, pred_fft.imag), -1)
        l_fft = self.cri_l1(pred_fft, label_fft)
        l_phase = self.cri_l1(y1_phase, gt_phase)
        l_amp = self.cri_l1(y1_amp,gt_amp)
        l_total = 0.5*self.cri_l1(y1,target)
        # l_inv = self.cri_l1(y1,target)
        return self.loss_weight*(l_fft+l_phase+l_amp+l_total)
