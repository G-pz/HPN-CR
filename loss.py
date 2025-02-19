import torch
import torch.nn as nn
import pytorch_msssim
import math

class sl1_ssim_loss(nn.Module):
    def __init__(self):
        super(sl1_ssim_loss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        self.ssim_loss = pytorch_msssim.MS_SSIM(data_range=1, channel=13)
    def forward(self, loc_pred, loc_target):
        l1_loss = self.smooth_l1_loss(loc_pred, loc_target).reshape([-1, 1])
        ssim_loss = 1 - self.ssim_loss(loc_pred, loc_target)
        return 0.2 * l1_loss + 0.8 * ssim_loss

class sl1_ssim_sam_loss(nn.Module):
    def __init__(self):
        super(sl1_ssim_sam_loss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='mean')
        self.ssim_loss = pytorch_msssim.MS_SSIM(data_range=1, channel=13)
        self.sam_loss = SAMLoss()
    def forward(self, loc_pred, loc_target):
        l1_loss = self.smooth_l1_loss(loc_pred, loc_target).reshape([-1, 1])
        ssim_loss = 1 - self.ssim_loss(loc_pred, loc_target)
        sam_Loss  = self.sam_loss(loc_pred, loc_target)
        return 0.2 * l1_loss + 0.8 * ssim_loss + 0.005 * sam_Loss
        # return ssim_loss + sam_Loss

def _sam(x1, x2):

    B,N,_,_ = x1.shape
    x1_ = x1.reshape(B * N, -1)
    x2_ = x2.reshape(B * N, -1)

    A = torch.sum(x1_ * x2_, axis=1)/(
        torch.sqrt(torch.sum(x1_ ** 2, axis=1)) * torch.sqrt(torch.sum(x2_ ** 2, axis=1)))
    SAM = torch.acos(A) * 180 / math.pi

    return torch.mean(SAM)

class SAMLoss(nn.Module):

    def __init__(self):
        super(SAMLoss, self).__init__()

    def forward(self, pred, target):
        return _sam(pred, target)