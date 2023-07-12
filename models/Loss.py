# _*_ coding : utf-8 _*_
# @Time : 2023/7/7 11:29
# @Author : Black
# @File : Loss
# @Project : BabyBeatAnalyzer

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 计算交叉熵损失
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # 预测概率

        # 计算Focal Loss
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        # 根据reduction参数进行损失的降维操作
        if self.reduction == 'mean':
            focal_loss = torch.mean(focal_loss)
        elif self.reduction == 'sum':
            focal_loss = torch.sum(focal_loss)

        return focal_loss
