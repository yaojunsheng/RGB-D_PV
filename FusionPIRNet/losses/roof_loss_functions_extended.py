"""
扩展的屋顶损失函数 - 支持三个任务：seg6, seg9, height
修改版本：高度任务使用 nn.SmoothL1Loss()
类名保持一致版本，可直接替换原有的losses/roof_loss_functions_extended.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
import numpy as np


class BalancedCrossEntropyLoss(Module):
    """
    Balanced Cross Entropy Loss with optional ignore regions
    """
    def __init__(self, size_average=True, batch_average=True, pos_weight=0.95):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = pos_weight

    def forward(self, output, label, void_pixels=None):
        assert (output.size() == label.size())
        labels = torch.ge(label, 0.5).float()

        # Weighting of the loss, default is HED-style
        if self.pos_weight is None:
            num_labels_pos = torch.sum(labels)
            num_labels_neg = torch.sum(1.0 - labels)
            num_total = num_labels_pos + num_labels_neg
            w = num_labels_neg / num_total
        else:
            w = self.pos_weight

        output_gt_zero = torch.ge(output, 0).float()
        loss_val = torch.mul(output, (labels - output_gt_zero)) - torch.log(
            1 + torch.exp(output - 2 * torch.mul(output, output_gt_zero)))

        loss_pos_pix = -torch.mul(labels, loss_val)
        loss_neg_pix = -torch.mul(1.0 - labels, loss_val)

        if void_pixels is not None and not self.pos_weight:
            w_void = torch.le(void_pixels, 0.5).float()
            loss_pos_pix = torch.mul(w_void, loss_pos_pix)
            loss_neg_pix = torch.mul(w_void, loss_neg_pix)
            num_total = num_total - torch.ge(void_pixels, 0.5).float().sum()
            w = num_labels_neg / num_total

        loss_pos = torch.sum(loss_pos_pix)
        loss_neg = torch.sum(loss_neg_pix)

        final_loss = w * loss_pos + (1 - w) * loss_neg

        if self.size_average:
            final_loss /= float(np.prod(label.size()))
        elif self.batch_average:
            final_loss /= label.size()[0]

        return final_loss


class ComputeRoofLoss(nn.Module):
    """原始的屋顶损失计算器，支持seg6, seg9两个任务"""
    
    def __init__(self):
        super(ComputeRoofLoss, self).__init__()
        self.comp_edge_loss = BalancedCrossEntropyLoss()
    
    def forward(self):
        pass

    def compute_supervision(self, x_pred_seg6, x_output_seg6, x_pred_seg9, x_output_seg9):
        """
        Compute supervised task-specific loss for both segmentation tasks
        
        Args:
            x_pred_seg6: predicted logits for seg6 task [B, 6, H, W]
            x_output_seg6: ground truth labels for seg6 task [B, H, W]
            x_pred_seg9: predicted logits for seg9 task [B, 9, H, W]
            x_output_seg9: ground truth labels for seg9 task [B, H, W]
        """
        
        # seg6 loss: cross entropy for 6-class segmentation
        x_pred_seg6 = F.log_softmax(x_pred_seg6, dim=1) 
        loss_seg6 = F.nll_loss(x_pred_seg6, x_output_seg6.long(), ignore_index=-1)
        
        # seg9 loss: cross entropy for 9-class segmentation
        x_pred_seg9 = F.log_softmax(x_pred_seg9, dim=1)
        loss_seg9 = F.nll_loss(x_pred_seg9, x_output_seg9.long(), ignore_index=-1)
        
        return [loss_seg6, loss_seg9]
    
    def compute_distill_loss(self, s_pred_seg6, t_pred_seg6, s_pred_seg9, t_pred_seg9):
        """
        Compute distillation loss between student and teacher predictions
        """
        # seg6 distillation loss
        loss_seg6 = self.comp_semantic_distill_loss(s_pred_seg6, t_pred_seg6.detach())
        # seg9 distillation loss
        loss_seg9 = self.comp_semantic_distill_loss(s_pred_seg9, t_pred_seg9.detach())
        return [loss_seg6, loss_seg9]
    
    def comp_semantic_distill_loss(self, y_s, y_t, T=5):
        """
        Compute semantic distillation loss using KL divergence
        """
        p_s = F.log_softmax(y_s / T, dim=1)
        p_t = F.softmax(y_t / T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='mean') * (T**4) / y_s.shape[0]
        return loss


class ComputeRoofLossExtended(ComputeRoofLoss):
    """扩展的屋顶损失计算器，支持seg6, seg9, height三个任务"""
    
    def __init__(self):
        super(ComputeRoofLossExtended, self).__init__()
        # 添加高度任务的SmoothL1Loss
        self.height_loss_fn = nn.SmoothL1Loss()
    
    def forward(self):
        pass

    def compute_supervision(self, x_pred_seg6, x_output_seg6, x_pred_seg9, x_output_seg9, 
                          x_pred_height=None, x_output_height=None):
        """
        计算监督损失，支持两个分割任务和一个高度回归任务
        
        Args:
            x_pred_seg6: seg6预测 [B, 6, H, W]
            x_output_seg6: seg6标签 [B, H, W]
            x_pred_seg9: seg9预测 [B, 9, H, W]
            x_output_seg9: seg9标签 [B, H, W]
            x_pred_height: 高度预测 [B, 1, H, W] (可选)
            x_output_height: 高度标签 [B, 1, H, W] (可选)
        
        Returns:
            losses: 损失列表 [loss_seg6, loss_seg9] 或 [loss_seg6, loss_seg9, loss_height]
        """
        
        # 调用父类方法获取seg6和seg9的损失
        seg_losses = super().compute_supervision(x_pred_seg6, x_output_seg6, x_pred_seg9, x_output_seg9)
        
        # 如果没有height任务，返回原有的两个损失
        if x_pred_height is None:
            return seg_losses
        
        # height loss: 使用 SmoothL1Loss，高度标签中的0值也参与损失计算
        loss_height = self.height_loss_fn(x_pred_height, x_output_height)
        
        return [seg_losses[0], seg_losses[1], loss_height]

    def compute_distill_loss(self, s_pred_seg6, t_pred_seg6, s_pred_seg9, t_pred_seg9,
                           s_pred_height=None, t_pred_height=None):
        """
        计算蒸馏损失，支持两个分割任务和一个高度回归任务
        
        Args:
            s_pred_seg6: 学生模型seg6预测 [B, 6, H, W]
            t_pred_seg6: 教师模型seg6预测 [B, 6, H, W]
            s_pred_seg9: 学生模型seg9预测 [B, 9, H, W] 
            t_pred_seg9: 教师模型seg9预测 [B, 9, H, W]
            s_pred_height: 学生模型高度预测 [B, 1, H, W] (可选)
            t_pred_height: 教师模型高度预测 [B, 1, H, W] (可选)
        
        Returns:
            distill_losses: 蒸馏损失列表
        """
        
        # 调用父类方法获取seg6和seg9的蒸馏损失
        seg_distill_losses = super().compute_distill_loss(s_pred_seg6, t_pred_seg6, s_pred_seg9, t_pred_seg9)
        
        # 如果没有height任务，返回原有的两个蒸馏损失
        if s_pred_height is None:
            return seg_distill_losses
        
        # height distillation loss (使用SmoothL1Loss)
        loss_height = self.comp_height_distill_loss(s_pred_height, t_pred_height.detach())
        
        return [seg_distill_losses[0], seg_distill_losses[1], loss_height]

    def comp_height_distill_loss(self, y_s, y_t, T=2):
        """
        计算高度回归的蒸馏损失 (使用SmoothL1Loss)
        """
        # 对于回归任务，直接使用SmoothL1Loss进行蒸馏
        loss = self.height_loss_fn(y_s, y_t)
        return loss