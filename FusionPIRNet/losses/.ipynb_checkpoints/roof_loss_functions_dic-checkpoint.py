import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BalancedCrossEntropyLoss(nn.Module):
    """
    保持原始的BalancedCrossEntropyLoss不变
    """
    def __init__(self, size_average=True, batch_average=True, pos_weight=0.95):
        super(BalancedCrossEntropyLoss, self).__init__()
        self.size_average = size_average
        self.batch_average = batch_average
        self.pos_weight = pos_weight
    
    def forward(self, output, label, void_pixels=None):
        assert (output.size() == label.size())
        labels = torch.ge(label, 0.5).float()
        
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
    """
    关键修复：完全保持原始的ComputeRoofLoss类
    确保损失计算与原始训练脚本完全一致
    """
    def __init__(self):
        super(ComputeRoofLoss, self).__init__()
        self.comp_edge_loss = BalancedCrossEntropyLoss()
    
    def forward(self):
        pass
    
    def compute_supervision(self, x_pred_seg6, x_output_seg6, x_pred_seg9, x_output_seg9):
        """
        关键修复：完全保持原始监督损失计算逻辑
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

# 关键修复：提供一个简化版的统一损失，但优先保持原始逻辑
class OptionalUnifiedRoofLoss(nn.Module):
    """
    可选的统一损失函数 - 仅在需要时使用
    优先使用原始的ComputeRoofLoss以确保性能稳定
    """
    
    def __init__(self, 
                 tasks=['seg6', 'seg9'],
                 class_nb_seg6=6, 
                 class_nb_seg9=9,
                 use_unified=False):  # 默认关闭统一损失
        super(OptionalUnifiedRoofLoss, self).__init__()
        
        self.tasks = tasks
        self.class_nb_seg6 = class_nb_seg6
        self.class_nb_seg9 = class_nb_seg9
        self.use_unified = use_unified
        
        # 关键修复：优先使用原始损失函数
        self.original_loss = ComputeRoofLoss()
        
    def forward(self, 
                pred_seg6, label_seg6,
                pred_seg9, label_seg9,
                region_consistency_loss=None,
                consistency_weight=2.0):
        """
        关键修复：默认使用原始损失计算
        """
        if not self.use_unified:
            # 使用原始损失计算逻辑
            supervision_losses = self.original_loss.compute_supervision(
                pred_seg6, label_seg6, pred_seg9, label_seg9
            )
            
            total_supervision = supervision_losses[0] + supervision_losses[1]
            
            if region_consistency_loss is not None:
                if not isinstance(region_consistency_loss, torch.Tensor):
                    region_consistency_loss = torch.tensor(float(region_consistency_loss), 
                                                         device=pred_seg6.device, requires_grad=True)
                total_loss = total_supervision + consistency_weight * region_consistency_loss
            else:
                total_loss = total_supervision
                region_consistency_loss = torch.tensor(0.0, device=pred_seg6.device)
            
            loss_dict = {
                'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss,
                'supervision_loss': total_supervision.item() if isinstance(total_supervision, torch.Tensor) else total_supervision,
                'seg6_loss': supervision_losses[0].item() if isinstance(supervision_losses[0], torch.Tensor) else supervision_losses[0],
                'seg9_loss': supervision_losses[1].item() if isinstance(supervision_losses[1], torch.Tensor) else supervision_losses[1],
                'consistency_loss': region_consistency_loss.item() if isinstance(region_consistency_loss, torch.Tensor) else region_consistency_loss,
            }
            
            return total_loss, loss_dict
        else:
            # 统一损失逻辑（仅在实验时使用）
            return self._compute_unified_loss(pred_seg6, label_seg6, pred_seg9, label_seg9, 
                                            region_consistency_loss, consistency_weight)
    
    def _compute_unified_loss(self, pred_seg6, label_seg6, pred_seg9, label_seg9, 
                            region_consistency_loss, consistency_weight):
        """
        统一损失计算（实验性功能）
        """
        # 保持与原始相同的监督损失计算
        supervision_losses = self.original_loss.compute_supervision(
            pred_seg6, label_seg6, pred_seg9, label_seg9
        )
        
        total_supervision = supervision_losses[0] + supervision_losses[1]
        
        if region_consistency_loss is not None:
            if not isinstance(region_consistency_loss, torch.Tensor):
                region_consistency_loss = torch.tensor(float(region_consistency_loss), 
                                                     device=pred_seg6.device, requires_grad=True)
            total_loss = total_supervision + consistency_weight * region_consistency_loss
        else:
            total_loss = total_supervision
            region_consistency_loss = torch.tensor(0.0, device=pred_seg6.device)
        
        loss_dict = {
            'total_loss': total_loss.item(),
            'supervision_loss': total_supervision.item(),
            'seg6_loss': supervision_losses[0].item(),
            'seg9_loss': supervision_losses[1].item(),
            'consistency_loss': region_consistency_loss.item(),
        }
        
        return total_loss, loss_dict
    
    def enable_unified_loss(self, enable=True):
        """
        启用/禁用统一损失模式
        """
        self.use_unified = enable
        return self
    
    def get_loss_info(self):
        """
        获取损失函数信息
        """
        info = {
            'mode': 'unified' if self.use_unified else 'original',
            'primary_components': [
                'Original ComputeRoofLoss (supervision)',
                'Region Consistency Loss (from RoofMapCons)',
            ],
            'optional_components': [
                'Unified Loss Framework (experimental)',
                'Dynamic Dictionary Learning (minimal impact)',
            ],
            'recommendation': 'Use original mode for stable performance, unified for experiments',
            'performance_priority': 'Original mode prioritized for performance stability'
        }
        return info

# 性能诊断工具
class PerformanceDiagnostics:
    """
    性能诊断和修复建议
    """
    
    @staticmethod
    def diagnose_performance_drop():
        """
        诊断性能下降的可能原因
        """
        diagnosis = {
            'likely_causes': [
                '1. Dynamic Dictionary Learning 增加了训练复杂度',
                '2. 统一损失框架改变了损失尺度和权重',
                '3. 新增的组件干扰了原有的训练动态',
                '4. 损失函数的梯度流发生变化'
            ],
            'immediate_fixes': [
                '1. 禁用动态词典学习 (enable_dynamic_dict=False)',
                '2. 使用原始损失计算 (use_unified=False)',
                '3. 保持原始的权重更新策略',
                '4. 确保所有损失组件的权重与原版一致'
            ],
            'verification_steps': [
                '1. 对比修复版和原版的损失曲线',
                '2. 检查各个损失组件的数值范围',
                '3. 监控梯度范数是否正常',
                '4. 验证mIoU收敛趋势'
            ]
        }
        return diagnosis
    
    @staticmethod
    def get_recommended_settings():
        """
        推荐的设置以恢复性能
        """
        settings = {
            'RoofMapCons': {
                'enable_dynamic_dict': True,
                'use_original_region_loss': True,
                'preserve_weight_update': True
            },
            'Loss_Function': {
                'use_unified': False,
                'use_original_compute_supervision': True,
                'consistency_weight': 2.0  # 原始默认值
            },
            'Training': {
                'preserve_original_logic': True,
                'no_additional_components': True,
                'minimal_modifications': True
            }
        }
        return settings