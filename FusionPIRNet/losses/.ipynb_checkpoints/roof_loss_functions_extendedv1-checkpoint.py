"""
扩展的屋顶损失函数 - 支持三个任务：seg6, seg9, height
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from losses.roof_loss_functions import ComputeRoofLoss  # 导入原始损失函数
from model.height_tasks import HeightLoss


class ComputeRoofLossExtended(ComputeRoofLoss):
    """扩展的屋顶损失计算器，支持高度回归任务"""
    
    def __init__(self, height_loss_type='smooth_l1', height_loss_weight=1.0):
        super(ComputeRoofLossExtended, self).__init__()
        
        # 高度回归损失
        self.height_loss = HeightLoss(loss_type=height_loss_type)
        self.height_loss_weight = height_loss_weight
        
        # 任务权重（用于多任务平衡）
        self.task_weights = {
            'seg6': 1.0,
            'seg9': 1.0, 
            'height': height_loss_weight
        }
    
    def compute_supervision_extended(self, pred_dict, target_dict, task_weights=None):
        """
        计算三任务的监督损失
        
        Args:
            pred_dict: 预测字典 {'seg6': [B,6,H,W], 'seg9': [B,9,H,W], 'height': [B,1,H,W]}
            target_dict: 目标字典 {'seg6': [B,H,W], 'seg9': [B,H,W], 'height': [B,1,H,W]}
            task_weights: 可选的任务权重字典
        
        Returns:
            task_losses: 各任务损失列表 [seg6_loss, seg9_loss, height_loss]
        """
        task_losses = []
        
        # 使用传入的权重或默认权重
        weights = task_weights if task_weights is not None else self.task_weights
        
        # seg6分割损失
        if 'seg6' in pred_dict and 'seg6' in target_dict:
            pred_seg6 = F.log_softmax(pred_dict['seg6'], dim=1)
            loss_seg6 = F.nll_loss(pred_seg6, target_dict['seg6'].long(), ignore_index=-1)
            task_losses.append(loss_seg6 * weights.get('seg6', 1.0))
        else:
            task_losses.append(torch.tensor(0.0, device=next(iter(pred_dict.values())).device))
        
        # seg9分割损失
        if 'seg9' in pred_dict and 'seg9' in target_dict:
            pred_seg9 = F.log_softmax(pred_dict['seg9'], dim=1)
            loss_seg9 = F.nll_loss(pred_seg9, target_dict['seg9'].long(), ignore_index=-1)
            task_losses.append(loss_seg9 * weights.get('seg9', 1.0))
        else:
            task_losses.append(torch.tensor(0.0, device=next(iter(pred_dict.values())).device))
        
        # 高度回归损失
        if 'height' in pred_dict and 'height' in target_dict:
            loss_height = self.height_loss(pred_dict['height'], target_dict['height'])
            task_losses.append(loss_height * weights.get('height', 1.0))
        else:
            task_losses.append(torch.tensor(0.0, device=next(iter(pred_dict.values())).device))
        
        return task_losses
    
    def compute_supervision(self, x_pred_seg6, x_output_seg6, x_pred_seg9, x_output_seg9, 
                          x_pred_height=None, x_output_height=None):
        """
        兼容原始接口的监督损失计算
        """
        # 构建预测和目标字典
        pred_dict = {}
        target_dict = {}
        
        if x_pred_seg6 is not None:
            pred_dict['seg6'] = x_pred_seg6
            target_dict['seg6'] = x_output_seg6
        
        if x_pred_seg9 is not None:
            pred_dict['seg9'] = x_pred_seg9
            target_dict['seg9'] = x_output_seg9
        
        if x_pred_height is not None:
            pred_dict['height'] = x_pred_height
            target_dict['height'] = x_output_height
        
        return self.compute_supervision_extended(pred_dict, target_dict)
    
    def compute_height_consistency_loss(self, height_pred1, height_pred2, consistency_type='mse'):
        """
        计算高度预测的一致性损失（用于数据增强后的一致性）
        
        Args:
            height_pred1: 第一个高度预测 [B, 1, H, W]
            height_pred2: 第二个高度预测 [B, 1, H, W] 
            consistency_type: 一致性损失类型
        """
        if consistency_type == 'mse':
            consistency_loss = F.mse_loss(height_pred1, height_pred2)
        elif consistency_type == 'mae':
            consistency_loss = F.l1_loss(height_pred1, height_pred2)
        elif consistency_type == 'smooth_l1':
            consistency_loss = F.smooth_l1_loss(height_pred1, height_pred2)
        else:
            raise ValueError(f"Unsupported consistency loss type: {consistency_type}")
        
        return consistency_loss
    
    def compute_cross_task_loss(self, seg_pred, height_pred, loss_weight=0.1):
        """
        计算跨任务损失 - 鼓励分割边界与高度变化一致
        
        Args:
            seg_pred: 分割预测 [B, C, H, W]
            height_pred: 高度预测 [B, 1, H, W]
            loss_weight: 损失权重
        """
        # 计算分割预测的边界
        seg_prob = F.softmax(seg_pred, dim=1)
        seg_entropy = -torch.sum(seg_prob * torch.log(seg_prob + 1e-8), dim=1, keepdim=True)
        
        # 计算高度的梯度（边界）
        height_grad_x = torch.abs(height_pred[:, :, :, 1:] - height_pred[:, :, :, :-1])
        height_grad_y = torch.abs(height_pred[:, :, 1:, :] - height_pred[:, :, :-1, :])
        
        # 填充以匹配尺寸
        height_grad_x = F.pad(height_grad_x, (0, 1, 0, 0))
        height_grad_y = F.pad(height_grad_y, (0, 0, 0, 1))
        height_grad = height_grad_x + height_grad_y
        
        # 鼓励高熵（边界）区域与高度梯度一致
        cross_task_loss = F.mse_loss(seg_entropy, height_grad)
        
        return cross_task_loss * loss_weight
    
    def compute_distill_loss_extended(self, s_pred_dict, t_pred_dict):
        """
        扩展的蒸馏损失计算，支持三个任务
        
        Args:
            s_pred_dict: 学生模型预测字典
            t_pred_dict: 教师模型预测字典
        """
        distill_losses = []
        
        # seg6蒸馏损失
        if 'seg6' in s_pred_dict and 'seg6' in t_pred_dict:
            loss_seg6 = self.comp_semantic_distill_loss(s_pred_dict['seg6'], t_pred_dict['seg6'].detach())
            distill_losses.append(loss_seg6)
        else:
            distill_losses.append(torch.tensor(0.0))
        
        # seg9蒸馏损失
        if 'seg9' in s_pred_dict and 'seg9' in t_pred_dict:
            loss_seg9 = self.comp_semantic_distill_loss(s_pred_dict['seg9'], t_pred_dict['seg9'].detach())
            distill_losses.append(loss_seg9)
        else:
            distill_losses.append(torch.tensor(0.0))
        
        # 高度蒸馏损失
        if 'height' in s_pred_dict and 'height' in t_pred_dict:
            loss_height = self.comp_height_distill_loss(s_pred_dict['height'], t_pred_dict['height'].detach())
            distill_losses.append(loss_height)
        else:
            distill_losses.append(torch.tensor(0.0))
        
        return distill_losses
    
    def comp_height_distill_loss(self, student_height, teacher_height, temperature=3.0):
        """
        计算高度回归的蒸馏损失
        
        Args:
            student_height: 学生模型的高度预测 [B, 1, H, W]
            teacher_height: 教师模型的高度预测 [B, 1, H, W]
            temperature: 温度参数
        """
        # 对于回归任务，使用MSE作为蒸馏损失
        distill_loss = F.mse_loss(student_height / temperature, teacher_height / temperature)
        return distill_loss * (temperature ** 2)
    
    def update_task_weights(self, new_weights):
        """更新任务权重"""
        self.task_weights.update(new_weights)
    
    def get_task_weights(self):
        """获取当前任务权重"""
        return self.task_weights.copy()


class DynamicTaskWeighting(nn.Module):
    """动态任务权重调整"""
    
    def __init__(self, num_tasks=3, temperature=2.0):
        super(DynamicTaskWeighting, self).__init__()
        self.num_tasks = num_tasks
        self.temperature = temperature
        
        # 可学习的任务权重参数
        self.task_weights = nn.Parameter(torch.ones(num_tasks))
        
    def forward(self, task_losses):
        """
        根据任务损失动态调整权重
        
        Args:
            task_losses: 任务损失列表
        """
        # 将损失转换为权重
        loss_ratios = torch.stack([loss.detach() for loss in task_losses])
        
        # 计算相对损失比例
        loss_ratios = loss_ratios / (loss_ratios.sum() + 1e-8)
        
        # 使用softmax计算权重
        weights = F.softmax(self.task_weights / self.temperature, dim=0)
        
        # 应用权重
        weighted_losses = [w * loss for w, loss in zip(weights, task_losses)]
        
        return weighted_losses, weights


class GradientBasedTaskWeighting(nn.Module):
    """基于梯度的任务权重调整（GradNorm风格）"""
    
    def __init__(self, num_tasks=3, alpha=1.5):
        super(GradientBasedTaskWeighting, self).__init__()
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.task_weights = nn.Parameter(torch.ones(num_tasks))
        
    def forward(self, shared_representation, task_losses, initial_losses=None):
        """
        基于梯度平衡调整任务权重
        
        Args:
            shared_representation: 共享表示（用于计算梯度）
            task_losses: 当前任务损失
            initial_losses: 初始任务损失（用于归一化）
        """
        if initial_losses is None:
            initial_losses = [loss.detach() for loss in task_losses]
        
        # 计算梯度
        grads = []
        for i, loss in enumerate(task_losses):
            grad = torch.autograd.grad(
                self.task_weights[i] * loss, 
                shared_representation, 
                retain_graph=True, 
                create_graph=True
            )[0]
            grads.append(torch.norm(grad))
        
        # 计算目标梯度幅度
        avg_grad = sum(grads) / len(grads)
        relative_losses = [loss / initial_loss for loss, initial_loss in zip(task_losses, initial_losses)]
        target_grads = [avg_grad * (rel_loss ** self.alpha) for rel_loss in relative_losses]
        
        # 计算梯度平衡损失
        grad_balance_loss = sum([
            torch.abs(grad - target_grad) 
            for grad, target_grad in zip(grads, target_grads)
        ])
        
        return grad_balance_loss