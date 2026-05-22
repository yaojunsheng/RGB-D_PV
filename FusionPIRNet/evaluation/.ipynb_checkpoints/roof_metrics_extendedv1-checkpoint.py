"""
扩展的屋顶评估指标 - 支持三个任务：seg6, seg9, height
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from evaluation.roof_metrics import ComputeRoofMetric  # 导入原始指标
from model.height_tasks import HeightMetrics


class ComputeRoofMetricExtended(ComputeRoofMetric):
    """扩展的屋顶指标计算器，支持高度回归任务"""
    
    def __init__(self, class_nb_seg6=6, class_nb_seg9=9):
        super(ComputeRoofMetricExtended, self).__init__(class_nb_seg6, class_nb_seg9)
        
        # 高度回归指标
        self.height_metrics = HeightMetrics()
        
        # 统计信息
        self.reset_stats()
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'seg6': {'total_miou': 0.0, 'total_acc': 0.0, 'count': 0},
            'seg9': {'total_miou': 0.0, 'total_acc': 0.0, 'count': 0},
            'height': {'total_rmse': 0.0, 'total_mae': 0.0, 'total_rel_error': 0.0, 
                      'total_delta_acc': 0.0, 'count': 0}
        }
    
    def compute_all_metrics(self, pred_dict, target_dict):
        """
        计算所有任务的指标
        
        Args:
            pred_dict: 预测字典 {'seg6': [B,6,H,W], 'seg9': [B,9,H,W], 'height': [B,1,H,W]}
            target_dict: 目标字典 {'seg6': [B,H,W], 'seg9': [B,H,W], 'height': [B,1,H,W]}
        
        Returns:
            metrics_dict: 指标字典
        """
        metrics = {}
        
        # seg6指标
        if 'seg6' in pred_dict and 'seg6' in target_dict:
            seg6_miou = self.compute_miou_seg6(pred_dict['seg6'], target_dict['seg6'])
            seg6_acc = self.compute_accuracy_seg6(pred_dict['seg6'], target_dict['seg6'])
            metrics['seg6_miou'] = seg6_miou.item() if isinstance(seg6_miou, torch.Tensor) else seg6_miou
            metrics['seg6_acc'] = seg6_acc.item() if isinstance(seg6_acc, torch.Tensor) else seg6_acc
            
            # 更新统计信息
            self.stats['seg6']['total_miou'] += metrics['seg6_miou']
            self.stats['seg6']['total_acc'] += metrics['seg6_acc']
            self.stats['seg6']['count'] += 1
        
        # seg9指标
        if 'seg9' in pred_dict and 'seg9' in target_dict:
            seg9_miou = self.compute_miou_seg9(pred_dict['seg9'], target_dict['seg9'])
            seg9_acc = self.compute_accuracy_seg9(pred_dict['seg9'], target_dict['seg9'])
            metrics['seg9_miou'] = seg9_miou.item() if isinstance(seg9_miou, torch.Tensor) else seg9_miou
            metrics['seg9_acc'] = seg9_acc.item() if isinstance(seg9_acc, torch.Tensor) else seg9_acc
            
            # 更新统计信息
            self.stats['seg9']['total_miou'] += metrics['seg9_miou']
            self.stats['seg9']['total_acc'] += metrics['seg9_acc']
            self.stats['seg9']['count'] += 1
        
        # 高度指标
        if 'height' in pred_dict and 'height' in target_dict:
            height_rmse = self.height_metrics.compute_rmse(pred_dict['height'], target_dict['height'])
            height_mae = self.height_metrics.compute_mae(pred_dict['height'], target_dict['height'])
            height_rel_error = self.height_metrics.compute_relative_error(pred_dict['height'], target_dict['height'])
            height_delta_acc = self.height_metrics.compute_delta_accuracy(pred_dict['height'], target_dict['height'])
            
            metrics['height_rmse'] = height_rmse.item() if isinstance(height_rmse, torch.Tensor) else height_rmse
            metrics['height_mae'] = height_mae.item() if isinstance(height_mae, torch.Tensor) else height_mae
            metrics['height_rel_error'] = height_rel_error.item() if isinstance(height_rel_error, torch.Tensor) else height_rel_error
            metrics['height_delta_acc'] = height_delta_acc.item() if isinstance(height_delta_acc, torch.Tensor) else height_delta_acc
            
            # 更新统计信息
            self.stats['height']['total_rmse'] += metrics['height_rmse']
            self.stats['height']['total_mae'] += metrics['height_mae']
            self.stats['height']['total_rel_error'] += metrics['height_rel_error']
            self.stats['height']['total_delta_acc'] += metrics['height_delta_acc']
            self.stats['height']['count'] += 1
        
        return metrics
    
    def get_average_metrics(self):
        """获取平均指标"""
        avg_metrics = {}
        
        # seg6平均指标
        if self.stats['seg6']['count'] > 0:
            avg_metrics['avg_seg6_miou'] = self.stats['seg6']['total_miou'] / self.stats['seg6']['count']
            avg_metrics['avg_seg6_acc'] = self.stats['seg6']['total_acc'] / self.stats['seg6']['count']
        
        # seg9平均指标
        if self.stats['seg9']['count'] > 0:
            avg_metrics['avg_seg9_miou'] = self.stats['seg9']['total_miou'] / self.stats['seg9']['count']
            avg_metrics['avg_seg9_acc'] = self.stats['seg9']['total_acc'] / self.stats['seg9']['count']
        
        # 高度平均指标
        if self.stats['height']['count'] > 0:
            avg_metrics['avg_height_rmse'] = self.stats['height']['total_rmse'] / self.stats['height']['count']
            avg_metrics['avg_height_mae'] = self.stats['height']['total_mae'] / self.stats['height']['count']
            avg_metrics['avg_height_rel_error'] = self.stats['height']['total_rel_error'] / self.stats['height']['count']
            avg_metrics['avg_height_delta_acc'] = self.stats['height']['total_delta_acc'] / self.stats['height']['count']
        
        return avg_metrics
    
    def compute_height_rmse(self, x_pred, x_output):
        """计算高度RMSE（兼容接口）"""
        return self.height_metrics.compute_rmse(x_pred, x_output)
    
    def compute_height_mae(self, x_pred, x_output):
        """计算高度MAE（兼容接口）"""
        return self.height_metrics.compute_mae(x_pred, x_output)
    
    def compute_height_relative_error(self, x_pred, x_output):
        """计算高度相对误差（兼容接口）"""
        return self.height_metrics.compute_relative_error(x_pred, x_output)
    
    def compute_height_delta_accuracy(self, x_pred, x_output, delta=1.25):
        """计算高度delta准确率（兼容接口）"""
        return self.height_metrics.compute_delta_accuracy(x_pred, x_output, delta)
    
    def compute_cross_task_consistency(self, seg_pred, height_pred):
        """
        计算跨任务一致性指标
        
        Args:
            seg_pred: 分割预测 [B, C, H, W]
            height_pred: 高度预测 [B, 1, H, W]
        """
        # 计算分割边界
        seg_prob = F.softmax(seg_pred, dim=1)
        seg_entropy = -torch.sum(seg_prob * torch.log(seg_prob + 1e-8), dim=1, keepdim=True)
        
        # 计算高度梯度
        height_grad_x = torch.abs(height_pred[:, :, :, 1:] - height_pred[:, :, :, :-1])
        height_grad_y = torch.abs(height_pred[:, :, 1:, :] - height_pred[:, :, :-1, :])
        
        # 填充以匹配尺寸
        height_grad_x = F.pad(height_grad_x, (0, 1, 0, 0))
        height_grad_y = F.pad(height_grad_y, (0, 0, 0, 1))
        height_grad = height_grad_x + height_grad_y
        
        # 计算相关性
        seg_entropy_flat = seg_entropy.view(-1)
        height_grad_flat = height_grad.view(-1)
        
        # 归一化
        seg_entropy_norm = (seg_entropy_flat - seg_entropy_flat.mean()) / (seg_entropy_flat.std() + 1e-8)
        height_grad_norm = (height_grad_flat - height_grad_flat.mean()) / (height_grad_flat.std() + 1e-8)
        
        # 计算皮尔逊相关系数
        correlation = torch.mean(seg_entropy_norm * height_grad_norm)
        
        return correlation
    
    def print_metrics_summary(self, metrics_dict, epoch=None, prefix=""):
        """打印指标摘要"""
        print(f"\n{prefix}=== Metrics Summary ===")
        if epoch is not None:
            print(f"Epoch: {epoch}")
        
        # 分割任务指标
        if 'seg6_miou' in metrics_dict:
            print(f"Seg6 - mIoU: {metrics_dict['seg6_miou']:.4f}, Acc: {metrics_dict.get('seg6_acc', 0.0):.4f}")
        
        if 'seg9_miou' in metrics_dict:
            print(f"Seg9 - mIoU: {metrics_dict['seg9_miou']:.4f}, Acc: {metrics_dict.get('seg9_acc', 0.0):.4f}")
        
        # 高度任务指标
        if 'height_rmse' in metrics_dict:
            print(f"Height - RMSE: {metrics_dict['height_rmse']:.4f}, MAE: {metrics_dict.get('height_mae', 0.0):.4f}")
            if 'height_rel_error' in metrics_dict:
                print(f"         Rel Error: {metrics_dict['height_rel_error']:.4f}, Delta Acc: {metrics_dict.get('height_delta_acc', 0.0):.4f}")
        
        # 平均指标
        avg_metrics = self.get_average_metrics()
        if avg_metrics:
            print(f"\nRunning Averages:")
            for key, value in avg_metrics.items():
                print(f"  {key}: {value:.4f}")
        
        print("=" * 25)


class TaskPerformanceTracker:
    """任务性能追踪器"""
    
    def __init__(self, tasks=['seg6', 'seg9', 'height']):
        self.tasks = tasks
        self.history = {task: [] for task in tasks}
        self.best_metrics = {}
        self.improvement_count = {task: 0 for task in tasks}
    
    def update(self, metrics_dict, epoch):
        """更新性能历史"""
        current_metrics = {}
        
        # 记录主要指标
        if 'seg6_miou' in metrics_dict:
            current_metrics['seg6'] = metrics_dict['seg6_miou']
        if 'seg9_miou' in metrics_dict:
            current_metrics['seg9'] = metrics_dict['seg9_miou']
        if 'height_rmse' in metrics_dict:
            # 对于RMSE，越小越好，所以取负值
            current_metrics['height'] = -metrics_dict['height_rmse']
        
        # 更新历史和最佳指标
        for task, metric in current_metrics.items():
            self.history[task].append((epoch, metric))
            
            if task not in self.best_metrics or metric > self.best_metrics[task][1]:
                self.best_metrics[task] = (epoch, metric)
                self.improvement_count[task] += 1
    
    def get_best_performance(self):
        """获取最佳性能"""
        return self.best_metrics.copy()
    
    def get_recent_trend(self, task, window=5):
        """获取最近的性能趋势"""
        if task not in self.history or len(self.history[task]) < window:
            return 0.0
        
        recent_metrics = [metric for _, metric in self.history[task][-window:]]
        if len(recent_metrics) < 2:
            return 0.0
        
        # 计算趋势（简单的线性趋势）
        trend = (recent_metrics[-1] - recent_metrics[0]) / (len(recent_metrics) - 1)
        return trend
    
    def is_improving(self, task, window=5):
        """判断任务是否在改善"""
        trend = self.get_recent_trend(task, window)
        return trend > 0.0
    
    def print_performance_summary(self):
        """打印性能摘要"""
        print("\n=== Performance Summary ===")
        for task in self.tasks:
            if task in self.best_metrics:
                epoch, metric = self.best_metrics[task]
                trend = self.get_recent_trend(task)
                improving = "↑" if self.is_improving(task) else "↓"
                
                if task == 'height':
                    # 对于高度任务，显示实际的RMSE值
                    metric = -metric
                    print(f"{task}: Best RMSE {metric:.4f} (Epoch {epoch}) {improving} Trend: {-trend:.4f}")
                else:
                    print(f"{task}: Best mIoU {metric:.4f} (Epoch {epoch}) {improving} Trend: {trend:.4f}")
        print("=" * 27)