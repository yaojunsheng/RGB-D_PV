"""
扩展的屋顶评估指标 - 支持三个任务：seg6, seg9, height
优化版本，确保与原有代码完全兼容并添加了错误处理
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from evaluation.roof_metrics import ComputeRoofMetric  # 导入原始指标
from model.height_tasks import HeightMetrics


class ComputeRoofMetricExtended(ComputeRoofMetric):
    """扩展的屋顶指标计算器，支持高度回归任务"""
    
    def __init__(self, class_nb_seg6=6, class_nb_seg9=6):
        super(ComputeRoofMetricExtended, self).__init__(class_nb_seg6, class_nb_seg9)
        
        # 高度回归指标
        self.height_metrics = HeightMetrics()
        
        # 统计信息
        self.reset_stats()
        
        # 任务配置
        self.supported_tasks = ['seg6', 'seg9', 'height']
    
    def reset_stats(self):
        """重置统计信息"""
        self.stats = {
            'seg6': {'total_miou': 0.0, 'total_acc': 0.0, 'count': 0},
            'seg9': {'total_miou': 0.0, 'total_acc': 0.0, 'count': 0},
            'height': {'total_rmse': 0.0, 'total_mae': 0.0, 'total_rel_error': 0.0, 
                      'total_delta_acc': 0.0, 'count': 0}
        }
    
    def compute_all_metrics(self, pred_dict, target_dict, tasks=None):
        """
        计算所有任务的指标
        
        Args:
            pred_dict: 预测字典 {'seg6': [B,6,H,W], 'seg9': [B,9,H,W], 'height': [B,1,H,W]}
            target_dict: 目标字典 {'seg6': [B,H,W], 'seg9': [B,H,W], 'height': [B,1,H,W]}
            tasks: 要计算的任务列表，如果为None则计算所有可用任务
        
        Returns:
            metrics_dict: 指标字典
        """
        if tasks is None:
            tasks = self.supported_tasks
        
        metrics = {}
        
        # seg6指标
        if 'seg6' in tasks and 'seg6' in pred_dict and 'seg6' in target_dict:
            try:
                seg6_miou = self.compute_miou_seg6(pred_dict['seg6'], target_dict['seg6'])
                seg6_acc = self.compute_accuracy_seg6(pred_dict['seg6'], target_dict['seg6'])
                
                metrics['seg6_miou'] = self._tensor_to_float(seg6_miou)
                metrics['seg6_acc'] = self._tensor_to_float(seg6_acc)
                
                # 更新统计信息
                self.stats['seg6']['total_miou'] += metrics['seg6_miou']
                self.stats['seg6']['total_acc'] += metrics['seg6_acc']
                self.stats['seg6']['count'] += 1
                
            except Exception as e:
                print(f"Warning: Failed to compute seg6 metrics: {e}")
                metrics['seg6_miou'] = 0.0
                metrics['seg6_acc'] = 0.0
        
        # seg9指标
        if 'seg9' in tasks and 'seg9' in pred_dict and 'seg9' in target_dict:
            try:
                seg9_miou = self.compute_miou_seg9(pred_dict['seg9'], target_dict['seg9'])
                seg9_acc = self.compute_accuracy_seg9(pred_dict['seg9'], target_dict['seg9'])
                
                metrics['seg9_miou'] = self._tensor_to_float(seg9_miou)
                metrics['seg9_acc'] = self._tensor_to_float(seg9_acc)
                
                # 更新统计信息
                self.stats['seg9']['total_miou'] += metrics['seg9_miou']
                self.stats['seg9']['total_acc'] += metrics['seg9_acc']
                self.stats['seg9']['count'] += 1
                
            except Exception as e:
                print(f"Warning: Failed to compute seg9 metrics: {e}")
                metrics['seg9_miou'] = 0.0
                metrics['seg9_acc'] = 0.0
        
        # 高度指标
        if 'height' in tasks and 'height' in pred_dict and 'height' in target_dict:
            try:
                height_rmse = self.height_metrics.compute_rmse(pred_dict['height'], target_dict['height'])
                height_mae = self.height_metrics.compute_mae(pred_dict['height'], target_dict['height'])
                height_rel_error = self.height_metrics.compute_relative_error(pred_dict['height'], target_dict['height'])
                height_delta_acc = self.height_metrics.compute_delta_accuracy(pred_dict['height'], target_dict['height'])
                
                metrics['height_rmse'] = self._tensor_to_float(height_rmse)
                metrics['height_mae'] = self._tensor_to_float(height_mae)
                metrics['height_rel_error'] = self._tensor_to_float(height_rel_error)
                metrics['height_delta_acc'] = self._tensor_to_float(height_delta_acc)
                
                # 更新统计信息
                self.stats['height']['total_rmse'] += metrics['height_rmse']
                self.stats['height']['total_mae'] += metrics['height_mae']
                self.stats['height']['total_rel_error'] += metrics['height_rel_error']
                self.stats['height']['total_delta_acc'] += metrics['height_delta_acc']
                self.stats['height']['count'] += 1
                
            except Exception as e:
                print(f"Warning: Failed to compute height metrics: {e}")
                metrics['height_rmse'] = float('inf')
                metrics['height_mae'] = float('inf')
                metrics['height_rel_error'] = float('inf')
                metrics['height_delta_acc'] = 0.0
        
        return metrics
    
    def _tensor_to_float(self, tensor_value):
        """安全地将tensor转换为float"""
        if isinstance(tensor_value, torch.Tensor):
            return tensor_value.item()
        else:
            return float(tensor_value)
    
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
    
    # 兼容接口 - 保持与原有代码的兼容性
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
    
    # 参考开源版本的深度误差计算方法
    def depth_error(self, x_pred, x_output):
        """
        参考开源版本的深度误差计算方法
        返回绝对误差和相对误差
        """
        try:
            # 创建有效像素掩码
            binary_mask = (torch.sum(x_output, dim=1) != 0).unsqueeze(1)
            if x_pred.is_cuda:
                binary_mask = binary_mask.cuda()
            
            # 选择有效像素
            x_pred_true = x_pred.masked_select(binary_mask)
            x_output_true = x_output.masked_select(binary_mask)
            
            if x_pred_true.numel() == 0:
                return torch.tensor(0.0), torch.tensor(0.0)
            
            # 计算绝对误差和相对误差
            abs_err = torch.abs(x_pred_true - x_output_true)
            rel_err = torch.abs(x_pred_true - x_output_true) / (x_output_true + 1e-8)  # 避免除零
            
            return (torch.sum(abs_err) / torch.nonzero(binary_mask).size(0), 
                    torch.sum(rel_err) / torch.nonzero(binary_mask).size(0))
        except Exception as e:
            print(f"Warning: Failed to compute depth error: {e}")
            return torch.tensor(0.0), torch.tensor(0.0)
    
    def compute_cross_task_consistency(self, seg_pred, height_pred):
        """
        计算跨任务一致性指标
        分割边界应该与高度变化边界相对应
        
        Args:
            seg_pred: 分割预测 [B, C, H, W]
            height_pred: 高度预测 [B, 1, H, W]
        """
        try:
            # 计算分割不确定性（熵）
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
        except Exception as e:
            print(f"Warning: Failed to compute cross-task consistency: {e}")
            return torch.tensor(0.0)
    
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
    """任务性能追踪器 - 增强版本"""
    
    def __init__(self, tasks=['seg6', 'seg9', 'height']):
        self.tasks = tasks
        self.history = {task: [] for task in tasks}
        self.best_metrics = {}
        self.improvement_count = {task: 0 for task in tasks}
        self.plateau_count = {task: 0 for task in tasks}  # 平台期计数
        self.patience = 10  # 平台期容忍度
    
    def update(self, metrics_dict, epoch):
        """更新性能历史"""
        current_metrics = {}
        
        # 记录主要指标
        if 'seg6_miou' in metrics_dict:
            current_metrics['seg6'] = metrics_dict['seg6_miou']
        if 'seg9_miou' in metrics_dict:
            current_metrics['seg9'] = metrics_dict['seg9_miou']
        if 'height_rmse' in metrics_dict:
            # 对于RMSE，越小越好，所以取负值用于比较
            current_metrics['height'] = -metrics_dict['height_rmse']
        
        # 更新历史和最佳指标
        for task, metric in current_metrics.items():
            self.history[task].append((epoch, metric))
            
            # 更新最佳指标
            if task not in self.best_metrics or metric > self.best_metrics[task][1]:
                self.best_metrics[task] = (epoch, metric)
                self.improvement_count[task] += 1
                self.plateau_count[task] = 0  # 重置平台期计数
            else:
                self.plateau_count[task] += 1
    
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
    
    def is_in_plateau(self, task):
        """判断任务是否进入平台期"""
        return self.plateau_count.get(task, 0) > self.patience
    
    def get_task_status(self, task):
        """获取任务状态"""
        if task not in self.history:
            return "No data"
        
        if self.is_in_plateau(task):
            return "Plateau"
        elif self.is_improving(task):
            return "Improving"
        else:
            return "Declining"
    
    def print_performance_summary(self):
        """打印性能摘要"""
        print("\n=== Performance Summary ===")
        for task in self.tasks:
            if task in self.best_metrics:
                epoch, metric = self.best_metrics[task]
                trend = self.get_recent_trend(task)
                status = self.get_task_status(task)
                
                if task == 'height':
                    # 对于高度任务，显示实际的RMSE值
                    metric = -metric
                    print(f"{task}: Best RMSE {metric:.4f} (Epoch {epoch}) | Status: {status} | Trend: {-trend:.4f}")
                else:
                    print(f"{task}: Best mIoU {metric:.4f} (Epoch {epoch}) | Status: {status} | Trend: {trend:.4f}")
        print("=" * 27)
    
    def save_history(self, filepath):
        """保存历史记录"""
        import json
        
        history_data = {
            'tasks': self.tasks,
            'history': {task: [(int(epoch), float(metric)) for epoch, metric in hist] 
                       for task, hist in self.history.items()},
            'best_metrics': {task: (int(epoch), float(metric)) 
                           for task, (epoch, metric) in self.best_metrics.items()},
            'improvement_count': self.improvement_count,
            'plateau_count': self.plateau_count
        }
        
        with open(filepath, 'w') as f:
            json.dump(history_data, f, indent=2)
    
    def load_history(self, filepath):
        """加载历史记录"""
        import json
        
        try:
            with open(filepath, 'r') as f:
                history_data = json.load(f)
            
            self.tasks = history_data['tasks']
            self.history = {task: [(epoch, metric) for epoch, metric in hist] 
                           for task, hist in history_data['history'].items()}
            self.best_metrics = {task: (epoch, metric) 
                               for task, (epoch, metric) in history_data['best_metrics'].items()}
            self.improvement_count = history_data.get('improvement_count', {task: 0 for task in self.tasks})
            self.plateau_count = history_data.get('plateau_count', {task: 0 for task in self.tasks})
            
            print(f"Successfully loaded performance history from {filepath}")
        except Exception as e:
            print(f"Warning: Failed to load history from {filepath}: {e}")


# 使用示例
def example_usage():
    """使用示例"""
    
    # 初始化评估器
    evaluator = ComputeRoofMetricExtended(class_nb_seg6=6, class_nb_seg9=6)
    tracker = TaskPerformanceTracker()
    
    # 模拟一些预测和目标数据
    batch_size, height, width = 4, 256, 256
    
    pred_dict = {
        'seg6': torch.randn(batch_size, 6, height, width),
        'seg9': torch.randn(batch_size, 9, height, width),
        'height': torch.randn(batch_size, 1, height, width)
    }
    
    target_dict = {
        'seg6': torch.randint(0, 6, (batch_size, height, width)),
        'seg9': torch.randint(0, 9, (batch_size, height, width)),
        'height': torch.randn(batch_size, 1, height, width)
    }
    
    # 计算指标
    metrics = evaluator.compute_all_metrics(pred_dict, target_dict)
    
    # 打印结果
    evaluator.print_metrics_summary(metrics, epoch=1)
    
    # 更新追踪器
    tracker.update(metrics, epoch=1)
    tracker.print_performance_summary()
    
    return evaluator, tracker

if __name__ == "__main__":
    example_usage()