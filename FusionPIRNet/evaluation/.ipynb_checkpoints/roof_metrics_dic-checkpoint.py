import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import Dict, List, Tuple

class ComputeRoofMetric(nn.Module):
    """
    改进版屋顶分割评估模块
    
    保持原有功能的同时，新增对动态词典学习效果的评估
    """
    
    def __init__(self, class_nb_seg6=6, class_nb_seg9=9):
        super(ComputeRoofMetric, self).__init__()
        self.class_nb_seg6 = class_nb_seg6
        self.class_nb_seg9 = class_nb_seg9
        
        # 用于动态词典质量评估的缓存
        self.dict_quality_history = []
        self.feature_consistency_history = []
        
    def forward(self):
        pass
    
    def compute_miou_seg6(self, x_pred, x_output):
        """
        计算seg6任务的mIoU（排除背景类）
        
        Args:
            x_pred: 预测结果 [B, 6, H, W]
            x_output: 真值标签 [B, H, W]
        """
        _, pred = torch.max(x_pred, 1)
        pred = pred.view(-1)
        target = x_output.view(-1)
        
        # 排除ignore_index=-1和背景类=0
        valid_mask = (target != -1) & (target != 255)
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        if len(pred) == 0:
            return torch.tensor(0.0)
        
        # 计算每个前景类的IoU（类别1-5）
        ious = []
        for class_id in range(1, self.class_nb_seg6):  # 排除背景类0
            pred_mask = (pred == class_id)
            target_mask = (target == class_id)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou)
        
        if len(ious) > 0:
            return torch.stack(ious).mean()
        else:
            return torch.tensor(0.0)
    
    def compute_miou_seg9(self, x_pred, x_output):
        """
        计算seg9任务的mIoU（排除背景类）
        
        Args:
            x_pred: 预测结果 [B, 9, H, W]
            x_output: 真值标签 [B, H, W]
        """
        _, pred = torch.max(x_pred, 1)
        pred = pred.view(-1)
        target = x_output.view(-1)
        
        # 排除ignore_index=-1和背景类=0
        valid_mask = (target != -1) & (target != 255)
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        if len(pred) == 0:
            return torch.tensor(0.0)
        
        # 计算每个前景类的IoU（类别1-8）
        ious = []
        for class_id in range(1, self.class_nb_seg9):  # 排除背景类0
            pred_mask = (pred == class_id)
            target_mask = (target == class_id)
            
            intersection = (pred_mask & target_mask).sum().float()
            union = (pred_mask | target_mask).sum().float()
            
            if union > 0:
                iou = intersection / union
                ious.append(iou)
        
        if len(ious) > 0:
            return torch.stack(ious).mean()
        else:
            return torch.tensor(0.0)
    
    def compute_accuracy_seg6(self, x_pred, x_output):
        """
        计算seg6任务的像素准确率
        """
        _, pred = torch.max(x_pred, 1)
        pred = pred.view(-1)
        target = x_output.view(-1)
        
        # 排除ignore_index
        valid_mask = (target != -1) & (target != 255)
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        if len(pred) == 0:
            return torch.tensor(0.0)
        
        correct = (pred == target).sum().float()
        total = len(pred)
        accuracy = correct / total
        
        return accuracy
    
    def compute_accuracy_seg9(self, x_pred, x_output):
        """
        计算seg9任务的像素准确率
        """
        _, pred = torch.max(x_pred, 1)
        pred = pred.view(-1)
        target = x_output.view(-1)
        
        # 排除ignore_index
        valid_mask = (target != -1) & (target != 255)
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        if len(pred) == 0:
            return torch.tensor(0.0)
        
        correct = (pred == target).sum().float()
        total = len(pred)
        accuracy = correct / total
        
        return accuracy
    
    def compute_detailed_metrics_seg6(self, x_pred, x_output):
        """
        计算seg6任务的详细指标（包括每个类别的IoU、F1等）
        """
        _, pred = torch.max(x_pred, 1)
        pred = pred.view(-1).cpu().numpy()
        target = x_output.view(-1).cpu().numpy()
        
        # 排除ignore_index
        valid_mask = (target != -1) & (target != 255)
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        if len(pred) == 0:
            return {
                'per_class_iou': [0.0] * self.class_nb_seg6,
                'per_class_f1': [0.0] * self.class_nb_seg6,
                'mean_iou': 0.0,
                'mean_f1': 0.0,
                'overall_accuracy': 0.0
            }
        
        # 计算混淆矩阵
        cm = confusion_matrix(target, pred, labels=list(range(self.class_nb_seg6)))
        
        # 计算每个类别的IoU和F1
        per_class_iou = []
        per_class_f1 = []
        
        for i in range(self.class_nb_seg6):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            # IoU
            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
            else:
                iou = 0.0
            per_class_iou.append(iou)
            
            # F1
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            per_class_f1.append(f1)
        
        # 排除背景类计算均值
        mean_iou = np.mean(per_class_iou[1:]) if len(per_class_iou) > 1 else 0.0
        mean_f1 = np.mean(per_class_f1[1:]) if len(per_class_f1) > 1 else 0.0
        
        # 整体准确率
        overall_accuracy = np.trace(cm) / np.sum(cm)
        
        return {
            'per_class_iou': per_class_iou,
            'per_class_f1': per_class_f1,
            'mean_iou': mean_iou,
            'mean_f1': mean_f1,
            'overall_accuracy': overall_accuracy,
            'confusion_matrix': cm
        }
    
    def compute_detailed_metrics_seg9(self, x_pred, x_output):
        """
        计算seg9任务的详细指标（包括每个类别的IoU、F1等）
        """
        _, pred = torch.max(x_pred, 1)
        pred = pred.view(-1).cpu().numpy()
        target = x_output.view(-1).cpu().numpy()
        
        # 排除ignore_index
        valid_mask = (target != -1) & (target != 255)
        pred = pred[valid_mask]
        target = target[valid_mask]
        
        if len(pred) == 0:
            return {
                'per_class_iou': [0.0] * self.class_nb_seg9,
                'per_class_f1': [0.0] * self.class_nb_seg9,
                'mean_iou': 0.0,
                'mean_f1': 0.0,
                'overall_accuracy': 0.0
            }
        
        # 计算混淆矩阵
        cm = confusion_matrix(target, pred, labels=list(range(self.class_nb_seg9)))
        
        # 计算每个类别的IoU和F1
        per_class_iou = []
        per_class_f1 = []
        
        for i in range(self.class_nb_seg9):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            
            # IoU
            if tp + fp + fn > 0:
                iou = tp / (tp + fp + fn)
            else:
                iou = 0.0
            per_class_iou.append(iou)
            
            # F1
            if tp + fp > 0:
                precision = tp / (tp + fp)
            else:
                precision = 0.0
            
            if tp + fn > 0:
                recall = tp / (tp + fn)
            else:
                recall = 0.0
            
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            per_class_f1.append(f1)
        
        # 排除背景类计算均值
        mean_iou = np.mean(per_class_iou[1:]) if len(per_class_iou) > 1 else 0.0
        mean_f1 = np.mean(per_class_f1[1:]) if len(per_class_f1) > 1 else 0.0
        
        # 整体准确率
        overall_accuracy = np.trace(cm) / np.sum(cm)
        
        return {
            'per_class_iou': per_class_iou,
            'per_class_f1': per_class_f1,
            'mean_iou': mean_iou,
            'mean_f1': mean_f1,
            'overall_accuracy': overall_accuracy,
            'confusion_matrix': cm
        }
    
    def evaluate_dictionary_quality(self, dictionary_stats: Dict):
        """
        新增：评估动态词典学习的质量
        
        Args:
            dictionary_stats: 来自DynamicDictionary.get_dictionary_statistics()的统计信息
        
        Returns:
            质量评估结果
        """
        if not dictionary_stats:
            return {'quality_score': 0.0, 'status': 'no_stats'}
        
        quality_metrics = {}
        
        # 评估词典使用均衡性
        if 'usage_freq' in dictionary_stats:
            usage_freq = dictionary_stats['usage_freq']
            if torch.is_tensor(usage_freq):
                usage_std = torch.std(usage_freq.float())
                usage_mean = torch.mean(usage_freq.float())
                if usage_mean > 0:
                    usage_balance = 1.0 / (1.0 + usage_std / usage_mean)
                else:
                    usage_balance = 0.0
                quality_metrics['usage_balance'] = float(usage_balance)
        
        # 评估词典原子的范数分布
        if 'class_dict_norms' in dictionary_stats:
            norms = dictionary_stats['class_dict_norms']
            if torch.is_tensor(norms):
                norm_std = torch.std(norms)
                norm_mean = torch.mean(norms)
                if norm_mean > 0:
                    norm_consistency = 1.0 / (1.0 + norm_std / norm_mean)
                else:
                    norm_consistency = 0.0
                quality_metrics['norm_consistency'] = float(norm_consistency)
        
        # 评估词典相干性
        if 'dictionary_coherence' in dictionary_stats:
            coherence = dictionary_stats['dictionary_coherence']
            if torch.is_tensor(coherence):
                # 相干性越低，质量越好
                coherence_quality = torch.sigmoid(-coherence * 10)
                quality_metrics['coherence_quality'] = float(coherence_quality)
        
        # 计算综合质量分数
        if quality_metrics:
            quality_score = np.mean(list(quality_metrics.values()))
            self.dict_quality_history.append(quality_score)
            
            # 保持历史记录长度
            if len(self.dict_quality_history) > 100:
                self.dict_quality_history = self.dict_quality_history[-100:]
            
            quality_metrics['quality_score'] = quality_score
            quality_metrics['quality_trend'] = self._compute_quality_trend()
            quality_metrics['status'] = 'good' if quality_score > 0.7 else 'moderate' if quality_score > 0.4 else 'poor'
        else:
            quality_metrics = {'quality_score': 0.0, 'status': 'no_valid_metrics'}
        
        return quality_metrics
    
    def _compute_quality_trend(self):
        """
        计算词典质量的趋势
        """
        if len(self.dict_quality_history) < 5:
            return 'insufficient_data'
        
        recent_scores = self.dict_quality_history[-5:]
        earlier_scores = self.dict_quality_history[-10:-5] if len(self.dict_quality_history) >= 10 else self.dict_quality_history[:-5]
        
        if len(earlier_scores) == 0:
            return 'insufficient_data'
        
        recent_mean = np.mean(recent_scores)
        earlier_mean = np.mean(earlier_scores)
        
        if recent_mean > earlier_mean + 0.05:
            return 'improving'
        elif recent_mean < earlier_mean - 0.05:
            return 'declining'
        else:
            return 'stable'
    
    def compute_feature_consistency(self, features_before, features_after):
        """
        新增：计算动态词典增强前后特征的一致性
        
        Args:
            features_before: 增强前的特征 [B, C, H, W]
            features_after: 增强后的特征 [B, C, H, W]
        
        Returns:
            特征一致性指标
        """
        try:
            # 计算特征相似度
            features_before_flat = features_before.view(features_before.size(0), -1)
            features_after_flat = features_after.view(features_after.size(0), -1)
            
            # 归一化特征
            features_before_norm = torch.nn.functional.normalize(features_before_flat, p=2, dim=1)
            features_after_norm = torch.nn.functional.normalize(features_after_flat, p=2, dim=1)
            
            # 计算余弦相似度
            cosine_sim = torch.sum(features_before_norm * features_after_norm, dim=1)
            mean_cosine_sim = torch.mean(cosine_sim)
            
            # 计算特征变化幅度
            feature_change = torch.norm(features_after_flat - features_before_flat, p=2, dim=1)
            mean_feature_change = torch.mean(feature_change)
            
            consistency_metrics = {
                'cosine_similarity': float(mean_cosine_sim),
                'feature_change_magnitude': float(mean_feature_change),
                'consistency_score': float(mean_cosine_sim * torch.sigmoid(-mean_feature_change))
            }
            
            self.feature_consistency_history.append(consistency_metrics['consistency_score'])
            
            # 保持历史记录长度
            if len(self.feature_consistency_history) > 100:
                self.feature_consistency_history = self.feature_consistency_history[-100:]
            
            return consistency_metrics
            
        except Exception as e:
            return {
                'cosine_similarity': 0.0,
                'feature_change_magnitude': 0.0,
                'consistency_score': 0.0,
                'error': str(e)
            }
    
    def get_comprehensive_report(self, seg6_pred, seg6_gt, seg9_pred, seg9_gt, dictionary_stats=None):
        """
        新增：生成综合评估报告
        
        Args:
            seg6_pred: seg6任务预测结果
            seg6_gt: seg6任务真值
            seg9_pred: seg9任务预测结果
            seg9_gt: seg9任务真值
            dictionary_stats: 动态词典统计信息
        
        Returns:
            综合评估报告
        """
        report = {}
        
        # 基础任务指标
        report['seg6_metrics'] = self.compute_detailed_metrics_seg6(seg6_pred, seg6_gt)
        report['seg9_metrics'] = self.compute_detailed_metrics_seg9(seg9_pred, seg9_gt)
        
        # 多任务综合指标
        avg_miou = (report['seg6_metrics']['mean_iou'] + report['seg9_metrics']['mean_iou']) / 2
        avg_f1 = (report['seg6_metrics']['mean_f1'] + report['seg9_metrics']['mean_f1']) / 2
        avg_accuracy = (report['seg6_metrics']['overall_accuracy'] + report['seg9_metrics']['overall_accuracy']) / 2
        
        report['multitask_metrics'] = {
            'average_miou': avg_miou,
            'average_f1': avg_f1,
            'average_accuracy': avg_accuracy,
            'task_balance': 1.0 - abs(report['seg6_metrics']['mean_iou'] - report['seg9_metrics']['mean_iou'])
        }
        
        # 动态词典质量评估
        if dictionary_stats:
            report['dictionary_quality'] = self.evaluate_dictionary_quality(dictionary_stats)
        
        # 整体性能评分
        performance_score = (avg_miou * 0.5 + avg_f1 * 0.3 + avg_accuracy * 0.2)
        
        if dictionary_stats and 'quality_score' in report.get('dictionary_quality', {}):
            dict_quality = report['dictionary_quality']['quality_score']
            # 将词典质量作为性能加权因子
            performance_score = performance_score * (0.8 + 0.2 * dict_quality)
        
        report['overall_performance'] = {
            'performance_score': performance_score,
            'grade': 'A' if performance_score > 0.8 else 'B' if performance_score > 0.6 else 'C' if performance_score > 0.4 else 'D'
        }
        
        return report
    
    def reset_history(self):
        """
        重置历史记录
        """
        self.dict_quality_history = []
        self.feature_consistency_history = []
    
    def get_training_insights(self):
        """
        新增：获取训练洞察和建议
        
        Returns:
            训练建议和分析
        """
        insights = {
            'recommendations': [],
            'warnings': [],
            'status': 'normal'
        }
        
        # 分析词典质量趋势
        if len(self.dict_quality_history) >= 10:
            recent_quality = np.mean(self.dict_quality_history[-5:])
            
            if recent_quality < 0.3:
                insights['warnings'].append("词典质量较低，可能需要调整学习率或词典大小")
                insights['status'] = 'poor_dictionary'
            elif recent_quality > 0.8:
                insights['recommendations'].append("词典学习效果良好，可以考虑增加词典复杂度")
        
        # 分析特征一致性趋势
        if len(self.feature_consistency_history) >= 10:
            recent_consistency = np.mean(self.feature_consistency_history[-5:])
            
            if recent_consistency < 0.5:
                insights['warnings'].append("特征增强效果不稳定，建议检查词典更新策略")
            elif recent_consistency > 0.85:
                insights['recommendations'].append("特征增强效果优秀，当前设置较为合适")
        
        # 综合建议
        if not insights['warnings'] and not insights['recommendations']:
            insights['recommendations'].append("训练状态正常，继续当前设置")
        
        return insights