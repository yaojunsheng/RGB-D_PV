import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ComputeRoofMetric(nn.Module):
    def __init__(self, class_nb_seg6=6, class_nb_seg9=6):
        super(ComputeRoofMetric, self).__init__()
        self.class_nb_seg6 = class_nb_seg6  # 包含背景
        self.class_nb_seg9 = class_nb_seg9  # 包含背景
        # 实际有效类别数（排除背景）
        self.effective_seg6 = class_nb_seg6 - 1  # 5类
        self.effective_seg9 = class_nb_seg9 - 1  # 8类
        
    def forward(self):
        pass
        
    def compute_miou(self, x_pred, x_output, num_classes, exclude_background=True):
        """
        计算平均IoU，默认排除背景类（类别0）
        num_classes: 包含背景的总类别数
        """
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        
        total_miou = 0.0
        valid_batches = 0
        
        for i in range(batch_size):
            true_class = 0  # 有效类别计数（排除背景）
            class_iou_sum = 0.0
            
            # 遍历所有类别，排除背景类（j=0）
            start_class = 1 if exclude_background else 0
            for j in range(start_class, num_classes):  # 从1开始计算，跳过背景
                pred_mask = torch.eq(
                    x_pred_label[i], 
                    j * torch.ones(x_pred_label[i].shape).type(torch.LongTensor).cuda()
                )
                true_mask = torch.eq(
                    x_output_label[i], 
                    j * torch.ones(x_output_label[i].shape).type(torch.LongTensor).cuda()
                )
                mask_comb = pred_mask.type(torch.FloatTensor) + true_mask.type(torch.FloatTensor)
                union = torch.sum((mask_comb > 0).type(torch.FloatTensor))
                intsec = torch.sum((mask_comb > 1).type(torch.FloatTensor))
                
                if union == 0:
                    continue  # 跳过无像素的类别
                    
                class_iou_sum += intsec / union
                true_class += 1
            
            if true_class > 0:  # 避免除以0
                total_miou += class_iou_sum / true_class
                valid_batches += 1
        
        if valid_batches > 0:
            return total_miou / valid_batches
        else:
            return torch.tensor(0.0)
    
    def compute_miou_seg6(self, x_pred, x_output):
        """计算seg6任务的mIoU（排除背景类，仅计算1-5类）"""
        return self.compute_miou(x_pred, x_output, self.class_nb_seg6, exclude_background=True)
    
    def compute_miou_seg9(self, x_pred, x_output):
        """计算seg9任务的mIoU（排除背景类，仅计算1-8类）"""
        return self.compute_miou(x_pred, x_output, self.class_nb_seg9, exclude_background=True)
    
    def compute_pixel_accuracy(self, x_pred, x_output):
        """
        Compute pixel accuracy for segmentation task
        """
        _, x_pred_label = torch.max(x_pred, dim=1)
        x_output_label = x_output
        batch_size = x_pred.size(0)
        
        total_accuracy = 0.0
        
        for i in range(batch_size):
            correct_pixels = torch.sum(torch.eq(x_pred_label[i], x_output_label[i]).type(torch.FloatTensor))
            total_pixels = torch.sum((x_output_label[i] >= 0).type(torch.FloatTensor))
            
            if total_pixels > 0:
                total_accuracy += correct_pixels / total_pixels
        
        return total_accuracy / batch_size
    
    def compute_accuracy_seg6(self, x_pred, x_output):
        """Compute pixel accuracy for seg6 task"""
        return self.compute_pixel_accuracy(x_pred, x_output)
    
    def compute_accuracy_seg9(self, x_pred, x_output):
        """Compute pixel accuracy for seg9 task"""
        return self.compute_pixel_accuracy(x_pred, x_output)