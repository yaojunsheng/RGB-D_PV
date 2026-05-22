"""
增强的Height损失函数 - 解决损失过高问题
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EnhancedHeightLoss(nn.Module):
    """
    增强的Height损失函数，解决损失过高的问题
    """
    
    def __init__(self, loss_type='smooth_l1', reduction='mean', ignore_value=None):
        super(EnhancedHeightLoss, self).__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        self.ignore_value = ignore_value
        
        print(f"EnhancedHeightLoss initialized with loss_type={loss_type}, reduction={reduction}")
    
    def forward(self, pred, target):
        """
        计算height损失
        
        Args:
            pred: 预测值 [B, 1, H, W] 范围应该在[0,1]
            target: 目标值 [B, 1, H, W] 已归一化到[0,1]
        """
        
        # 数据范围检查和调试
        with torch.no_grad():
            pred_min, pred_max = pred.min().item(), pred.max().item()
            target_min, target_max = target.min().item(), target.max().item()
            pred_mean, target_mean = pred.mean().item(), target.mean().item()
            
            # 如果数据范围异常，打印警告
            if pred_min < -0.1 or pred_max > 1.1:
                print(f"⚠️  WARNING: Pred range [{pred_min:.4f}, {pred_max:.4f}] outside expected [0,1]")
            if target_min < -0.1 or target_max > 1.1:
                print(f"⚠️  WARNING: Target range [{target_min:.4f}, {target_max:.4f}] outside expected [0,1]")
        
        # 创建有效像素掩码（如果指定了ignore_value）
        if self.ignore_value is not None:
            valid_mask = (target != self.ignore_value).float()
        else:
            # 假设0值可能是无效的（根据实际数据调整）
            valid_mask = (target > 1e-6).float()
        
        valid_pixels = valid_mask.sum()
        
        if valid_pixels == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        # 计算不同类型的损失
        if self.loss_type == 'l1':
            loss = F.l1_loss(pred * valid_mask, target * valid_mask, reduction='none')
        elif self.loss_type == 'l2' or self.loss_type == 'mse':
            loss = F.mse_loss(pred * valid_mask, target * valid_mask, reduction='none')
        elif self.loss_type == 'smooth_l1':
            loss = F.smooth_l1_loss(pred * valid_mask, target * valid_mask, reduction='none', beta=0.1)
        elif self.loss_type == 'huber':
            # Huber损失，对异常值更鲁棒
            diff = torch.abs(pred - target) * valid_mask
            huber_loss = torch.where(diff < 0.1, 
                                   0.5 * diff.pow(2), 
                                   0.1 * (diff - 0.05))
            loss = huber_loss
        elif self.loss_type == 'scaled_l1':
            # 缩放L1损失，适用于小范围数据
            loss = F.l1_loss(pred * valid_mask, target * valid_mask, reduction='none') * 10.0
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # 应用reduction
        if self.reduction == 'mean':
            if valid_pixels > 0:
                return loss.sum() / valid_pixels
            else:
                return torch.tensor(0.0, device=pred.device, requires_grad=True)
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AdaptiveHeightLoss(nn.Module):
    """
    自适应Height损失函数，根据训练阶段调整权重
    """
    
    def __init__(self, initial_weight=1.0, final_weight=1.0, adaptive_epochs=50):
        super(AdaptiveHeightLoss, self).__init__()
        self.initial_weight = initial_weight
        self.final_weight = final_weight
        self.adaptive_epochs = adaptive_epochs
        self.base_loss = EnhancedHeightLoss(loss_type='smooth_l1')
        
        self.loss_history = []
        self.epoch = 0
        
    def set_epoch(self, epoch):
        """设置当前epoch"""
        self.epoch = epoch
    
    def get_current_weight(self):
        """获取当前权重"""
        if self.epoch < self.adaptive_epochs:
            # 线性插值权重
            alpha = self.epoch / self.adaptive_epochs
            weight = self.initial_weight * (1 - alpha) + self.final_weight * alpha
        else:
            weight = self.final_weight
        
        return weight
    
    def forward(self, pred, target):
        """计算自适应权重的height损失"""
        base_loss = self.base_loss(pred, target)
        weight = self.get_current_weight()
        
        # 记录损失历史
        self.loss_history.append(base_loss.item())
        
        # 如果损失历史过长，保留最近的1000个
        if len(self.loss_history) > 1000:
            self.loss_history = self.loss_history[-1000:]
        
        # 动态调整权重（如果损失过高）
        if len(self.loss_history) > 10:
            recent_avg_loss = sum(self.loss_history[-10:]) / 10
            if recent_avg_loss > 2.0:  # 如果平均损失过高
                weight *= 0.5  # 减少权重
                print(f"⚠️  Height loss too high ({recent_avg_loss:.4f}), reducing weight to {weight:.4f}")
        
        return base_loss * weight


def test_height_loss():
    """测试height损失函数"""
    print("Testing Height Loss Functions...")
    
    # 创建测试数据
    batch_size, height, width = 2, 32, 32
    
    # 模拟归一化后的数据 [0,1]
    pred = torch.sigmoid(torch.randn(batch_size, 1, height, width))  # 模拟sigmoid输出
    target = torch.rand(batch_size, 1, height, width)  # 归一化后的目标
    
    print(f"Pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
    print(f"Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")
    
    # 测试不同损失函数
    loss_functions = {
        'Enhanced L1': EnhancedHeightLoss(loss_type='l1'),
        'Enhanced Smooth L1': EnhancedHeightLoss(loss_type='smooth_l1'),
        'Enhanced Huber': EnhancedHeightLoss(loss_type='huber'),
        'Enhanced Scaled L1': EnhancedHeightLoss(loss_type='scaled_l1'),
        'Standard L1': nn.L1Loss(),
        'Standard MSE': nn.MSELoss(),
    }
    
    print("\nLoss Function Comparison:")
    print("-" * 40)
    
    for name, loss_fn in loss_functions.items():
        try:
            if 'Standard' in name:
                loss_value = loss_fn(pred, target)
            else:
                loss_value = loss_fn(pred, target)
            print(f"{name:20}: {loss_value.item():.6f}")
        except Exception as e:
            print(f"{name:20}: ERROR - {e}")
    
    # 测试自适应损失
    print("\nTesting Adaptive Loss:")
    adaptive_loss = AdaptiveHeightLoss(initial_weight=0.1, final_weight=1.0)
    
    for epoch in [0, 10, 25, 50, 100]:
        adaptive_loss.set_epoch(epoch)
        loss_value = adaptive_loss(pred, target)
        weight = adaptive_loss.get_current_weight()
        print(f"Epoch {epoch:3d}: Loss={loss_value.item():.6f}, Weight={weight:.3f}")


class HeightLossDebugger:
    """Height损失调试器"""
    
    def __init__(self):
        self.loss_history = []
        self.pred_history = []
        self.target_history = []
        
    def log_batch(self, pred, target, loss_value, batch_idx=None):
        """记录batch信息"""
        with torch.no_grad():
            self.loss_history.append(loss_value)
            
            pred_stats = {
                'min': pred.min().item(),
                'max': pred.max().item(),
                'mean': pred.mean().item(),
                'std': pred.std().item()
            }
            
            target_stats = {
                'min': target.min().item(),
                'max': target.max().item(),
                'mean': target.mean().item(),
                'std': target.std().item()
            }
            
            self.pred_history.append(pred_stats)
            self.target_history.append(target_stats)
    
    def print_diagnosis(self):
        """打印诊断信息"""
        if not self.loss_history:
            print("No data to diagnose")
            return
        
        print("\n" + "="*60)
        print("HEIGHT LOSS DIAGNOSIS")
        print("="*60)
        
        # 损失统计
        recent_losses = self.loss_history[-10:] if len(self.loss_history) >= 10 else self.loss_history
        avg_loss = sum(recent_losses) / len(recent_losses)
        max_loss = max(recent_losses)
        min_loss = min(recent_losses)
        
        print(f"Recent Loss Stats (last {len(recent_losses)} batches):")
        print(f"  Average: {avg_loss:.6f}")
        print(f"  Min: {min_loss:.6f}")
        print(f"  Max: {max_loss:.6f}")
        
        # 预测统计
        recent_pred = self.pred_history[-1]
        recent_target = self.target_history[-1]
        
        print(f"\nLatest Batch Data Range:")
        print(f"  Predictions: [{recent_pred['min']:.4f}, {recent_pred['max']:.4f}] "
              f"(mean: {recent_pred['mean']:.4f}, std: {recent_pred['std']:.4f})")
        print(f"  Targets:     [{recent_target['min']:.4f}, {recent_target['max']:.4f}] "
              f"(mean: {recent_target['mean']:.4f}, std: {recent_target['std']:.4f})")
        
        # 诊断问题
        print(f"\nDiagnosis:")
        
        if avg_loss > 1.0:
            print("  ❌ HIGH LOSS detected!")
            print("     Possible causes:")
            print("     - Data range mismatch between pred and target")
            print("     - Model not converging")
            print("     - Learning rate too high")
            print("     - Wrong loss function choice")
        
        if recent_pred['max'] > 1.2 or recent_pred['min'] < -0.2:
            print("  ❌ PREDICTION RANGE issue!")
            print("     - Predictions outside expected [0,1] range")
            print("     - Check if sigmoid is applied correctly")
        
        if recent_target['max'] > 1.2 or recent_target['min'] < -0.2:
            print("  ❌ TARGET RANGE issue!")
            print("     - Targets outside expected [0,1] range")
            print("     - Check data normalization")
        
        mean_diff = abs(recent_pred['mean'] - recent_target['mean'])
        if mean_diff > 0.3:
            print(f"  ❌ LARGE MEAN DIFFERENCE: {mean_diff:.4f}")
            print("     - Systematic bias between predictions and targets")
        
        if avg_loss < 0.1:
            print("  ✅ Loss looks reasonable")
        
        print("="*60)
    
    def suggest_fixes(self):
        """建议修复方案"""
        if not self.loss_history:
            return
        
        avg_loss = sum(self.loss_history[-10:]) / min(10, len(self.loss_history))
        
        print("\nSUGGESTED FIXES:")
        print("-" * 30)
        
        if avg_loss > 1.0:
            print("1. Try different loss functions:")
            print("   - EnhancedHeightLoss with 'smooth_l1' or 'huber'")
            print("   - AdaptiveHeightLoss for dynamic weight adjustment")
            print("\n2. Check data preprocessing:")
            print("   - Verify height data normalization to [0,1]")
            print("   - Ensure model output uses sigmoid activation")
            print("\n3. Adjust training parameters:")
            print("   - Lower learning rate for height task")
            print("   - Use separate optimizer for height head")
            print("   - Apply gradient clipping")


if __name__ == "__main__":
    test_height_loss()