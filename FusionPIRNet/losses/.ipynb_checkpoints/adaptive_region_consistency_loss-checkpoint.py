import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.modules.module import Module
import copy


class FocalCrossEntropyLoss(nn.Module):
    """Focal Loss for handling severe class imbalance"""
    def __init__(self, alpha=None, gamma=2.0, ignore_index=255, background_suppression=0.1):
        super(FocalCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.background_suppression = background_suppression
        
    def forward(self, pred, target):
        if target.dim() == 4:
            target = target.squeeze(1)
        target = target.long()
        
        valid_mask = (target != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        log_probs = F.log_softmax(pred, dim=1)
        ce_loss = F.nll_loss(log_probs, target, ignore_index=self.ignore_index, reduction='none')
        
        probs = F.softmax(pred, dim=1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1)
        
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply background suppression
        background_mask = (target == 0)
        focal_weight = torch.where(
            background_mask & valid_mask,
            focal_weight * self.background_suppression,
            focal_weight
        )
        
        focal_loss = focal_weight * ce_loss
        
        return focal_loss[valid_mask].mean()


class DiceFocalLoss(nn.Module):
    """Combined Dice Loss and Focal Loss for small object segmentation"""
    
    def __init__(self, alpha=0.5, gamma=2.0, ignore_index=255, dice_weight=0.3, focal_weight=0.7):
        super(DiceFocalLoss, self).__init__()
        self.focal_loss = FocalCrossEntropyLoss(alpha=alpha, gamma=gamma, ignore_index=ignore_index)
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ignore_index = ignore_index
        
    def dice_loss(self, pred, target):
        pred_softmax = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()
        
        valid_mask = (target != self.ignore_index).unsqueeze(1).float()
        pred_softmax = pred_softmax * valid_mask
        target_one_hot = target_one_hot * valid_mask
        
        intersection = (pred_softmax * target_one_hot).sum(dim=(2, 3))
        union = pred_softmax.sum(dim=(2, 3)) + target_one_hot.sum(dim=(2, 3))
        
        dice = (2 * intersection + 1e-8) / (union + 1e-8)
        dice_loss = 1 - dice.mean()
        
        return dice_loss
    
    def forward(self, pred, target):
        focal_loss = self.focal_loss(pred, target)
        dice_loss = self.dice_loss(pred, target)
        
        combined_loss = self.focal_weight * focal_loss + self.dice_weight * dice_loss
        return combined_loss


class OnlineHardExampleMining(nn.Module):
    """Online Hard Example Mining for focusing on difficult samples"""
    
    def __init__(self, base_loss_fn, keep_ratio=0.7, ignore_index=255):
        super(OnlineHardExampleMining, self).__init__()
        self.base_loss_fn = base_loss_fn
        self.keep_ratio = keep_ratio
        self.ignore_index = ignore_index
        
    def forward(self, pred, target):
        pixel_losses = F.cross_entropy(pred, target, ignore_index=self.ignore_index, reduction='none')
        
        valid_mask = (target != self.ignore_index)
        if not valid_mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        valid_losses = pixel_losses[valid_mask]
        
        num_hard = max(1, int(self.keep_ratio * valid_losses.numel()))
        hard_losses, _ = torch.topk(valid_losses, num_hard, sorted=False)
        
        return hard_losses.mean()


class AdaptiveLossSelector:
    """Adaptive loss selector based on data quality analysis"""
    
    def __init__(self, ignore_index=255):
        self.ignore_index = ignore_index
        self.loss_strategies = {}
        
    def analyze_batch_quality(self, target, task_name):
        """Analyze batch quality for dynamic loss selection"""
        if target is None:
            return {'severity': 'no_data', 'background_ratio': 1.0}
        
        if target.dim() == 4:
            target = target.squeeze(1)
        
        valid_mask = (target != self.ignore_index)
        if not valid_mask.any():
            return {'severity': 'no_data', 'background_ratio': 1.0}
        
        valid_pixels = valid_mask.sum().item()
        background_pixels = ((target == 0) & valid_mask).sum().item()
        background_ratio = background_pixels / valid_pixels if valid_pixels > 0 else 1.0
        
        if background_ratio > 0.98:
            severity = 'extreme'
        elif background_ratio > 0.95:
            severity = 'severe'
        elif background_ratio > 0.90:
            severity = 'moderate'
        else:
            severity = 'normal'
        
        return {
            'severity': severity,
            'background_ratio': background_ratio,
            'valid_pixels': valid_pixels,
            'foreground_pixels': valid_pixels - background_pixels
        }
    
    def original_nll_loss(self, pred, target):
        """Original NLL loss from roof_loss_functions.py"""
        # Apply log_softmax first, then nll_loss (same as original implementation)
        pred_log_softmax = F.log_softmax(pred, dim=1)
        return F.nll_loss(pred_log_softmax, target.long(), ignore_index=-1)
    
    def get_loss_strategy(self, task_name, batch_quality):
        """Get optimal loss strategy based on batch quality"""
        severity = batch_quality['severity']
        
        if severity == 'extreme':
            if task_name not in self.loss_strategies or self.loss_strategies[task_name]['type'] != 'dice_focal_extreme':
                self.loss_strategies[task_name] = {
                    'type': 'dice_focal_extreme',
                    'loss_fn': DiceFocalLoss(
                        alpha=1.0, gamma=3.0, ignore_index=self.ignore_index,
                        dice_weight=0.4, focal_weight=0.6
                    )
                }
        
        elif severity == 'severe':
            if task_name not in self.loss_strategies or self.loss_strategies[task_name]['type'] != 'focal_suppression':
                self.loss_strategies[task_name] = {
                    'type': 'focal_suppression', 
                    'loss_fn': FocalCrossEntropyLoss(
                        alpha=1.0, gamma=2.5, ignore_index=self.ignore_index,
                        background_suppression=0.05
                    )
                }
        
        elif severity == 'moderate':
            if task_name not in self.loss_strategies or self.loss_strategies[task_name]['type'] != 'hard_mining':
                base_focal = FocalCrossEntropyLoss(alpha=1.0, gamma=2.0, ignore_index=self.ignore_index)
                self.loss_strategies[task_name] = {
                    'type': 'hard_mining',
                    'loss_fn': OnlineHardExampleMining(
                        base_focal, keep_ratio=0.7, ignore_index=self.ignore_index
                    )
                }
        
        else:  # severity == 'normal'
            if task_name not in self.loss_strategies or self.loss_strategies[task_name]['type'] != 'original_nll':
                self.loss_strategies[task_name] = {
                    'type': 'original_nll',
                    'loss_fn': self.original_nll_loss
                }
        
        return self.loss_strategies[task_name]['loss_fn']


class RegionContrastiveLoss(nn.Module):
    """Region-based contrastive loss for cross-task consistency"""
    
    def __init__(self, tasks=['seg6', 'seg9'], feature_dim=512):
        super(RegionContrastiveLoss, self).__init__()
        self.eps = 1e-6
        self.contra_temp = 0.5
        self.simi_temp = 0.4
        self.feature_bank = []
        self.capacity = 4
        self.count = 0
        self.edge_feature_bank = []
        self.nonedge_feature_bank = []
        
        self.feature_bank_u = []
        self.feature_bank_s = []
        self.bankR_cnt = 1000

    def forward(self, map_s, map_t, mask=None, index=None):
        """Compute region contrastive loss"""
        region_list_x, ux_list, sx_list, region_list_y, uy_list, sy_list = self.get_region_features(map_s, map_t, mask)
        
        self.feature_bank_u += ux_list + uy_list
        self.feature_bank_s += sx_list + sy_list
        self.feature_bank_u = self.feature_bank_u[-self.bankR_cnt:]
        self.feature_bank_s = self.feature_bank_s[-self.bankR_cnt:]
        
        loss = 0
        numR = len(region_list_x)
        
        if numR == 0:
            return torch.tensor(0.01, device=map_s.device, requires_grad=True)
            
        for i in range(numR):
            loss += self.compute_contrastive_loss(
                region_list_x[i], region_list_y[i], 
                [ux_list[i], sx_list[i]], [uy_list[i], sy_list[i]],
                [self.feature_bank_u, self.feature_bank_s]
            )
            
        loss = loss / numR
        return loss        

    def get_region_features(self, x, y, sam):
        """Extract region features using SAM masks"""
        samv = sam[0] if isinstance(sam, (list, tuple)) else sam
        
        region_list_x, region_list_y = [], []
        ux_list, uy_list, sx_list, sy_list = [], [], [], []
        
        # Get the unique region IDs from SAM mask (dynamic range)
        unique_regions = torch.unique(samv)
        max_regions = min(len(unique_regions), 256)  # Process up to 256 regions

        for region_id in unique_regions[:max_regions]:
            if region_id < 0:  # Skip invalid region IDs
                continue
                
            index = (samv == region_id).nonzero(as_tuple=True)
            
            if len(index[0]) < 10:  # Skip small regions
                continue
            
            region_x = x[0, :, index[0], index[1]]
            region_y = y[0, :, index[0], index[1]]
            
            try:
                u_x = np.mean(region_x.detach().cpu().numpy(), axis=1)
                sigma_x = np.cov(region_x.detach().cpu().numpy())
                u_y = np.mean(region_y.detach().cpu().numpy(), axis=1)
                sigma_y = np.cov(region_y.detach().cpu().numpy())
                
                # Check covariance matrix dimensions
                if sigma_x.ndim == 0:
                    sigma_x = np.array([[sigma_x]])
                if sigma_y.ndim == 0:
                    sigma_y = np.array([[sigma_y]])
                    
                # Check for NaN values
                if (np.isnan(u_x).any() or np.isnan(u_y).any() or 
                    np.isnan(sigma_x).any() or np.isnan(sigma_y).any()):
                    continue
                
                region_list_x.append(region_x)
                region_list_y.append(region_y)
                ux_list.append(u_x)
                sx_list.append(sigma_x)
                uy_list.append(u_y)
                sy_list.append(sigma_y)
                
            except Exception as e:
                continue

        return region_list_x, ux_list, sx_list, region_list_y, uy_list, sy_list

    def compute_contrastive_loss(self, anchor, pos_pair, an_vs, n_vs, neg_vs, temp=1):
        """Compute contrastive loss with Frechet distance"""
        try:
            pos = -self.calculate_frechet_distance(an_vs[0], an_vs[1], n_vs[0], n_vs[1]) / 5.4 
            maxp = np.max(pos, axis=0, keepdims=True)
            exp_pos = np.exp(pos - maxp).mean()
            
            if len(neg_vs[0]) < 2 or len(neg_vs[1]) < 2:
                return torch.tensor(max(0.01, abs(pos) * 0.1), device=anchor.device, requires_grad=True)
            
            negNum = len(neg_vs)
            
            neg = 0
            for i in range(negNum):
                if i < len(neg_vs[0]) and i < len(neg_vs[1]):
                    try:
                        neg_dist = self.calculate_frechet_distance(an_vs[0], an_vs[1], neg_vs[0][i], neg_vs[1][i])
                        neg += np.exp(-neg_dist / 5.4)
                    except:
                        continue

            if neg == 0:
                return torch.tensor(max(0.01, abs(pos) * 0.1), device=anchor.device, requires_grad=True)
                
            exp_neg = neg / negNum
            
            loss = exp_pos / (exp_neg + self.eps)
            loss = -np.log(loss + self.eps)
            
            if np.isnan(loss) or np.isinf(loss):
                return torch.tensor(0.01, device=anchor.device, requires_grad=True)
            
            return torch.tensor(abs(loss), device=anchor.device, requires_grad=True)
            
        except Exception as e:
            return torch.tensor(0.01, device=anchor.device, requires_grad=True)

    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """Calculate Frechet distance between two Gaussians"""
        try:
            from scipy import linalg
        except ImportError:
            mu1 = np.atleast_1d(mu1)
            mu2 = np.atleast_1d(mu2)
            diff = mu1 - mu2
            return np.dot(diff, diff)
        
        try:
            mu1 = np.atleast_1d(mu1)
            mu2 = np.atleast_1d(mu2)

            sigma1 = np.atleast_2d(sigma1)
            sigma2 = np.atleast_2d(sigma2)

            if mu1.shape != mu2.shape:
                return 1000.0
            if sigma1.shape != sigma2.shape:
                return 1000.0

            diff = mu1 - mu2

            offset = np.eye(sigma1.shape[0]) * eps
            covmean, _ = linalg.sqrtm(np.dot(sigma1+offset, sigma2+offset), disp=False)

            if np.iscomplexobj(covmean):
                covmean = covmean.real

            covmean = np.abs(covmean)
            tr_covmean = np.trace(covmean)

            term1 = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) 
            term2 = - 2 * tr_covmean
            dist = term1 + term2

            if np.isnan(dist) or np.isinf(dist):
                return np.dot(diff, diff)

            return max(0.0, dist)
            
        except Exception as e:
            try:
                mu1 = np.atleast_1d(mu1)
                mu2 = np.atleast_1d(mu2)
                diff = mu1 - mu2
                return np.dot(diff, diff)
            except:
                return 1000.0


class AdaptiveRegionConsistencyLoss(nn.Module):
    """
    Adaptive Region Consistency Loss combining:
    1. Adaptive task-specific losses (Focal, Dice-Focal, Hard Mining)
    2. Region-based consistency loss
    """
    
    def __init__(self, tasks=['seg6', 'seg9'], class_nb_seg6=6, class_nb_seg9=9, 
                 use_adaptive_loss=True, use_region_consistency=True,
                 region_weight=0.5, ignore_index=255):
        super(AdaptiveRegionConsistencyLoss, self).__init__()
        
        self.tasks = tasks
        self.class_nb_seg6 = class_nb_seg6
        self.class_nb_seg9 = class_nb_seg9
        self.use_adaptive_loss = use_adaptive_loss
        self.use_region_consistency = use_region_consistency
        self.region_weight = region_weight
        self.ignore_index = ignore_index
        
        # Initialize adaptive loss selector
        if self.use_adaptive_loss:
            self.adaptive_selector = AdaptiveLossSelector(ignore_index=ignore_index)
        
        # Initialize region consistency loss
        if self.use_region_consistency:
            self.region_consistency = RegionContrastiveLoss(tasks=tasks)
        
        # Fallback standard losses (using original NLL implementation)
        self.standard_loss_seg6 = lambda pred, target: self._original_nll_loss(pred, target)
        self.standard_loss_seg9 = lambda pred, target: self._original_nll_loss(pred, target)
        
        print(f"✅ AdaptiveRegionConsistencyLoss initialized:")
        print(f"   Tasks: {tasks}")
        print(f"   Classes: seg6={class_nb_seg6}, seg9={class_nb_seg9}")
        print(f"   Adaptive Loss: {use_adaptive_loss}")
        print(f"   Region Consistency: {use_region_consistency}")
        print(f"   Region Weight: {region_weight}")
    
    def _original_nll_loss(self, pred, target):
        """Original NLL loss implementation from roof_loss_functions.py"""
        pred_log_softmax = F.log_softmax(pred, dim=1)
        return F.nll_loss(pred_log_softmax, target.long(), ignore_index=-1)
    
    def compute_supervision(self, x_pred_seg6, x_output_seg6, x_pred_seg9, x_output_seg9):
        """
        Compute supervised task-specific loss with adaptive loss selection
        """
        losses = []
        
        # seg6 loss
        if self.use_adaptive_loss:
            batch_quality = self.adaptive_selector.analyze_batch_quality(x_output_seg6, 'seg6')
            loss_fn = self.adaptive_selector.get_loss_strategy('seg6', batch_quality)
            loss_seg6 = loss_fn(x_pred_seg6, x_output_seg6)
        else:
            loss_seg6 = self.standard_loss_seg6(x_pred_seg6, x_output_seg6)
        
        losses.append(loss_seg6)
        
        # seg9 loss
        if self.use_adaptive_loss:
            batch_quality = self.adaptive_selector.analyze_batch_quality(x_output_seg9, 'seg9')
            loss_fn = self.adaptive_selector.get_loss_strategy('seg9', batch_quality)
            loss_seg9 = loss_fn(x_pred_seg9, x_output_seg9)
        else:
            loss_seg9 = self.standard_loss_seg9(x_pred_seg9, x_output_seg9)
        
        losses.append(loss_seg9)
        
        return losses
    
    def compute_region_consistency(self, pred_seg6, pred_seg9, feat, masks=None):
        """
        Compute region consistency loss between seg6 and seg9 predictions
        """
        if not self.use_region_consistency or masks is None:
            return torch.tensor(0.0, device=pred_seg6.device, requires_grad=True)
        
        try:
            # Convert predictions to probability maps
            prob_seg6 = F.softmax(pred_seg6, dim=1)
            prob_seg9 = F.softmax(pred_seg9, dim=1)
            
            # Apply region consistency loss
            consistency_loss = self.region_consistency(prob_seg6, prob_seg9, masks)
            
            return consistency_loss * self.region_weight
            
        except Exception as e:
            print(f"Warning: Region consistency computation failed: {e}")
            return torch.tensor(0.0, device=pred_seg6.device, requires_grad=True)
    
    def forward(self, outputs, targets, feat=None, masks=None):
        """
        Forward pass combining adaptive supervision and region consistency
        
        Args:
            outputs: Dict containing 'seg6' and 'seg9' predictions
            targets: Dict containing 'seg6' and 'seg9' ground truth
            feat: Feature maps for region consistency (optional)
            masks: SAM masks for region consistency (optional)
        """
        total_loss = 0.0
        loss_dict = {}
        
        # Extract predictions and targets
        pred_seg6 = outputs.get('seg6')
        pred_seg9 = outputs.get('seg9')
        target_seg6 = targets.get('seg6')
        target_seg9 = targets.get('seg9')
        
        # Compute supervised losses
        if pred_seg6 is not None and target_seg6 is not None and pred_seg9 is not None and target_seg9 is not None:
            supervision_losses = self.compute_supervision(pred_seg6, target_seg6, pred_seg9, target_seg9)
            
            loss_dict['seg6'] = supervision_losses[0]
            loss_dict['seg9'] = supervision_losses[1]
            
            total_loss += supervision_losses[0] + supervision_losses[1]
        
        # Compute region consistency loss
        if pred_seg6 is not None and pred_seg9 is not None:
            region_loss = self.compute_region_consistency(pred_seg6, pred_seg9, feat, masks)
            loss_dict['region_consistency'] = region_loss
            total_loss += region_loss
        
        loss_dict['total'] = total_loss
        
        return loss_dict
    
    def get_loss_info(self):
        """Get information about current loss strategies"""
        info = {
            'adaptive_loss': self.use_adaptive_loss,
            'region_consistency': self.use_region_consistency,
            'region_weight': self.region_weight
        }
        
        if self.use_adaptive_loss and hasattr(self, 'adaptive_selector'):
            info['strategies'] = {}
            for task, strategy in self.adaptive_selector.loss_strategies.items():
                info['strategies'][task] = strategy['type']
        
        return info


# Compatibility functions for existing codebase
class ComputeRoofLoss(nn.Module):
    """Wrapper for backward compatibility"""
    
    def __init__(self, use_adaptive=False, use_region_consistency=False):
        super(ComputeRoofLoss, self).__init__()
        
        if use_adaptive or use_region_consistency:
            self.loss_fn = AdaptiveRegionConsistencyLoss(
                tasks=['seg6', 'seg9'],
                use_adaptive_loss=use_adaptive,
                use_region_consistency=use_region_consistency
            )
            self.use_new_loss = True
        else:
            # Original implementation
            self.comp_edge_loss = BalancedCrossEntropyLoss()
            self.use_new_loss = False
        
    def compute_supervision(self, x_pred_seg6, x_output_seg6, x_pred_seg9, x_output_seg9):
        """Compute supervised task-specific loss"""
        if self.use_new_loss:
            outputs = {'seg6': x_pred_seg6, 'seg9': x_pred_seg9}
            targets = {'seg6': x_output_seg6, 'seg9': x_output_seg9}
            loss_dict = self.loss_fn(outputs, targets)
            return [loss_dict['seg6'], loss_dict['seg9']]
        else:
            # Original implementation - exactly matching roof_loss_functions.py
            x_pred_seg6_log = F.log_softmax(x_pred_seg6, dim=1) 
            loss_seg6 = F.nll_loss(x_pred_seg6_log, x_output_seg6.long(), ignore_index=-1)
            
            x_pred_seg9_log = F.log_softmax(x_pred_seg9, dim=1)
            loss_seg9 = F.nll_loss(x_pred_seg9_log, x_output_seg9.long(), ignore_index=-1)
            
            return [loss_seg6, loss_seg9]


# Legacy functions for compatibility
class BalancedCrossEntropyLoss(Module):
    """Balanced Cross Entropy Loss with optional ignore regions"""
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