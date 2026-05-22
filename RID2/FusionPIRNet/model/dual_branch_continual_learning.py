# dual_branch_continual_learning.py（简化版 - 去除FQA和TPB）
import torch
import torch.nn as nn
import torch.nn.functional as F
import threading
from collections import deque


class DelayedUpdateBuffer:
    def __init__(self):
        self.pending_updates = []
        self.lock = threading.Lock()
    
    def schedule_update(self, buffer_ref, new_value, update_type='replace'):
        with self.lock:
            self.pending_updates.append({
                'buffer': buffer_ref,
                'value': new_value.detach().clone() if isinstance(new_value, torch.Tensor) else new_value,
                'type': update_type
            })
    
    def apply_updates(self):
        with self.lock:
            with torch.no_grad():
                for update in self.pending_updates:
                    buffer = update['buffer']
                    value = update['value']
                    update_type = update['type']
                    
                    if update_type == 'replace':
                        buffer.data.copy_(value)
                    elif update_type == 'momentum':
                        buffer.data.mul_(0.9).add_(value, alpha=0.1)
                    elif update_type == 'accumulate':
                        buffer.data.add_(value)
            self.pending_updates.clear()
    
    def clear(self):
        with self.lock:
            self.pending_updates.clear()


_global_update_buffer = DelayedUpdateBuffer()

def get_update_buffer():
    return _global_update_buffer


# ==================== 创新点2: CFO持续融合优化 ====================

# 2.1 MPFPA: Multi-Prototype Fusion Pattern Adaptation
class FusedFeaturePrototypeLearning(nn.Module):
    """
    Multi-Prototype Fusion Pattern Adaptation (MPFPA)
    多原型融合模式适应 - 学习融合模式原型
    """
    def __init__(self, feature_channels=18, num_prototypes=8, pattern_dim=32):
        super().__init__()
        
        self.feature_channels = feature_channels
        self.num_prototypes = num_prototypes
        self.pattern_dim = pattern_dim
        
        # Fusion Pattern Representation Extractor (FPRE)
        self.pattern_extractor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_channels, pattern_dim),
            nn.ReLU(),
            nn.Linear(pattern_dim, pattern_dim),
            nn.LayerNorm(pattern_dim)
        )
        
        # Adaptive Prototype Learning (APL)
        self.fusion_prototypes = nn.Parameter(torch.randn(num_prototypes, pattern_dim) * 0.1)
        
        # Pattern Modulator
        self.pattern_modulator = nn.Sequential(
            nn.Linear(pattern_dim + 6, feature_channels),
            nn.Tanh()
        )
        
    def forward(self, fused_features, update_prototypes=True):
        B, C, H, W = fused_features.shape
        
        if C != self.feature_channels:
            if C > self.feature_channels:
                fused_features = fused_features[:, :self.feature_channels, :, :]
            else:
                padding = torch.zeros(B, self.feature_channels - C, H, W, 
                                    device=fused_features.device, dtype=fused_features.dtype)
                fused_features = torch.cat([fused_features, padding], dim=1)
        
        pattern_representation = self.pattern_extractor(fused_features)
        similarity = self._pattern_matching(pattern_representation)
        context_info = self._generate_pattern_context(similarity)
        
        modulation_input = torch.cat([pattern_representation, context_info], dim=1)
        modulation = self.pattern_modulator(modulation_input)
        
        modulation_spatial = modulation.view(B, C, 1, 1)
        modulated_features = fused_features * (1.0 + 0.15 * modulation_spatial)
        
        if self.training and update_prototypes:
            self._schedule_prototype_update(pattern_representation.detach(), similarity.detach())
        
        return modulated_features, {
            'pattern_weights': similarity,
            'fusion_representation': pattern_representation
        }
    
    def _pattern_matching(self, pattern_features):
        pattern_norm = F.normalize(pattern_features, dim=1)
        proto_norm = F.normalize(self.fusion_prototypes.detach(), dim=1)
        raw_similarity = torch.matmul(pattern_norm, proto_norm.T)
        return F.softmax(raw_similarity * 2.5, dim=1)
    
    def _generate_pattern_context(self, similarity):
        context_features = []
        similarity_detached = similarity.detach()
        
        entropy = -torch.sum(similarity_detached * torch.log(similarity_detached + 1e-8), dim=1, keepdim=True)
        context_features.append(entropy)
        
        max_activation = similarity_detached.max(dim=1, keepdim=True)[0]
        context_features.append(max_activation)
        
        context_features.append(torch.zeros_like(entropy))
        context_features.append(torch.zeros_like(entropy))
        context_features.append(entropy * 0.5)
        context_features.append(max_activation * 0.3)
        
        return torch.cat(context_features[:6], dim=1)
    
    def _schedule_prototype_update(self, pattern_features, similarity):
        update_buffer = get_update_buffer()
        
        for i in range(self.num_prototypes):
            weights = similarity[:, i:i+1]
            if weights.sum() > 0.1:
                weighted_avg = (weights * pattern_features).sum(0) / weights.sum()
                current_proto = self.fusion_prototypes[i].detach()
                new_proto = 0.9 * current_proto + 0.1 * weighted_avg
                
                update_buffer.schedule_update(
                    self.fusion_prototypes.data[i:i+1], 
                    new_proto.unsqueeze(0), 
                    'replace'
                )


# 2.2 PIR: Progressive Iterative Refinement
class ProgressiveIterativeRefinement(nn.Module):
    """
    Progressive Iterative Refinement (PIR)
    渐进式迭代精炼 - 6步流程：自引导 → 迭代×3 → 细粒度 → 融合
    """
    def __init__(self, feature_channels=18):
        super().__init__()
        
        # Self-Refinement Network (SRN)
        self.self_refinement_network = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 1),
            nn.Sigmoid()
        )
        
        # Iterative Refinement Network (IRN) - 3次迭代
        self.iterative_refinement = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU(),
                nn.Conv2d(feature_channels, feature_channels, 1),
                nn.Sigmoid()
            ) for _ in range(3)
        ])
        
        # Fine-Grained Detection Network (FGDN)
        self.fine_grained_detector = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 2, 3, padding=1),
            nn.BatchNorm2d(feature_channels // 2),
            nn.ReLU(),
            nn.Conv2d(feature_channels // 2, feature_channels // 2, 3, padding=2, dilation=2),
            nn.BatchNorm2d(feature_channels // 2),
            nn.ReLU(),
            nn.Conv2d(feature_channels // 2, feature_channels, 1),
            nn.Sigmoid()
        )
        
        # Knowledge Fusion Module (KFM)
        self.knowledge_fusion = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU()
        )
        
    def forward(self, features):
        # Step 1: Self-refinement
        self_refined = self.self_refinement_network(features)
        enhanced_features = features * self_refined
        
        # Step 2-4: Iterative refinement
        current_features = enhanced_features
        for refine_module in self.iterative_refinement:
            refine_weight = refine_module(current_features)
            current_features = current_features + refine_weight * enhanced_features
        
        # Step 5: Fine-grained detection
        fine_grained_weight = self.fine_grained_detector(current_features)
        fine_enhanced = current_features * fine_grained_weight
        
        # Step 6: Knowledge fusion
        combined_features = torch.cat([current_features, fine_enhanced], dim=1)
        final_features = self.knowledge_fusion(combined_features)
        
        return final_features, {
            'self_refined': self_refined,
            'fine_grained_weight': fine_grained_weight
        }


class DualBranchContinualEnhancer(nn.Module):
    """
    双分支持续学习增强器
    包含：2.1 MPFPA + 2.2 PIR（去除FQA）
    """
    def __init__(self, feature_channels=18, enable_fusion_pattern=True):
        super().__init__()
        
        self.feature_channels = feature_channels
        self.enable_fusion_pattern = enable_fusion_pattern
        
        # 2.1 MPFPA
        if enable_fusion_pattern:
            self.fusion_pattern_learning = FusedFeaturePrototypeLearning(feature_channels)
        
        # 2.2 PIR
        self.progressive_refinement = ProgressiveIterativeRefinement(feature_channels)
        
        # 自适应融合权重
        fusion_multiplier = 2 if enable_fusion_pattern else 1
        
        self.adaptive_fusion = nn.Sequential(
            nn.Linear(feature_channels, feature_channels),
            nn.ReLU(),
            nn.Linear(feature_channels, fusion_multiplier),
            nn.Softmax(dim=1)
        )
        
        self.cross_modal_aligner = nn.Sequential(
            nn.Linear(feature_channels, feature_channels//2),
            nn.ReLU(),
            nn.Linear(feature_channels//2, feature_channels),
            nn.Tanh()
        )
        
        self.dimension_checker = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
    def forward(self, fused_features, depth_features=None, task_id=None, enable_update=True):
        B, C, H, W = fused_features.shape
        enhanced_features_list = [fused_features]
        contexts = []
        info_dict = {}
        
        # 维度检查
        if C != self.feature_channels:
            if C > self.feature_channels:
                fused_features = fused_features[:, :self.feature_channels, :, :]
            else:
                padding = torch.zeros(B, self.feature_channels - C, H, W, 
                                    device=fused_features.device, dtype=fused_features.dtype)
                fused_features = torch.cat([fused_features, padding], dim=1)
            enhanced_features_list[0] = fused_features
            C = self.feature_channels
        
        # 2.1 MPFPA
        if self.enable_fusion_pattern:
            pattern_features, pattern_info = self.fusion_pattern_learning(
                fused_features, update_prototypes=enable_update
            )
            enhanced_features_list.append(pattern_features)
            
            pattern_context = self.dimension_checker(pattern_features)
            if pattern_context.size(1) != self.feature_channels:
                pattern_context = F.linear(pattern_context, 
                                       torch.eye(self.feature_channels, pattern_context.size(1), 
                                               device=pattern_context.device)[:, :pattern_context.size(1)])
            contexts.append(pattern_context)
            info_dict['pattern_info'] = pattern_info
        
        # 自适应融合
        if len(contexts) > 0:
            base_context = self.dimension_checker(fused_features)
            
            if base_context.size(1) != self.feature_channels:
                base_context = F.linear(base_context, 
                                      torch.eye(self.feature_channels, base_context.size(1), 
                                              device=base_context.device)[:, :base_context.size(1)])
            
            fusion_weights = self.adaptive_fusion(base_context)
            
            final_features = torch.zeros_like(fused_features)
            for i, (features, weight) in enumerate(zip(enhanced_features_list, fusion_weights.t())):
                if i < len(enhanced_features_list):
                    final_features += weight.view(B, 1, 1, 1) * features
            
            alignment_input = self.dimension_checker(final_features)
            if alignment_input.size(1) == self.feature_channels:
                alignment_factor = self.cross_modal_aligner(alignment_input)
                final_features = final_features * (1 + 0.1 * alignment_factor.view(B, C, 1, 1))
            
            info_dict['fusion_weights'] = fusion_weights
        else:
            final_features = fused_features
        
        # 2.2 PIR：渐进式迭代精炼
        refined_features, refinement_info = self.progressive_refinement(final_features)
        info_dict['refinement'] = refinement_info
        
        return refined_features, info_dict
    
    def compute_continual_loss(self, info_dict):
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        
        # MPFPA loss
        if 'pattern_info' in info_dict and 'pattern_weights' in info_dict['pattern_info']:
            pattern_weights = info_dict['pattern_info']['pattern_weights']
            entropy = -torch.sum(pattern_weights * torch.log(pattern_weights + 1e-8), dim=1).mean()
            total_loss += 0.01 * (2.0 - entropy)
        
        # Fusion balance loss
        if 'fusion_weights' in info_dict:
            fusion_weights = info_dict['fusion_weights']
            balance_loss = torch.var(fusion_weights, dim=1).mean()
            total_loss += 0.002 * balance_loss
        
        return total_loss
    
    def get_stats(self):
        stats = {'module_type': 'dual_branch_continual_cfo_simplified'}
        if self.enable_fusion_pattern:
            stats['fusion_pattern_enabled'] = True
        return stats
    
    def apply_delayed_updates(self):
        get_update_buffer().apply_updates()
    
    def clear_pending_updates(self):
        get_update_buffer().clear()


def enhance_dual_branch_model_with_continual_learning(model, 
                                                     rgb_feature_channels=18,
                                                     enable_fusion_pattern=True,
                                                     enable_meta=False):
    """为双分支模型增强持续学习能力（简化版：2.1 MPFPA + 2.2 PIR）"""
    continual_enhancer = DualBranchContinualEnhancer(
        feature_channels=rgb_feature_channels,
        enable_fusion_pattern=enable_fusion_pattern
    )
    
    model.continual_enhancer = continual_enhancer
    model._cl_enabled = True
    
    original_forward = model.forward
    
    def enhanced_forward(x, height_map=None, task_id=None, mode='train'):
        if hasattr(model, 'base_model'):
            results = original_forward(x, height_map, task_id, mode)
        else:
            results = original_forward(x, height_map, task_id, mode)
        
        if hasattr(model, 'continual_enhancer') and model._cl_enabled:
            if isinstance(results, tuple) and len(results) >= 2:
                predictions, features = results[0], results[1]
                
                if isinstance(features, torch.Tensor):
                    if features.size(1) >= rgb_feature_channels:
                        fused_feat = features[:, :rgb_feature_channels, :, :]
                    else:
                        B, C, H, W = features.shape
                        if C < rgb_feature_channels:
                            padding = torch.zeros(B, rgb_feature_channels - C, H, W, 
                                                device=features.device, dtype=features.dtype)
                            fused_feat = torch.cat([features, padding], dim=1)
                        else:
                            fused_feat = features
                elif isinstance(features, list):
                    fused_feat = features[0] if len(features) > 0 else torch.zeros(1, rgb_feature_channels, 1, 1)
                else:
                    fused_feat = features
                
                enhanced_feat, cl_info = model.continual_enhancer(
                    fused_feat, None, task_id, enable_update=(mode == 'train')
                )
                
                if isinstance(features, torch.Tensor) and features.size(1) > rgb_feature_channels:
                    enhanced_features = features.clone()
                    enhanced_features[:, :rgb_feature_channels, :, :] = enhanced_feat
                    final_enhanced_feat = enhanced_features
                else:
                    final_enhanced_feat = enhanced_feat
                
                model._last_cl_info = cl_info
                return predictions, final_enhanced_feat
        
        return results
    
    model.forward = enhanced_forward
    
    def compute_continual_learning_loss(outputs=None):
        if hasattr(model, 'continual_enhancer') and hasattr(model, '_last_cl_info'):
            return model.continual_enhancer.compute_continual_loss(model._last_cl_info)
        return torch.tensor(0.0, device=next(model.parameters()).device)
    
    def get_continual_stats():
        if hasattr(model, 'continual_enhancer'):
            return model.continual_enhancer.get_stats()
        return {}
    
    def set_continual_learning_mode(enable=True):
        model._cl_enabled = enable
    
    def apply_delayed_updates():
        if hasattr(model, 'continual_enhancer'):
            model.continual_enhancer.apply_delayed_updates()
    
    def clear_pending_updates():
        if hasattr(model, 'continual_enhancer'):
            model.continual_enhancer.clear_pending_updates()
    
    model.compute_continual_learning_loss = compute_continual_learning_loss
    model.get_continual_stats = get_continual_stats
    model.set_continual_learning_mode = set_continual_learning_mode
    model.apply_delayed_updates = apply_delayed_updates
    model.clear_pending_updates = clear_pending_updates
    
    return model