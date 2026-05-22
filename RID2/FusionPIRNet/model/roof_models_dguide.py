# 主文件（最小化修改 - 固定权重融合GSE信息）
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.seg_hrnet import hrnet_w18, HighResolutionHead
from model.最终代码精简版.dual_branch_continual_learning import (
    enhance_dual_branch_model_with_continual_learning,
    get_update_buffer
)


# ==================== 创新点1: Hierarchical Cross-Modal Feature Integration ====================
# 1.1 Multi-Aspect Visual Semantic Enhancement (MASE)

class RGBStyleEvolution(nn.Module):
    """
    Multi-Aspect Visual Semantic Extraction (MASE)
    多视角视觉语义提取 - 包含4个语义编码器
    """
    def __init__(self, feature_channels=18, num_styles=8, style_dim=32):
        super().__init__()
        
        self.num_styles = num_styles
        self.style_dim = style_dim
        
        # 4个语义编码器
        self.semantic_encoder = nn.ModuleDict({
            'texture': nn.Sequential(
                nn.Conv2d(feature_channels, style_dim//4, 3, padding=1),
                nn.BatchNorm2d(style_dim//4),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            ),
            'color': nn.Sequential(
                nn.Conv2d(feature_channels, style_dim//4, 5, padding=2),
                nn.BatchNorm2d(style_dim//4),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(2),
                nn.Flatten(),
                nn.Linear(style_dim//4 * 4, style_dim//4)
            ),
            'appearance': nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feature_channels, style_dim//4)
            ),
            'semantic': nn.Sequential(
                nn.Conv2d(feature_channels, style_dim//4, 1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
        })
        
        self.semantic_fusion = nn.Sequential(
            nn.Linear(style_dim, style_dim),
            nn.ReLU(),
            nn.Linear(style_dim, style_dim),
            nn.LayerNorm(style_dim)
        )
        
        self.visual_modulator = nn.Sequential(
            nn.Linear(style_dim + 6, feature_channels),
            nn.Tanh()
        )
        
    def forward(self, rgb_features, update_prototypes=True):
        B, C, H, W = rgb_features.shape
        
        semantics = [encoder(rgb_features) for encoder in self.semantic_encoder.values()]
        combined_semantic = torch.cat(semantics, dim=1)
        refined_semantic = self.semantic_fusion(combined_semantic)
        
        context_info = self._generate_visual_context(refined_semantic)
        modulation_input = torch.cat([refined_semantic, context_info], dim=1)
        modulation = self.visual_modulator(modulation_input)
        
        modulation_spatial = modulation.view(B, C, 1, 1)
        modulated_features = rgb_features * (1.0 + 0.15 * modulation_spatial)
        
        return modulated_features, {'visual_features': refined_semantic}
    
    def _generate_visual_context(self, semantic_features):
        B = semantic_features.size(0)
        semantic_norm = torch.norm(semantic_features, dim=1, keepdim=True)
        semantic_mean = torch.mean(semantic_features, dim=1, keepdim=True)
        
        context_features = [semantic_norm, semantic_mean]
        while len(context_features) < 6:
            context_features.append(torch.zeros_like(semantic_norm))
        
        return torch.cat(context_features[:6], dim=1)


# ==================== 1.2 Hierarchical Geometric-Spatial Reasoning (HGSR) ====================

# Level 1: GSE - Geometric Structure Encoder
class DepthGeometricEncoder(nn.Module):
    """
    Geometric Structure Encoder (GSE)
    几何结构编码器 - 提取边界、坡度等几何特征
    """
    def __init__(self, feature_channels=18):
        super().__init__()
        
        # Multi-Scale Geometric Extractor (MSGE)
        self.geometric_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, feature_channels // 4, 3, padding=1),
                nn.BatchNorm2d(feature_channels // 4),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(1, feature_channels // 4, 5, padding=2),
                nn.BatchNorm2d(feature_channels // 4),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(1, feature_channels // 4, 7, padding=3),
                nn.BatchNorm2d(feature_channels // 4),
                nn.ReLU()
            )
        ])
        
        # Gradient-Boundary Enhancer (GBE)
        self.gradient_processor = nn.Sequential(
            nn.Conv2d(3, feature_channels // 4, 3, padding=1),
            nn.BatchNorm2d(feature_channels // 4),
            nn.ReLU()
        )
        
        self.geometric_fusion = nn.Sequential(
            nn.Conv2d(16, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU()
        )
        
        self.boundary_enhancer = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels // 2, 3, padding=1),
            nn.BatchNorm2d(feature_channels // 2),
            nn.ReLU(),
            nn.Conv2d(feature_channels // 2, feature_channels // 2, 3, padding=2, dilation=2),
            nn.BatchNorm2d(feature_channels // 2),
            nn.ReLU(),
            nn.Conv2d(feature_channels // 2, 1, 1),
            nn.Sigmoid()
        )
        
    def forward(self, depth_map):
        grad_x = torch.abs(depth_map[:, :, :, 1:] - depth_map[:, :, :, :-1])
        grad_y = torch.abs(depth_map[:, :, 1:, :] - depth_map[:, :, :-1, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        geometric_features = [encoder(depth_map) for encoder in self.geometric_encoders]
        
        gradient_input = torch.cat([grad_x, grad_y, grad_magnitude], dim=1)
        gradient_features = self.gradient_processor(gradient_input)
        
        all_features = torch.cat(geometric_features + [gradient_features], dim=1)
        fused_features = self.geometric_fusion(all_features)
        
        boundary_attention = self.boundary_enhancer(fused_features)
        enhanced_features = fused_features * (1 + boundary_attention)
        
        return enhanced_features


# Level 2: SRR - Spatial Relation Reasoner
class DepthSpatialReasoner(nn.Module):
    """
    Spatial Relation Reasoner (SRR)
    空间关系推理器 - 推理屋顶构件间的空间关系
    """
    def __init__(self, feature_channels=18):
        super().__init__()
        
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU()
        )
        
        # Multi-Scale Spatial Reasoning
        self.multi_scale_reasoning = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(feature_channels, feature_channels // 4, 1),
                nn.ReLU(),
                nn.Upsample(scale_factor=1, mode='bilinear', align_corners=True)
            ) for size in [(1, 1), (2, 2), (4, 4)]
        ])
        
        self.reasoning_fusion = nn.Sequential(
            nn.Conv2d(30, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 1),
            nn.BatchNorm2d(feature_channels),
            nn.Sigmoid()
        )
        
    def forward(self, depth_features):
        B, C, H, W = depth_features.shape
        
        spatial_features = self.spatial_conv(depth_features)
        
        scale_features = []
        for reasoning_module in self.multi_scale_reasoning:
            scale_feat = reasoning_module[0](depth_features)
            scale_feat = reasoning_module[1](scale_feat)
            scale_feat = F.interpolate(scale_feat, size=(H, W), mode='bilinear', align_corners=True)
            scale_features.append(scale_feat)
        
        all_features = torch.cat([spatial_features] + scale_features, dim=1)
        reasoning_weights = self.reasoning_fusion(all_features)
        
        reasoned_features = depth_features * reasoning_weights
        
        return reasoned_features


# Level 3: HHAP - Hierarchical Height-Aware Prompting
class LocalHeightAttention(nn.Module):
    """Local Height Attention (LHA) - HHAP的子模块"""
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.kernel_size = kernel_size
        
        self.multi_scale_conv = nn.ModuleList([
            nn.Conv2d(1, channels // 8, 3, padding=1),
            nn.Conv2d(1, channels // 8, 5, padding=2),
            nn.Conv2d(1, channels // 8, 7, padding=3),
        ])
        
        self.gradient_conv = nn.Sequential(
            nn.Conv2d(1, channels // 8, 3, padding=1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU()
        )
        
        self.edge_enhance = nn.Sequential(
            nn.Conv2d(1, channels // 8, 3, padding=1),
            nn.BatchNorm2d(channels // 8),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels // 8, 3, padding=2, dilation=2),
            nn.BatchNorm2d(channels // 8),
            nn.Sigmoid()
        )
        
        self.refine_base = nn.Sequential(
            nn.Conv2d(channels // 2, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels//4),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        self.refine_edge = nn.Sequential(
            nn.Conv2d(channels // 8, channels // 4, 1),
            nn.BatchNorm2d(channels // 4),
            nn.ReLU()
        )
        
        self.final_fusion = nn.Sequential(
            nn.Conv2d(channels + channels // 4, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
        self.attention_gen = nn.Sequential(
            nn.Conv2d(channels, channels // 4, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, height_map, enhance_edges=True):
        grad_x = torch.abs(height_map[:, :, :, 1:] - height_map[:, :, :, :-1])
        grad_y = torch.abs(height_map[:, :, 1:, :] - height_map[:, :, :-1, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        height_gradient = torch.max(grad_x, grad_y)
        
        multi_scale_features = [conv(height_map) for conv in self.multi_scale_conv]
        gradient_features = self.gradient_conv(height_gradient)
        
        base_features = torch.cat(multi_scale_features + [gradient_features], dim=1)
        refined_base = self.refine_base(base_features)
        
        if enhance_edges:
            edge_features = self.edge_enhance(height_gradient)
            refined_edge = self.refine_edge(edge_features)
            combined_features = torch.cat([refined_base, refined_edge], dim=1)
            final_features = self.final_fusion(combined_features)
        else:
            final_features = refined_base
        
        attention_weights = self.attention_gen(final_features)
        final_features = final_features * attention_weights
        
        return final_features


class UnifiedHeightEncoder(nn.Module):
    """
    Hierarchical Height Encoder (HHE)
    层次化高度编码器 - HHAP的核心组件
    """
    def __init__(self, input_channels=1, prompt_channels=16):
        super().__init__()
        self.prompt_channels = prompt_channels
        
        # Shared Convolution
        self.shared_conv = nn.Sequential(
            nn.Conv2d(input_channels, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Unified Branch
        self.unified_branch = nn.Sequential(
            nn.Conv2d(64, prompt_channels, 5, 1, 2),
            nn.BatchNorm2d(prompt_channels),
            nn.ReLU(),
            nn.Conv2d(prompt_channels, prompt_channels, 3, 1, 1),
            nn.BatchNorm2d(prompt_channels),
            nn.ReLU(),
            nn.Conv2d(prompt_channels, prompt_channels, 3, 1, 1),
            nn.BatchNorm2d(prompt_channels),
            nn.ReLU(),
            nn.Conv2d(prompt_channels, prompt_channels, 3, 1, 2, dilation=2),
            nn.BatchNorm2d(prompt_channels),
            nn.ReLU(),
            nn.Conv2d(prompt_channels, prompt_channels, 1, 1, 0),
            nn.BatchNorm2d(prompt_channels),
            nn.ReLU()
        )
        
        # Local Height Attention (LHA)
        self.local_height_attention = LocalHeightAttention(prompt_channels, kernel_size=5)
        
        # Fine Detail Detector (FDD)
        self.fine_detail_detector = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, prompt_channels // 2, 1),
            nn.Sigmoid()
        )
        
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(prompt_channels + prompt_channels // 2, prompt_channels, 3, padding=1),
            nn.BatchNorm2d(prompt_channels),
            nn.ReLU(),
            nn.Conv2d(prompt_channels, prompt_channels, 1),
            nn.BatchNorm2d(prompt_channels)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_fc = nn.Sequential(
            nn.Linear(prompt_channels, prompt_channels // 4),
            nn.ReLU(),
            nn.Linear(prompt_channels // 4, prompt_channels),
            nn.Sigmoid()
        )
        
    def forward(self, height_map):
        shared_features = self.shared_conv(height_map)
        unified_features = self.unified_branch(shared_features)
        local_height_features = self.local_height_attention(height_map, enhance_edges=True)
        
        grad_x = torch.abs(height_map[:, :, :, 1:] - height_map[:, :, :, :-1])
        grad_y = torch.abs(height_map[:, :, 1:, :] - height_map[:, :, :-1, :])
        grad_x = F.pad(grad_x, (0, 1, 0, 0), mode='replicate')
        grad_y = F.pad(grad_y, (0, 0, 0, 1), mode='replicate')
        height_gradient = torch.sqrt(grad_x**2 + grad_y**2 + 1e-6)
        
        fine_details = self.fine_detail_detector(height_gradient)
        
        all_features = torch.cat([unified_features, fine_details], dim=1)
        unified_features = self.feature_fusion(all_features)
        
        fusion_weight = torch.sigmoid(torch.mean(unified_features, dim=(2,3), keepdim=True))
        unified_features = unified_features + fusion_weight * local_height_features
        
        global_context = self.global_pool(unified_features)
        global_context = global_context.view(global_context.size(0), -1)
        global_prompt = self.global_fc(global_context)
        
        return unified_features, global_prompt


class UnifiedPrompt(nn.Module):
    """
    Height-Aware Prompt Generator (HAPG)
    高度感知提示生成器
    """
    def __init__(self, feature_channels, prompt_channels=64):
        super().__init__()
        self.feature_channels = feature_channels
        self.prompt_channels = prompt_channels
        
        # Channel Prompt
        self.channel_prompt = nn.Sequential(
            nn.Conv2d(prompt_channels, feature_channels, 1),
            nn.Sigmoid()
        )
        
        # Spatial Prompt
        self.spatial_prompt = nn.Sequential(
            nn.Conv2d(prompt_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # Point-wise Prompt
        self.point_wise_prompt = nn.Sequential(
            nn.Conv2d(prompt_channels, feature_channels, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 1),
            nn.Sigmoid()
        )
        
        self.residual_enhance = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 1),
            nn.Sigmoid()
        )
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_channels + prompt_channels, feature_channels * 2, 3, 1, 1),
            nn.BatchNorm2d(feature_channels * 2),
            nn.ReLU(),
            nn.Conv2d(feature_channels * 2, feature_channels, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU()
        )
        
    def forward(self, features, height_prompt, global_prompt=None):
        B, C, H, W = features.shape
        
        if isinstance(height_prompt, (list, tuple)):
            height_prompt = height_prompt[0] if len(height_prompt) > 0 else torch.zeros(B, self.prompt_channels, H, W, device=features.device, dtype=features.dtype)
        
        if hasattr(height_prompt, 'size') and (height_prompt.size(2) != H or height_prompt.size(3) != W):
            height_prompt = F.interpolate(height_prompt, size=(H, W), mode='bilinear', align_corners=True)
        
        channel_attention = self.channel_prompt(height_prompt)
        spatial_attention = self.spatial_prompt(height_prompt)
        point_attention = self.point_wise_prompt(height_prompt)
        
        combined_attention = channel_attention * spatial_attention * point_attention
        features_enhanced = features * combined_attention
        
        residual_weight = self.residual_enhance(features_enhanced)
        features_enhanced = features_enhanced + residual_weight * features
        
        if global_prompt is not None:
            global_weight = global_prompt.unsqueeze(-1).unsqueeze(-1)
            if global_weight.size(1) != C:
                global_avg = torch.mean(global_weight, dim=1, keepdim=True)
                global_weight = global_avg.expand(-1, C, -1, -1)
            features_enhanced = features_enhanced * (1 + 0.1 * global_weight)
        
        concat_features = torch.cat([features_enhanced, height_prompt], dim=1)
        output_features = self.fusion_conv(concat_features)
        
        return output_features


class UnifiedMultiScaleHeightPrompt(nn.Module):
    """
    Multi-Scale Prompt Coordinator (MSPC)
    多尺度提示协调器 - 在4个特征分支上应用高度感知提示
    """
    def __init__(self, branch_channels=[18, 36, 72, 144], prompt_channels=64):
        super().__init__()
        self.num_branches = len(branch_channels)
        self.branch_channels = branch_channels
        
        self.height_encoder = UnifiedHeightEncoder(
            input_channels=1, 
            prompt_channels=prompt_channels
        )
        
        self.unified_prompts = nn.ModuleList([
            UnifiedPrompt(branch_channels[i], prompt_channels)
            for i in range(self.num_branches)
        ])
        
        self.unified_weight = nn.Parameter(torch.tensor(0.6))
        
        self.feature_recalibration = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(branch_channels[i], branch_channels[i] // 4, 1),
                nn.ReLU(),
                nn.Conv2d(branch_channels[i] // 4, branch_channels[i], 1),
                nn.Sigmoid()
            ) for i in range(self.num_branches)
        ])
        
        self.self_guided_attention = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(branch_channels[i], branch_channels[i] // 2, 3, padding=1),
                nn.BatchNorm2d(branch_channels[i] // 2),
                nn.ReLU(),
                nn.Conv2d(branch_channels[i] // 2, 1, 1),
                nn.Sigmoid()
            ) for i in range(self.num_branches)
        ])
        
    def forward(self, multi_scale_features, height_map):
        height_features, global_prompt = self.height_encoder(height_map)
        enhanced_features = []
        
        for branch_idx, features in enumerate(multi_scale_features):
            enhanced_feat = self.unified_prompts[branch_idx](
                features, height_features, global_prompt
            )
            
            recal_weight = self.feature_recalibration[branch_idx](enhanced_feat)
            enhanced_feat = enhanced_feat * recal_weight
            
            spatial_attention = self.self_guided_attention[branch_idx](enhanced_feat)
            enhanced_feat = enhanced_feat * (1 + 0.3 * spatial_attention)
            
            weight = torch.sigmoid(self.unified_weight)
            final_feat = weight * enhanced_feat + (1 - weight) * features
            enhanced_features.append(final_feat)
        
        return enhanced_features


# ==================== 1.3 Adaptive Weight Cross-Modal Fusion (AWCMF) ====================

class CrossModalInnovativeFusion(nn.Module):
    """
    Adaptive Weight Cross-Modal Fusion (AWCMF)
    自适应权重跨模态融合
    """
    def __init__(self, feature_channels=18):
        super().__init__()
        
        # Modal Alignment Transformer (MAT)
        self.modal_alignment = nn.ModuleDict({
            'rgb': nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU()
            ),
            'depth': nn.Sequential(
                nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
                nn.BatchNorm2d(feature_channels),
                nn.ReLU()
            )
        })
        
        self.cross_fusion = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU(),
            nn.Conv2d(feature_channels, feature_channels, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU()
        )
        
        # Dynamic Weight Generator (DWG)
        self.adaptive_weight_gen = nn.Sequential(
            nn.Linear(feature_channels * 4, feature_channels * 2),
            nn.ReLU(),
            nn.Linear(feature_channels * 2, 3),
            nn.Softmax(dim=1)
        )
        
        # Feature Enhancement Processor (FEP)
        self.feature_enhancer = nn.Sequential(
            nn.Conv2d(feature_channels * 2, feature_channels * 2, 3, padding=1),
            nn.BatchNorm2d(feature_channels * 2),
            nn.ReLU(),
            nn.Conv2d(feature_channels * 2, feature_channels, 1),
            nn.BatchNorm2d(feature_channels),
            nn.ReLU()
        )
        
        self.cross_residual = nn.Sequential(
            nn.Conv2d(feature_channels, feature_channels, 3, padding=1),
            nn.BatchNorm2d(feature_channels),
            nn.Sigmoid()
        )
        
    def forward(self, rgb_features, depth_features):
        B, C, H, W = rgb_features.shape
        
        aligned_rgb = self.modal_alignment['rgb'](rgb_features)
        aligned_depth = self.modal_alignment['depth'](depth_features)
        
        rgb_global = F.adaptive_avg_pool2d(aligned_rgb, (1, 1)).squeeze(-1).squeeze(-1)
        depth_global = F.adaptive_avg_pool2d(aligned_depth, (1, 1)).squeeze(-1).squeeze(-1)
        
        cross_features = torch.cat([aligned_rgb, aligned_depth], dim=1)
        fused_cross = self.cross_fusion(cross_features)
        
        combined_global = torch.cat([rgb_global, depth_global, rgb_global - depth_global, rgb_global + depth_global], dim=1)
        fusion_weights = self.adaptive_weight_gen(combined_global)
        
        w_rgb, w_depth, w_fusion = fusion_weights.chunk(3, dim=1)
        
        weighted_rgb = w_rgb.view(B, 1, 1, 1) * aligned_rgb
        weighted_depth = w_depth.view(B, 1, 1, 1) * aligned_depth
        
        concat_features = torch.cat([weighted_rgb, weighted_depth], dim=1)
        enhanced_features = self.feature_enhancer(concat_features)
        
        residual_weight = self.cross_residual(enhanced_features)
        final_features = enhanced_features + residual_weight * (aligned_rgb + aligned_depth) / 2
        
        final_features = w_fusion.view(B, 1, 1, 1) * final_features
        
        return final_features, {
            'fusion_weights': fusion_weights,
            'cross_modal_similarity': F.cosine_similarity(rgb_global, depth_global, dim=1).mean().item()
        }


# ==================== 双分支架构 ====================

class DualBranchInnovativeRGBD(nn.Module):
    def __init__(self, num_classes=9, backbone_name='hrnet_w18', enable_chfi=True):
        super().__init__()
        
        self.num_classes = num_classes
        self.enable_chfi = enable_chfi
        
        # RGB分支
        self.rgb_backbone = hrnet_w18(pretrained=False)
        
        # 1.1 MASE
        if self.enable_chfi:
            self.rgb_style_evolution = RGBStyleEvolution(feature_channels=18)
        
        # Depth分支
        self.depth_backbone = hrnet_w18(pretrained=False)
        self.depth_backbone.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False)
        # 1.2 HGSR
        backbone_channels = [18, 36, 72, 144]
        if self.enable_chfi:
            self.depth_geometric_encoder = DepthGeometricEncoder(feature_channels=18)
            self.depth_spatial_reasoner = DepthSpatialReasoner(feature_channels=18)
            
            self.unified_height_prompt = UnifiedMultiScaleHeightPrompt(
                branch_channels=backbone_channels, 
                prompt_channels=64
            )
        
        # 1.3 AWCMF
        if self.enable_chfi:
            self.cross_modal_fusion = nn.ModuleList([
                CrossModalInnovativeFusion(feature_channels=ch) for ch in backbone_channels
            ])
        else:
            self.cross_modal_fusion = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(ch * 2, ch, 3, padding=1),
                    nn.BatchNorm2d(ch),
                    nn.ReLU()
                ) for ch in backbone_channels
            ])
        
        self.segmentation_head = HighResolutionHead(backbone_channels, num_outputs=num_classes)
        
        self._fusion_stats = {}
        self._cl_enabled = False
        
    def forward(self, x, masks=None, task_id=None, mode='train'):
        out_size = x.size()[2:]
        
        rgb = x[:, :3, :, :]
        depth = x[:, 3:4, :, :]
        
        # RGB分支
        rgb_features = self.rgb_backbone(rgb)
        
        # 1.1 MASE
        if self.enable_chfi:
            rgb_enhanced_0, rgb_style_info = self.rgb_style_evolution(rgb_features[0])
            rgb_features = [rgb_enhanced_0] + rgb_features[1:]
        else:
            rgb_style_info = {}
        
        # Depth分支
        depth_features = self.depth_backbone(depth)
        
        # 1.2 HGSR - 三路独立处理后融合（修复版）
        if self.enable_chfi:
            # 路径1: GSE - 从原始depth提取几何边界特征
            depth_geometric_0 = self.depth_geometric_encoder(depth)
            
            # 路径2: SRR - 处理backbone第0层，推理空间关系
            depth_reasoned_0 = self.depth_spatial_reasoner(depth_features[0])
            
            # 路径3: HHAP - 处理所有层，高度提示增强
            depth_prompted_features = self.unified_height_prompt(depth_features, depth)
            
            # 尺寸对齐
            target_size = depth_reasoned_0.shape[2:]
            if depth_geometric_0.shape[2:] != target_size:
                depth_geometric_0 = F.interpolate(
                    depth_geometric_0, 
                    size=target_size, 
                    mode='bilinear', 
                    align_corners=True
                )
            
            # 融合三路信息
            fused_depth_0 = (
                0.3 * depth_geometric_0 + 
                0.4 * depth_reasoned_0 + 
                0.3 * depth_prompted_features[0]
            )
            
            depth_features = [fused_depth_0] + depth_prompted_features[1:]
        
        # 1.3 AWCMF: 跨模态融合
        fused_features = []
        fusion_stats = []
        
        for i, (rgb_feat, depth_feat) in enumerate(zip(rgb_features, depth_features)):
            if rgb_feat.shape[2:] != depth_feat.shape[2:]:
                depth_feat = F.interpolate(depth_feat, size=rgb_feat.shape[2:], 
                                         mode='bilinear', align_corners=True)
            
            if self.enable_chfi:
                fused_feat, fusion_info = self.cross_modal_fusion[i](rgb_feat, depth_feat)
                fusion_stats.append(fusion_info)
            else:
                concat_feat = torch.cat([rgb_feat, depth_feat], dim=1)
                fused_feat = self.cross_modal_fusion[i](concat_feat)
                fusion_stats.append({})
            
            fused_features.append(fused_feat)
        
        # 2. CFO: Continual Fusion Optimization
        enhanced_feat_0 = fused_features[0]
        cl_info = {}
        
        if hasattr(self, 'continual_enhancer') and self._cl_enabled:
            enhanced_feat_0, cl_info = self.continual_enhancer(
                fused_features[0], None, task_id, enable_update=(mode == 'train')
            )
        
        fused_features[0] = enhanced_feat_0
        guided_features = fused_features
        
        # 分割头
        segmentation_pred = self.segmentation_head(guided_features)
        final_output = F.interpolate(segmentation_pred, out_size, mode='bilinear', align_corners=True)
        
        # 多尺度特征拼接
        x0_h, x0_w = guided_features[0].size(2), guided_features[0].size(3)
        x1 = F.interpolate(guided_features[1], (x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(guided_features[2], (x0_h, x0_w), mode='bilinear', align_corners=True)
        x3 = F.interpolate(guided_features[3], (x0_h, x0_w), mode='bilinear', align_corners=True)
        final_feat = torch.cat([guided_features[0], x1, x2, x3], 1)
        
        self._fusion_stats = {
            'rgb_style': rgb_style_info,
            'fusion_stats': fusion_stats,
            'continual_learning': cl_info
        }
        
        return final_output, final_feat


class ContinualLearningRoofSegmentationMTL(nn.Module):
    def __init__(self, base_model_class=DualBranchInnovativeRGBD, 
                 enable_continual_learning=True, **base_model_kwargs):
        super().__init__()
        
        self.base_model = base_model_class(**base_model_kwargs)
        self.enable_continual_learning = enable_continual_learning
        self._cl_info = {}
        
        if enable_continual_learning:
            self.base_model = enhance_dual_branch_model_with_continual_learning(
                self.base_model,
                rgb_feature_channels=18,
                enable_fusion_pattern=True,
                enable_meta=False
            )
    
    def forward(self, x, height_map=None, mode='train'):
        if height_map is not None:
            rgbd_input = torch.cat([x, height_map], dim=1)
        else:
            zero_depth = torch.zeros(x.size(0), 1, x.size(2), x.size(3), 
                                   device=x.device, dtype=x.dtype)
            rgbd_input = torch.cat([x, zero_depth], dim=1)
        
        predictions, features = self.base_model(rgbd_input, mode=mode)
        
        if hasattr(self.base_model, '_fusion_stats'):
            self._cl_info = self.base_model._fusion_stats
        
        return predictions, features
    
    def compute_continual_learning_loss(self, outputs=None):
        if not self.enable_continual_learning:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        if hasattr(self.base_model, 'compute_continual_learning_loss'):
            return self.base_model.compute_continual_learning_loss()
        else:
            return torch.tensor(0.0, device=next(self.parameters()).device)
    
    def set_continual_learning_mode(self, enable=True):
        self.enable_continual_learning = enable
        if hasattr(self.base_model, 'set_continual_learning_mode'):
            self.base_model.set_continual_learning_mode(enable)
    
    def get_memory_stats(self):
        if hasattr(self.base_model, '_fusion_stats'):
            stats = self.base_model._fusion_stats
            if hasattr(self.base_model, 'get_continual_stats'):
                cl_stats = self.base_model.get_continual_stats()
                stats.update(cl_stats)
            return stats
        return {}


def get_roof_MTL(num_classes=9, backbone_name='hrnet_w18',
                enable_continual_learning=True, enable_chfi=True):
    model = ContinualLearningRoofSegmentationMTL(
        base_model_class=DualBranchInnovativeRGBD,
        enable_continual_learning=enable_continual_learning,
        enable_chfi=enable_chfi,
        num_classes=num_classes,
        backbone_name=backbone_name
    )
    return model


def apply_delayed_updates_after_backward():
    get_update_buffer().apply_updates()


def clear_pending_updates():
    get_update_buffer().clear()


__all__ = [
    'get_roof_MTL',
    'apply_delayed_updates_after_backward',
    'clear_pending_updates',
]