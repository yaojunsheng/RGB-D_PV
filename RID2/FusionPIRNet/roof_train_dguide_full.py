import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import argparse
import shutil
from dataset.roof_dataset_extended import *
from model.roof_map_cons import RoofMapCons
from utils.evaluation import ConfMatrix
from progress.bar import Bar as Bar
from utils import Logger, AverageMeter, mkdir_p
from torch.autograd import Variable
import copy
from model.roof_models_dguide import get_roof_MTL
from losses.roof_loss_functions import ComputeRoofLoss
from evaluation.roof_metrics import ComputeRoofMetric
from torch.utils.tensorboard import SummaryWriter
import random
import time
import datetime
import threading
from collections import OrderedDict
import logging

# 缓存数据集
class CachedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, cache_size_gb=4):
        self.dataset = original_dataset
        self.cache = OrderedDict()
        self.max_cache_size = cache_size_gb * 1024 * 1024 * 1024
        self.current_size = 0
        self.lock = threading.Lock()
        
    def _estimate_size(self, data):
        size = 0
        for item in data:
            if hasattr(item, 'nbytes'):
                size += item.nbytes
            elif isinstance(item, torch.Tensor):
                size += item.numel() * item.element_size()
        return size
    
    def __getitem__(self, idx):
        with self.lock:
            if idx in self.cache:
                item = self.cache.pop(idx)
                self.cache[idx] = item
                return item
        
        data = self.dataset[idx]
        
        with self.lock:
            data_size = self._estimate_size(data)
            if data_size < self.max_cache_size * 0.1:
                while self.current_size + data_size > self.max_cache_size and self.cache:
                    oldest_key, oldest_data = self.cache.popitem(last=False)
                    self.current_size -= self._estimate_size(oldest_data)
                
                self.cache[idx] = data
                self.current_size += data_size
        
        return data
    
    def __len__(self):
        return len(self.dataset)

parser = argparse.ArgumentParser(description='Ablation Study - Single-Task Roof Segmentation')

# ============ 消融实验核心参数 ============
parser.add_argument('--ablation-mode', default='M3', type=str, 
                    choices=['M0', 'M1', 'M2', 'M3'],
                    help='Ablation mode: M0=Baseline, M1=+CHFI, M2=+CFO, M3=Full(CHFI+CFO)')

# 基础训练参数
parser.add_argument('--train_bs', default=8, type=int)
parser.add_argument('--val_bs', default=8, type=int)
parser.add_argument('--dataroot', default='./data', type=str)
parser.add_argument('--out', default='./results/ablation_study', help='output directory')
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--seed', default=0, type=int)

# 任务参数
parser.add_argument('--num-classes', default=6, type=int)
parser.add_argument('--task-type', default='seg6', choices=['seg6', 'seg9'])
parser.add_argument('--use-height-prompt', default=True, type=bool)
parser.add_argument('--height-norm-method', default='roof_aware', type=str)

# CFO参数（M2和M3需要）
parser.add_argument('--cfo-loss-weight', default=0.2, type=float,
                    help='CFO (Continual Fusion Optimization) loss weight')

# GPU和数据加载
parser.add_argument('--use-multi-gpu', action='store_true', default=False)
parser.add_argument('--gpu-ids', default='0,1', type=str)
parser.add_argument('--single-gpu-id', default='0', type=str)
parser.add_argument('--cache-size-gb', default=6, type=int)
parser.add_argument('--val-cache-size-gb', default=3, type=int)
parser.add_argument('--num-workers', default=4, type=int)

opt = parser.parse_args()

# GPU设置
if opt.use_multi_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_ids
    if opt.train_bs == 8:
        opt.train_bs = 16
        opt.val_bs = 16
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.single_gpu_id

# 任务类型设置
if opt.task_type == 'seg6':
    opt.num_classes = 6
elif opt.task_type == 'seg9':
    opt.num_classes = 6

def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_torch(opt.seed)

# 创建输出目录
current_time = datetime.datetime.now()
time_str = current_time.strftime("%Y-%m-%d-%H-%M-%S")

if not os.path.isdir(opt.out):
    mkdir_p(opt.out)

# 输出路径包含消融模式
paths = [f'{opt.task_type}_height', opt.ablation_mode, time_str]
for i in range(len(paths)):
    opt.out = os.path.join(opt.out, paths[i])
    if not os.path.isdir(opt.out):
        mkdir_p(opt.out)

# Setup logging
log_file = os.path.join(opt.out, 'training.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("消融实验 - 单任务屋顶分割")
logger.info("="*80)
logger.info(f"消融模式: {opt.ablation_mode}")
logger.info(f"实验目录: {opt.out}")
logger.info("")

# 详细输出当前消融配置
logger.info("消融配置:")
if opt.ablation_mode == 'M0':
    logger.info("  M0: Baseline (基础双分支)")
    logger.info("    组件:")
    logger.info("      - 基础RGB分支")
    logger.info("      - 基础Depth分支")
    logger.info("      - 简单跨模态融合")
    logger.info("    ✗ 创新点1 (CHFI): 关闭")
    logger.info("    ✗ 创新点2 (CFO): 关闭")
    
elif opt.ablation_mode == 'M1':
    logger.info("  M1: Baseline + 创新点1 (CHFI)")
    logger.info("    ✓ 创新点1 (CHFI): 开启")
    logger.info("      - 1.1 MASE: Multi-Aspect Visual Semantic Enhancement")
    logger.info("      - 1.2 HGSR: Hierarchical Geometric-Spatial Reasoning")
    logger.info("           * Level 1: GSE (Geometric Structure Encoder)")
    logger.info("           * Level 2: SRR (Spatial Relation Reasoner)")
    logger.info("           * Level 3: HHAP (Hierarchical Height-Aware Prompting)")
    logger.info("      - 1.3 AWCMF: Adaptive Weight Cross-Modal Fusion")
    logger.info("    ✗ 创新点2 (CFO): 关闭")
    
elif opt.ablation_mode == 'M2':
    logger.info("  M2: Baseline + 创新点2 (CFO)")
    logger.info("    ✗ 创新点1 (CHFI): 关闭")
    logger.info("    ✓ 创新点2 (CFO): 开启")
    logger.info("      - 2.1 MPFPA: Multi-Prototype Fusion Pattern Adaptation")
    logger.info("           * FPRE: Fusion Pattern Representation Extractor")
    logger.info("           * APL: Adaptive Prototype Learning")
    logger.info("           * TPB: Temporal Pattern Bank")
    logger.info("      - 2.2 FAPR: Fusion-Aware Progressive Refinement")
    logger.info("           * 2.2.1 FQA: Fusion Quality Assessor")
    logger.info("           * 2.2.2 PIR: Progressive Iterative Refinement (6步)")
    
elif opt.ablation_mode == 'M3':
    logger.info("  M3: Full Model (CHFI + CFO)")
    logger.info("    ✓ 创新点1 (CHFI): 开启")
    logger.info("    ✓ 创新点2 (CFO): 开启")
    logger.info("    所有组件激活")

logger.info("")
logger.info(f"任务: {opt.task_type} ({opt.num_classes} 类别)")
logger.info(f"高度提示: {opt.use_height_prompt}")
logger.info(f"训练批次大小: {opt.train_bs}")
logger.info(f"验证批次大小: {opt.val_bs}")
logger.info(f"随机种子: {opt.seed}")
logger.info("="*80)

# 模型初始化 - 根据消融模式正确控制创新点
print(f"\n初始化模型，消融模式: {opt.ablation_mode}")

# 消融模式配置映射
ablation_config = {
    'M0': {'enable_chfi': False, 'enable_continual_learning': False},
    'M1': {'enable_chfi': True,  'enable_continual_learning': False},
    'M2': {'enable_chfi': False, 'enable_continual_learning': True},
    'M3': {'enable_chfi': True,  'enable_continual_learning': True},
}

config = ablation_config[opt.ablation_mode]

logger.info(f"创新点配置:")
logger.info(f"  创新点1 (CHFI): {'✓ 开启' if config['enable_chfi'] else '✗ 关闭'}")
logger.info(f"  创新点2 (CFO): {'✓ 开启' if config['enable_continual_learning'] else '✗ 关闭'}")

# 使用统一接口创建模型
model = get_roof_MTL(
    num_classes=opt.num_classes,
    backbone_name='hrnet_w18',
    enable_continual_learning=config['enable_continual_learning'],  # CFO开关
    enable_chfi=config['enable_chfi']  # CHFI开关
)

enable_continual = config['enable_continual_learning']

model = model.cuda()

if opt.use_multi_gpu and torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)
    logger.info(f"使用 {torch.cuda.device_count()} 个GPU")

# 优化器
params = list(model.parameters())
optimizer = optim.Adam(params, lr=5e-4)#seg6用7e-5
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, eta_min=1e-6)

# Resume模型
start_epoch = 0
if opt.resume:
    checkpoint = torch.load(opt.resume)
    if hasattr(model, 'module'):
        model.module.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    start_epoch = checkpoint['epoch']
    optimizer.load_state_dict(checkpoint['optimizer'])

# 计算参数空间
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

model_params = count_parameters(model)
logger.info(f'模型参数量: {model_params/1e6:.2f}M')

# 损失和度量
comp_loss = ComputeRoofLoss()
comp_metric = ComputeRoofMetric(class_nb_seg6=6, class_nb_seg9=6)

# 数据集
logger.info("创建数据集...")

if opt.use_height_prompt:
    roof_train_set_original = RoofCropExtended(
        root=opt.dataroot, 
        train=True, 
        augmentation=True, 
        aug_twice=False,
        sam_edge=False,
        height_norm_method=opt.height_norm_method
    )
    roof_test_set_original = RoofExtended(
        root=opt.dataroot, 
        train=False, 
        height_norm_method=opt.height_norm_method
    )
else:
    from dataset.roof import *
    roof_train_set_original = Roof_crop(
        root=opt.dataroot, 
        train=True, 
        augmentation=True, 
        aug_twice=False,
        sam_edge=False
    )
    roof_test_set_original = Roof(root=opt.dataroot, train=False)

roof_train_set = CachedDataset(roof_train_set_original, cache_size_gb=opt.cache_size_gb)
roof_test_set = CachedDataset(roof_test_set_original, cache_size_gb=opt.val_cache_size_gb)

# DataLoader
num_workers_train = min(opt.num_workers, os.cpu_count())
num_workers_val = max(1, opt.num_workers // 2)

roof_train_loader = torch.utils.data.DataLoader(
    dataset=roof_train_set, 
    batch_size=opt.train_bs, 
    shuffle=True,
    num_workers=num_workers_train, 
    pin_memory=True, 
    persistent_workers=True, 
    prefetch_factor=2, 
    drop_last=True
)

roof_test_loader = torch.utils.data.DataLoader(
    dataset=roof_test_set, 
    batch_size=opt.val_bs, 
    shuffle=False,
    num_workers=num_workers_val, 
    pin_memory=True,
    persistent_workers=True, 
    prefetch_factor=2
)

train_batch = len(roof_train_loader)
test_batch = len(roof_test_loader)

logger.info(f"训练数据集: {len(roof_train_set)} 样本")
logger.info(f"测试数据集: {len(roof_test_set)} 样本")
logger.info(f"训练批次: {train_batch}")
logger.info(f"测试批次: {test_batch}")

# 前向传播函数
def model_forward(model, rgb_data, height_data=None, mode='train'):
    """适配模型的前向传播"""
    if enable_continual:
        if hasattr(model, 'module'):
            return model.module.forward(rgb_data, height_data, mode=mode)
        else:
            return model.forward(rgb_data, height_data, mode=mode)
    else:
        if opt.use_height_prompt and height_data is not None:
            return model(rgb_data, height_data, mode=mode)
        else:
            batch_size, _, height, width = rgb_data.shape
            zero_depth = torch.zeros(batch_size, 1, height, width, 
                                   device=rgb_data.device, dtype=rgb_data.dtype)
            return model(rgb_data, zero_depth, mode=mode)

def compute_cfo_loss(model, outputs=None):
    """计算CFO损失（仅M2和M3）"""
    if not enable_continual:
        return torch.tensor(0.0).cuda()
    
    if hasattr(model, 'module') and hasattr(model.module, 'compute_continual_learning_loss'):
        return model.module.compute_continual_learning_loss(outputs)
    elif hasattr(model, 'compute_continual_learning_loss'):
        return model.compute_continual_learning_loss(outputs)
    else:
        return torch.tensor(0.0).cuda()

def save_best_model(state, is_best):
    if is_best:
        filename = f'roof_{opt.task_type}_{opt.ablation_mode}_best.pth.tar'
        filepath = os.path.join(opt.out, filename)
        torch.save(state, filepath)
        logger.info(f'*** 保存新的最佳模型: {filename} ***')
        return filepath
    return None

# 训练参数
total_epoch = 300
avg_cost = np.zeros([total_epoch, 6], dtype=np.float32)
best_performance = -100
best_epoch = 0
step_num = 0

logger.info(f"开始训练，总计 {total_epoch} 轮...")
logger.info("="*80)

# TensorBoard
tb_writer = SummaryWriter(log_dir=os.path.join(opt.out, 'tb_logs'))

# 主训练循环
for epoch in range(start_epoch, total_epoch):
    epoch_start_time = time.time()
    
    lr_main = optimizer.param_groups[0]["lr"]
    logger.info(f'轮次 {epoch:03d}/{total_epoch:03d} | 学习率: {lr_main:.6f} | 模式: {opt.ablation_mode}')
    
    # 训练
    model.train()
    
    cost_main = AverageMeter()
    cost_cfo = AverageMeter() if enable_continual else None
    
    roof_train_dataset = iter(roof_train_loader)
    batch_size = opt.train_bs
    
    for k in range(train_batch):
        step_num += 1
        
        # 数据加载
        train_data_tuple = roof_train_dataset.__next__()
        
        if opt.use_height_prompt:
            train_data, train_label_seg6, train_label_seg9, train_height = train_data_tuple[:4]
            train_data, train_height = train_data.cuda(), train_height.cuda()
            
            if opt.task_type == 'seg6':
                train_label = train_label_seg6.long().cuda()
            else:
                train_label = train_label_seg9.long().cuda()
        else:
            train_data, train_label_seg6, train_label_seg9 = train_data_tuple[:3]
            train_data = train_data.cuda()
            train_height = None
            
            if opt.task_type == 'seg6':
                train_label = train_label_seg6.long().cuda()
            else:
                train_label = train_label_seg9.long().cuda()

        # 前向传播
        outputs = model_forward(model, train_data, train_height, mode='train')
        
        if isinstance(outputs, tuple):
            train_pred, _ = outputs
        else:
            train_pred = outputs
        
        # 损失计算
        train_loss = F.cross_entropy(train_pred, train_label)
        total_loss = train_loss
        
        # CFO损失（仅M2和M3）
        total_cfo_loss = 0
        if enable_continual:
            cfo_loss_val = compute_cfo_loss(model, outputs if 'outputs' in locals() else None)
            total_cfo_loss = cfo_loss_val.item() if isinstance(cfo_loss_val, torch.Tensor) else 0
            total_loss += opt.cfo_loss_weight * cfo_loss_val
        
        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 更新损失记录
        cost_main.update(train_loss.item(), batch_size)
        if cost_cfo is not None:
            cost_cfo.update(total_cfo_loss, batch_size)
        
        # 计算指标
        cost = np.zeros(3)
        cost[0] = train_loss.item()
        
        if opt.task_type == 'seg6':
            cost[1] = comp_metric.compute_miou_seg6(train_pred, train_label).item()
            cost[2] = comp_metric.compute_accuracy_seg6(train_pred, train_label).item()
        else:
            cost[1] = comp_metric.compute_miou_seg9(train_pred, train_label).item()
            cost[2] = comp_metric.compute_accuracy_seg9(train_pred, train_label).item()
        
        avg_cost[epoch, :3] += cost[:3] / train_batch
        
        # TensorBoard logging
        tb_writer.add_scalar(f'Train/{opt.task_type}_Loss', train_loss.item(), step_num)
        if enable_continual:
            tb_writer.add_scalar('Train/CFO_Loss', total_cfo_loss, step_num)
        tb_writer.add_scalar('Train/Learning_Rate', optimizer.param_groups[0]['lr'], step_num)
        
        # 打印训练进度
        if (k + 1) % 10 == 0:
            log_msg = f'  批次 {k+1:03d}/{train_batch:03d} | Loss: {cost_main.avg:.4f}'
            if enable_continual:
                log_msg += f' | CFO: {cost_cfo.avg:.4f}'
            print(log_msg)
        
        torch.cuda.empty_cache()

    # 评估
    model.eval()
    conf_mat = ConfMatrix(opt.num_classes)
    
    val_loss_acc = 0.0
    
    with torch.no_grad():
        roof_test_dataset = iter(roof_test_loader)
        for k in range(test_batch):
            if opt.use_height_prompt:
                test_data, test_label_seg6, test_label_seg9, test_height = roof_test_dataset.__next__()
                test_data, test_height = test_data.cuda(), test_height.cuda()
                
                if opt.task_type == 'seg6':
                    test_label = test_label_seg6.long().cuda()
                else:
                    test_label = test_label_seg9.long().cuda()
            else:
                test_data, test_label_seg6, test_label_seg9 = roof_test_dataset.__next__()
                test_data = test_data.cuda()
                test_height = None
                
                if opt.task_type == 'seg6':
                    test_label = test_label_seg6.long().cuda()
                else:
                    test_label = test_label_seg9.long().cuda()

            outputs = model_forward(model, test_data, test_height, mode='eval')
            
            if isinstance(outputs, tuple):
                test_pred, _ = outputs
            else:
                test_pred = outputs
            
            test_loss = F.cross_entropy(test_pred, test_label)
            val_loss_acc += test_loss.item()
            
            conf_mat.update(test_pred.argmax(1).flatten(), test_label.flatten())
        
        avg_cost[epoch, 3] = val_loss_acc / test_batch
        
        metrics = conf_mat.get_metrics()
        miou = metrics[0]
        pix_acc = metrics[1]
        
        avg_cost[epoch, 4:6] = miou, pix_acc
    
    # 性能计算
    current_miou = miou * 100
    single_task_performance = current_miou
    
    isbest = single_task_performance > best_performance
    if isbest:
        best_performance = single_task_performance
        best_epoch = epoch

    # TensorBoard logging
    tb_writer.add_scalar(f'Val/{opt.task_type}_Loss', avg_cost[epoch, 3], epoch)
    tb_writer.add_scalar(f'Val/{opt.task_type}_mIoU', current_miou, epoch)
    
    epoch_time = time.time() - epoch_start_time
    
    # 记录训练结果
    log_msg = f'轮次 {epoch:03d} [{opt.ablation_mode}] | 时间: {epoch_time:.1f}s | 训练: {avg_cost[epoch, 1]*100:.2f}% | 验证: {current_miou:.2f}% | 最佳: {best_performance:.2f}%'
    print(log_msg)
    logger.info(log_msg)
    
    if isbest:
        logger.info(f"    *** 新的最佳模型 ({opt.ablation_mode}) 在轮次 {epoch} ***")

    # 学习率调度
    scheduler.step()

    # 保存最佳模型
    save_state = {
        'epoch': epoch + 1,
        'state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'best_performance': best_performance,
        'best_epoch': best_epoch,
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'avg_cost': avg_cost,
        'ablation_mode': opt.ablation_mode,
        'args': opt,
    }
    
    save_best_model(save_state, isbest)

tb_writer.close()

# 训练完成总结
logger.info("="*80)
logger.info(f"消融实验完成 - 模式: {opt.ablation_mode}")
logger.info("="*80)

config_desc = {
    'M0': '仅Baseline',
    'M1': 'Baseline + 创新点1 (CHFI)',
    'M2': 'Baseline + 创新点2 (CFO)',
    'M3': '完整模型 (CHFI + CFO)'
}
logger.info(f"配置: {config_desc[opt.ablation_mode]}")

logger.info("")
logger.info(f"最佳 {opt.task_type} mIoU: {best_performance:.4f}% 在轮次 {best_epoch}")
logger.info(f"最佳轮次 {best_epoch} 的最终结果:")
logger.info(f"  训练 mIoU: {avg_cost[best_epoch, 1]*100:.2f}%")
logger.info(f"  验证 mIoU: {avg_cost[best_epoch, 4]*100:.2f}%")
logger.info("="*80)

print(f"\n[{opt.ablation_mode}] 训练完成!")
print(f"最佳性能: {best_performance:.4f}% 在轮次 {best_epoch}")
print(f"完整训练日志保存至: {log_file}")