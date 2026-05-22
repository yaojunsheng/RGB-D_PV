import os
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from shapely.geometry import shape
from rasterio.features import shapes
import cv2
from matplotlib import pyplot as plt
from definitions import epsg_metric_germany
from electricity_generation import pv_electricity_generation
from masks_to_vector import segment_simplify_and_add_azimuth
from module_placement import module_placement, create_pv_modules_gdf
from spatial_operations import get_image_gdf_in_directory
from scipy.ndimage import gaussian_filter, distance_transform_edt
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# 线程安全的结果存储
results_nosmooth = []
results_rgb = []
results_lock = threading.Lock()  # 用于安全地修改结果列表

# ============================================================================
# 配置区域
# ============================================================================

dir_roof_segment_masks = './数据集/RID2_AsymFormer/gt_test_roof_segment/'
dir_roof_superstructure_masks = './数据集/RID2_AsymFormer/gt_test_superstructure/'
dir_geotifs = './img_tif/'
dir_ndsm = './height_labels/'
dir_pvgis_cache = '/mnt/yjs/tmp/RID2/pvgis_cache'
dir_results = './结果/RID2_AsymFormer/'

# GT数据目录（用于基准评估）
dir_gt_segment_masks = './数据集/RID2_标准数据集/gt_test_roof_segment/'

# 基准评估开关
EVALUATE_BASELINE = False  # True=先评估基准方法，False=直接运行主流程

# 基准评估参数
HUBER_DELTA = 5.0  # Huber损失的delta参数

PIXEL_SIZE = 0.08

COMPARE_WITH_FIXED_30DEG = False  # 是否计算30度
MAX_THREADS = 10  # 最大线程数

visualize = False
bg_is_0 = True
segment_classes = ['flat', 'W', 'S', 'E', 'N']
superstructure_classes = ['unknown', 'pvmodule']
a_min_segments = 2
a_min_superstructures = 0.5
pv_module_peak_power = 400
pv_module_height = 1.7
pv_module_width = 1
default_slope = 40

# ============================================================================


# ============== 地理参考处理函数 ==============

def get_georeference_from_tif(tif_path):
    """从TIF文件读取地理参考信息"""
    with rasterio.open(tif_path) as src:
        return {
            'bounds': src.bounds,
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height
        }


def load_ndsm_with_georeference(ndsm_path, tif_path):
    """加载nDSM高度数据，并从对应的TIF获取地理参考"""
    ndsm_data = cv2.imread(ndsm_path, cv2.IMREAD_UNCHANGED)
    if ndsm_data is None:
        raise ValueError(f"无法读取nDSM文件: {ndsm_path}")
    
    if len(ndsm_data.shape) == 3:
        ndsm_data = ndsm_data[:, :, 0]
    ndsm_data = ndsm_data.astype(float)
    
    georeference = get_georeference_from_tif(tif_path)
    
    return ndsm_data, georeference


def raster_to_vector_with_georeference(raster_mask, georeference, segment_classes, bg_is_0=True):
    """使用提供的地理参考信息将栅格掩码转为矢量"""
    transform = georeference['transform']
    crs = georeference['crs']
    
    geometries = []
    for geom, value in shapes(raster_mask.astype(np.int16), transform=transform):
        value = int(round(value))
        
        if bg_is_0 and value == 0:
            continue
        if value > 0 and value <= len(segment_classes):
            geometries.append({
                'geometry': shape(geom),
                'label': segment_classes[value - 1],
                'value': value
            })
    
    if len(geometries) == 0:
        gdf = gpd.GeoDataFrame(columns=['geometry', 'label', 'value'], crs=crs)
    else:
        gdf = gpd.GeoDataFrame(geometries, crs=crs)
    
    return gdf


# ============== 像素重映射函数 ==============

def remap_segment_mask(mask):
    """将用户的屋顶分割掩码重映射为项目默认格式"""
    remapped = np.zeros_like(mask)
    remapped[mask == 0] = 0
    remapped[mask == 1] = 5  # N
    remapped[mask == 2] = 4  # E
    remapped[mask == 3] = 3  # S
    remapped[mask == 4] = 2  # W
    remapped[mask == 5] = 1  # flat
    return remapped


def remap_superstructure_mask(mask):
    """将用户的屋顶附属结构掩码重映射为二分类格式"""
    remapped = np.zeros_like(mask)
    remapped[mask == 0] = 0
    remapped[mask > 0] = 1
    return remapped


def load_and_remap_mask(file_path, mask_type='segment'):
    """加载掩码并进行像素重映射"""
    mask = cv2.imread(file_path, 0)
    
    if mask_type == 'segment':
        return remap_segment_mask(mask)
    elif mask_type == 'superstructure':
        return remap_superstructure_mask(mask)
    else:
        raise ValueError(f"未知的mask_type: {mask_type}")


def get_vector_labels_remapped(file_path, mask_filename, gdf_images, label_classes, 
                                area_threshold, tif_dir, mask_type='segment', bg_is_0=True):
    """
    加载并重映射掩码，然后转为矢量（修复版）
    """
    mask = load_and_remap_mask(file_path, mask_type=mask_type)
    
    # 修复：正确拼接TIF文件名
    tif_filename = mask_filename + '.tif'
    tif_path = os.path.join(tif_dir, tif_filename)
    
    if not os.path.exists(tif_path):
        return gpd.GeoDataFrame()
    
    try:
        georeference = get_georeference_from_tif(tif_path)
        target_crs_string = f'EPSG:{epsg_metric_germany}'
        georeference['crs'] = target_crs_string
        
        gdf_labels = raster_to_vector_with_georeference(mask, georeference, label_classes, bg_is_0=bg_is_0)
    except Exception as e:
        print(f"    错误：无法处理 {mask_filename}: {e}")
        return gpd.GeoDataFrame()
    
    gdf_labels = gdf_labels[gdf_labels.geometry.area > area_threshold]
    gdf_labels = gdf_labels.reset_index(drop=True)
    
    return gdf_labels


# --------------------------------------------------------------------------- #
# 信息保留度计算
# --------------------------------------------------------------------------- #
def calculate_information_metrics(ndsm_original, ndsm_processed, segment_mask):
    """计算信息保留度指标"""
    roof_mask = segment_mask > 0
    
    if roof_mask.sum() == 0:
        return {
            'variance_retention': 1.0,
            'gradient_retention': 1.0,
            'entropy_retention': 1.0,
            'info_retention_avg': 1.0
        }
    
    original = ndsm_original[roof_mask]
    processed = ndsm_processed[roof_mask]
    
    # 1. 方差保留率
    var_orig = np.var(original)
    var_proc = np.var(processed)
    variance_retention = var_proc / (var_orig + 1e-6)
    
    # 2. 梯度保留率
    grad_y_orig, grad_x_orig = np.gradient(ndsm_original, PIXEL_SIZE)
    grad_y_proc, grad_x_proc = np.gradient(ndsm_processed, PIXEL_SIZE)
    
    grad_mag_orig = np.sqrt(grad_x_orig**2 + grad_y_orig**2)[roof_mask]
    grad_mag_proc = np.sqrt(grad_x_proc**2 + grad_y_proc**2)[roof_mask]
    
    gradient_retention = grad_mag_proc.mean() / (grad_mag_orig.mean() + 1e-6)
    
    # 3. 熵保留率
    hist_orig, _ = np.histogram(original, bins=50, density=True)
    hist_proc, _ = np.histogram(processed, bins=50, density=True)
    
    entropy_orig = -np.sum(hist_orig * np.log(hist_orig + 1e-10))
    entropy_proc = -np.sum(hist_proc * np.log(hist_proc + 1e-10))
    entropy_retention = entropy_proc / (entropy_orig + 1e-6)
    
    info_retention_avg = (variance_retention + gradient_retention + entropy_retention) / 3
    
    return {
        'variance_retention': float(np.clip(variance_retention, 0, 1)),
        'gradient_retention': float(np.clip(gradient_retention, 0, 1)),
        'entropy_retention': float(np.clip(entropy_retention, 0, 1)),
        'info_retention_avg': float(np.clip(info_retention_avg, 0, 1))
    }


# --------------------------------------------------------------------------- #
# 基准评估：无平滑 + 固定30度（带详细输出）
# --------------------------------------------------------------------------- #
def evaluate_baseline_methods(gdf_images, mask_files, max_images=None):
    """
    评估基准方法（添加详细输出）
    """
    if max_images is not None:
        mask_files = mask_files[:max_images]
    
    print(f"\n{'='*70}")
    print("基准评估：无平滑方法 vs 固定30度方法")
    print(f"测试图片数: {len(mask_files)}")
    print(f"{'='*70}\n")
    
    if not os.path.exists(dir_gt_segment_masks):
        print(f"错误：GT数据集路径不存在: {dir_gt_segment_masks}")
        return
    
    gt_files = os.listdir(dir_gt_segment_masks)
    print(f"GT数据集文件数: {len(gt_files)}")
    print(f"预测数据集文件数: {len(mask_files)}\n")
    
    # ========== 第一步：计算全局基准值 ==========
    print(f"\n{'='*70}")
    print("第一步：用无平滑方法计算全局基准值")
    print(f"{'='*70}\n")
    
    huber_values_nosmooth = []
    
    for mask_idx, mask_filename in enumerate(mask_files):
        print(f"\n[步骤1-{mask_idx+1}/{len(mask_files)}] {mask_filename}")
        
        mask_id = mask_filename[:-4]
        pred_mask_path = os.path.join(dir_roof_segment_masks, mask_filename)
        gt_mask_path = os.path.join(dir_gt_segment_masks, mask_filename)
        ndsm_path = os.path.join(dir_ndsm, mask_filename.replace('.png', '.tif'))
        tif_path = os.path.join(dir_geotifs, mask_filename.replace('.png', '.tif'))
        
        if not os.path.exists(ndsm_path) or not os.path.exists(pred_mask_path) or not os.path.exists(gt_mask_path) or not os.path.exists(tif_path):
            print("  → 跳过（文件缺失）")
            continue
        
        print("  → 读取nDSM...", end="", flush=True)
        try:
            ndsm_data, _ = load_ndsm_with_georeference(ndsm_path, tif_path)
            print("完成")
        except Exception as e:
            print(f"失败: {e}")
            continue
        
        print("  → 读取GT掩码...", end="", flush=True)
        gt_segment_mask = cv2.imread(gt_mask_path, 0)
        if gt_segment_mask is None:
            print("失败")
            continue
        gt_segment_mask = remap_segment_mask(gt_segment_mask)
        print("完成")
        
        print("  → 读取Pred掩码...", end="", flush=True)
        pred_segment_mask = cv2.imread(pred_mask_path, 0)
        if pred_segment_mask is None:
            print("失败")
            continue
        pred_segment_mask = remap_segment_mask(pred_segment_mask)
        print("完成")
        
        print("  → 转换GT为矢量...", end="", flush=True)
        gdf_gt = get_vector_labels_remapped(
            gt_mask_path, mask_id, gdf_images, segment_classes, 
            2, dir_geotifs, mask_type='segment', bg_is_0=True
        )
        
        if len(gdf_gt) == 0:
            print("失败（无有效区域）")
            continue
        print(f"完成（{len(gdf_gt)}个区域）")
        
        print("  → 处理GT方位角...", end="", flush=True)
        try:
            gdf_gt, _, _ = segment_simplify_and_add_azimuth(gdf_gt, visualize=False)
            print("完成")
        except Exception as e:
            print(f"失败: {e}")
            continue
        
        print("  → 转换Pred为矢量...", end="", flush=True)
        gdf_pred = get_vector_labels_remapped(
            pred_mask_path, mask_id, gdf_images, segment_classes, 
            2, dir_geotifs, mask_type='segment', bg_is_0=True
        )
        
        if len(gdf_pred) == 0:
            print("失败（无有效区域）")
            continue
        print(f"完成（{len(gdf_pred)}个区域）")
        
        print("  → 处理Pred方位角...", end="", flush=True)
        try:
            gdf_pred, _, _ = segment_simplify_and_add_azimuth(gdf_pred, visualize=False)
            print("完成")
        except Exception as e:
            print(f"失败: {e}")
            continue
        
        print(f"  → GT:{len(gdf_gt)} / Pred:{len(gdf_pred)}")
        
        print("  → 提取GT坡度（无平滑）...", end="", flush=True)
        gt_slopes_nosmooth = extract_slopes_no_smoothing(gdf_gt, ndsm_data, gt_segment_mask)
        print("完成")
        
        print("  → 提取Pred坡度（无平滑）...", end="", flush=True)
        pred_slopes_nosmooth = extract_slopes_no_smoothing(gdf_pred, ndsm_data, pred_segment_mask)
        print("完成")
        
        print("  → 计算Huber...", end="", flush=True)
        gt_mean_nosmooth = np.mean(gt_slopes_nosmooth)
        pred_mean_nosmooth = np.mean(pred_slopes_nosmooth)
        error_nosmooth = pred_mean_nosmooth - gt_mean_nosmooth
        abs_error_nosmooth = abs(error_nosmooth)
        
        if abs_error_nosmooth <= HUBER_DELTA:
            huber_nosmooth = 0.5 * (abs_error_nosmooth ** 2)
        else:
            huber_nosmooth = HUBER_DELTA * (abs_error_nosmooth - 0.5 * HUBER_DELTA)
        
        huber_values_nosmooth.append(huber_nosmooth)
        print(f"完成（Huber={huber_nosmooth:.4f}°）")
    
    # 计算全局基准值
    if len(huber_values_nosmooth) == 0:
        print("\n错误：没有成功处理任何图片")
        return
    
    global_huber_max = max(huber_values_nosmooth)
    
    print(f"\n{'='*70}")
    print(f"✓ 全局基准值计算完成")
    print(f"  global_huber_max = {global_huber_max:.4f}°")
    print(f"{'='*70}\n")
    
    # ========== 第二步：完整评估所有方法 ==========
    print(f"\n{'='*70}")
    print("第二步：完整评估所有方法（使用全局基准值）")
    print(f"{'='*70}\n")
    
    results_nosmooth = []
    results_fixed = []
    
    for mask_idx, mask_filename in enumerate(mask_files):
        print(f"\n[步骤2-{mask_idx+1}/{len(mask_files)}] {mask_filename}")
        
        mask_id = mask_filename[:-4]
        pred_mask_path = os.path.join(dir_roof_segment_masks, mask_filename)
        gt_mask_path = os.path.join(dir_gt_segment_masks, mask_filename)
        ndsm_path = os.path.join(dir_ndsm, mask_filename.replace('.png', '.tif'))
        tif_path = os.path.join(dir_geotifs, mask_filename.replace('.png', '.tif'))
        
        if not os.path.exists(ndsm_path) or not os.path.exists(pred_mask_path) or not os.path.exists(gt_mask_path) or not os.path.exists(tif_path):
            print("  → 跳过（文件缺失）")
            continue
        
        print("  → 读取nDSM...", end="", flush=True)
        try:
            ndsm_data, _ = load_ndsm_with_georeference(ndsm_path, tif_path)
            print("完成")
        except:
            print("失败")
            continue
        
        print("  → 读取GT掩码...", end="", flush=True)
        gt_segment_mask = cv2.imread(gt_mask_path, 0)
        if gt_segment_mask is None:
            print("失败")
            continue
        gt_segment_mask = remap_segment_mask(gt_segment_mask)
        print("完成")
        
        print("  → 读取Pred掩码...", end="", flush=True)
        pred_segment_mask = cv2.imread(pred_mask_path, 0)
        if pred_segment_mask is None:
            print("失败")
            continue
        pred_segment_mask = remap_segment_mask(pred_segment_mask)
        print("完成")
        
        print("  → 转换GT为矢量...", end="", flush=True)
        gdf_gt = get_vector_labels_remapped(
            gt_mask_path, mask_id, gdf_images, segment_classes, 
            2, dir_geotifs, mask_type='segment', bg_is_0=True
        )
        
        if len(gdf_gt) == 0:
            print("失败")
            continue
        print(f"完成（{len(gdf_gt)}个）")
        
        print("  → 处理GT方位角...", end="", flush=True)
        try:
            gdf_gt, _, _ = segment_simplify_and_add_azimuth(gdf_gt, visualize=False)
            print("完成")
        except:
            print("失败")
            continue
        
        print("  → 转换Pred为矢量...", end="", flush=True)
        gdf_pred = get_vector_labels_remapped(
            pred_mask_path, mask_id, gdf_images, segment_classes, 
            2, dir_geotifs, mask_type='segment', bg_is_0=True
        )
        
        if len(gdf_pred) == 0:
            print("失败")
            continue
        print(f"完成（{len(gdf_pred)}个）")
        
        print("  → 处理Pred方位角...", end="", flush=True)
        try:
            gdf_pred, _, _ = segment_simplify_and_add_azimuth(gdf_pred, visualize=False)
            print("完成")
        except:
            print("失败")
            continue
        
        print(f"  → GT:{len(gdf_gt)} / Pred:{len(gdf_pred)}")
        
        # 1. 无平滑方法
        print("  → [无平滑] 提取坡度...", end="", flush=True)
        gt_slopes_nosmooth = extract_slopes_no_smoothing(gdf_gt, ndsm_data, gt_segment_mask)
        ndsm_gt_nosmooth = ndsm_data.copy()
        
        pred_slopes_nosmooth = extract_slopes_no_smoothing(gdf_pred, ndsm_data, pred_segment_mask)
        ndsm_pred_nosmooth = ndsm_data.copy()
        print("完成")
        
        print("  → [无平滑] 计算指标...", end="", flush=True)
        gt_mean_nosmooth = np.mean(gt_slopes_nosmooth)
        pred_mean_nosmooth = np.mean(pred_slopes_nosmooth)
        error_nosmooth = pred_mean_nosmooth - gt_mean_nosmooth
        abs_error_nosmooth = abs(error_nosmooth)
        
        if abs_error_nosmooth <= HUBER_DELTA:
            huber_nosmooth = 0.5 * (abs_error_nosmooth ** 2)
        else:
            huber_nosmooth = HUBER_DELTA * (abs_error_nosmooth - 0.5 * HUBER_DELTA)
        
        info_gt_nosmooth = calculate_information_metrics(ndsm_data, ndsm_gt_nosmooth, gt_segment_mask)
        info_pred_nosmooth = calculate_information_metrics(ndsm_data, ndsm_pred_nosmooth, pred_segment_mask)
        
        info_avg_nosmooth = (info_gt_nosmooth['info_retention_avg'] + info_pred_nosmooth['info_retention_avg']) / 2
        accuracy_nosmooth = 1 - (huber_nosmooth / global_huber_max)
        accuracy_nosmooth = np.clip(accuracy_nosmooth, 0, 1)
        error_norm_nosmooth = 1 - accuracy_nosmooth
        efficiency_nosmooth = info_avg_nosmooth / (error_norm_nosmooth + 0.1)
        print("完成")
        
        results_nosmooth.append({
            'image_id': mask_id,
            'method': 'NoSmoothing',
            'gt_mean_slope': gt_mean_nosmooth,
            'pred_mean_slope': pred_mean_nosmooth,
            'Huber': huber_nosmooth,
            'MAE': abs_error_nosmooth,
            'VarianceRetention': (info_gt_nosmooth['variance_retention'] + info_pred_nosmooth['variance_retention']) / 2,
            'GradientRetention': (info_gt_nosmooth['gradient_retention'] + info_pred_nosmooth['gradient_retention']) / 2,
            'EntropyRetention': (info_gt_nosmooth['entropy_retention'] + info_pred_nosmooth['entropy_retention']) / 2,
            'AvgInfoRetention': info_avg_nosmooth,
            'Efficiency': efficiency_nosmooth
        })
        
        # 2. 固定30度方法
        print("  → [固定30°] 提取坡度...", end="", flush=True)
        gt_slopes_fixed = [default_slope if not np.isnan(az) else 0 for az in gdf_gt["azimuth"]]
        pred_slopes_fixed = [default_slope if not np.isnan(az) else 0 for az in gdf_pred["azimuth"]]
        
        ndsm_gt_fixed = np.full_like(ndsm_data, ndsm_data.mean())
        ndsm_pred_fixed = np.full_like(ndsm_data, ndsm_data.mean())
        print("完成")
        
        print("  → [固定30°] 计算指标...", end="", flush=True)
        gt_mean_fixed = np.mean(gt_slopes_fixed)
        pred_mean_fixed = np.mean(pred_slopes_fixed)
        error_fixed = pred_mean_fixed - gt_mean_fixed
        abs_error_fixed = abs(error_fixed)
        
        if abs_error_fixed <= HUBER_DELTA:
            huber_fixed = 0.5 * (abs_error_fixed ** 2)
        else:
            huber_fixed = HUBER_DELTA * (abs_error_fixed - 0.5 * HUBER_DELTA)
        
        info_gt_fixed = calculate_information_metrics(ndsm_data, ndsm_gt_fixed, gt_segment_mask)
        info_pred_fixed = calculate_information_metrics(ndsm_data, ndsm_pred_fixed, pred_segment_mask)
        
        info_avg_fixed = (info_gt_fixed['info_retention_avg'] + info_pred_fixed['info_retention_avg']) / 2
        accuracy_fixed = 1 - (huber_fixed / global_huber_max)
        accuracy_fixed = np.clip(accuracy_fixed, 0, 1)
        error_norm_fixed = 1 - accuracy_fixed
        efficiency_fixed = info_avg_fixed / (error_norm_fixed + 0.1)
        print("完成")
        
        results_fixed.append({
            'image_id': mask_id,
            'method': 'FixedSlope30',
            'gt_mean_slope': gt_mean_fixed,
            'pred_mean_slope': pred_mean_fixed,
            'Huber': huber_fixed,
            'MAE': abs_error_fixed,
            'VarianceRetention': (info_gt_fixed['variance_retention'] + info_pred_fixed['variance_retention']) / 2,
            'GradientRetention': (info_gt_fixed['gradient_retention'] + info_pred_fixed['gradient_retention']) / 2,
            'EntropyRetention': (info_gt_fixed['entropy_retention'] + info_pred_fixed['entropy_retention']) / 2,
            'AvgInfoRetention': info_avg_fixed,
            'Efficiency': efficiency_fixed
        })
        
        print(f"  ✓ 完成（Huber无平滑={huber_nosmooth:.2f}°, 固定={huber_fixed:.2f}°）")
    
    # 检查是否有有效结果
    if len(results_nosmooth) == 0:
        print("\n错误：第二步评估没有成功处理任何图片")
        return
    
    # 保存结果
    df_nosmooth = pd.DataFrame(results_nosmooth)
    df_fixed = pd.DataFrame(results_fixed)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    baseline_txt = os.path.join(dir_results, f"baseline_evaluation_{timestamp}.txt")
    
    with open(baseline_txt, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("基准方法评估报告\n")
        f.write("="*70 + "\n\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Huber Delta: {HUBER_DELTA}°\n")
        f.write(f"全局基准值: {global_huber_max:.4f}°\n")
        f.write(f"  (来自无平滑方法在 {len(huber_values_nosmooth)} 张图片上的最大Huber误差)\n")
        f.write(f"测试图片数: {len(df_nosmooth)}\n\n")
        
        f.write("="*70 + "\n")
        f.write("1. 无平滑方法\n")
        f.write("="*70 + "\n\n")
        f.write(f"平均Huber: {df_nosmooth['Huber'].mean():.4f}°\n")
        f.write(f"平均MAE: {df_nosmooth['MAE'].mean():.4f}°\n")
        f.write(f"平均方差保留: {df_nosmooth['VarianceRetention'].mean():.1%}\n")
        f.write(f"平均梯度保留: {df_nosmooth['GradientRetention'].mean():.1%}\n")
        f.write(f"平均熵保留: {df_nosmooth['EntropyRetention'].mean():.1%}\n")
        f.write(f"平均综合信息保留: {df_nosmooth['AvgInfoRetention'].mean():.1%}\n")
        f.write(f"平均效率: {df_nosmooth['Efficiency'].mean():.4f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("2. 固定30度方法\n")
        f.write("="*70 + "\n\n")
        f.write(f"平均Huber: {df_fixed['Huber'].mean():.4f}°\n")
        f.write(f"平均MAE: {df_fixed['MAE'].mean():.4f}°\n")
        f.write(f"平均方差保留: {df_fixed['VarianceRetention'].mean():.1%}\n")
        f.write(f"平均梯度保留: {df_fixed['GradientRetention'].mean():.1%}\n")
        f.write(f"平均熵保留: {df_fixed['EntropyRetention'].mean():.1%}\n")
        f.write(f"平均综合信息保留: {df_fixed['AvgInfoRetention'].mean():.1%}\n")
        f.write(f"平均效率: {df_fixed['Efficiency'].mean():.4f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("对比分析\n")
        f.write("="*70 + "\n\n")
        f.write(f"Huber差异: {df_nosmooth['Huber'].mean() - df_fixed['Huber'].mean():+.4f}°\n")
        f.write(f"MAE差异: {df_nosmooth['MAE'].mean() - df_fixed['MAE'].mean():+.4f}°\n")
        f.write(f"信息保留差异: {df_nosmooth['AvgInfoRetention'].mean() - df_fixed['AvgInfoRetention'].mean():+.1%}\n")
        f.write(f"效率差异: {df_nosmooth['Efficiency'].mean() - df_fixed['Efficiency'].mean():+.4f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("关键发现\n")
        f.write("="*70 + "\n\n")
        
        if df_fixed['AvgInfoRetention'].mean() < 0.1:
            f.write("✓ 固定坡度方法信息保留接近0%，证明'零误差陷阱'\n")
        
        if df_nosmooth['AvgInfoRetention'].mean() > df_fixed['AvgInfoRetention'].mean():
            f.write("✓ 无平滑方法保留更多信息\n")
        
        if df_nosmooth['Efficiency'].mean() > df_fixed['Efficiency'].mean():
            f.write("✓ 无平滑方法效率更高\n")
        
        f.write("\n")
        f.write("="*70 + "\n")
    
    print(f"\n✓ 基准评估完成！结果已保存到: {os.path.basename(baseline_txt)}\n")


# ============== 无平滑方法：从nDSM提取真实坡度 ==============

def extract_slopes_no_smoothing(gdf_segments, ndsm_data_or_path, segment_mask_data_or_path):
    """
    无平滑方法：直接从原始nDSM提取坡度，不做任何平滑
    """
    slopes = []
    
    if isinstance(ndsm_data_or_path, np.ndarray):
        ndsm_data = ndsm_data_or_path
    else:
        print(f"    警告：期望numpy数组，但收到路径")
        return [default_slope if not np.isnan(seg.azimuth) else 0 for seg in gdf_segments.itertuples()]
    
    # 无平滑：直接使用原始数据
    ndsm_smoothed = ndsm_data.copy()
    
    # 计算坡度
    grad_y, grad_x = np.gradient(ndsm_smoothed, PIXEL_SIZE)
    slope_angle = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
    
    # 处理分割掩码
    if isinstance(segment_mask_data_or_path, np.ndarray):
        segment_mask = segment_mask_data_or_path
    else:
        print(f"    警告：期望numpy数组")
        return [default_slope if not np.isnan(seg.azimuth) else 0 for seg in gdf_segments.itertuples()]
    
    label_to_id = {'flat': 1, 'W': 2, 'S': 3, 'E': 4, 'N': 5}
    
    for idx, segment in gdf_segments.iterrows():
        try:
            if segment['label'] == 'flat' or np.isnan(segment.get('azimuth', np.nan)):
                slopes.append(0)
                continue
            
            label_id = label_to_id.get(segment['label'], 0)
            mask = (segment_mask == label_id).astype('uint8')
            
            dist_transform = distance_transform_edt(mask)
            weights = (dist_transform / (dist_transform.max() + 1e-6))
            
            segment_slopes = slope_angle[mask == 1]
            segment_weights = weights[mask == 1]
            
            if len(segment_slopes) > 10:
                q1, q3 = np.percentile(segment_slopes, [25, 75])
                iqr = q3 - q1
                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr
                valid_mask = (segment_slopes >= lower_bound) & (segment_slopes <= upper_bound)
                
                if valid_mask.sum() > 5:
                    avg_slope = np.average(segment_slopes[valid_mask], weights=segment_weights[valid_mask])
                else:
                    avg_slope = np.average(segment_slopes, weights=segment_weights)
                
                avg_slope = int(np.clip(np.round(avg_slope), 0, 90))
                slopes.append(avg_slope)
            else:
                slopes.append(default_slope)
                
        except Exception as e:
            slopes.append(default_slope)
    
    return slopes


# ============== 单文件处理函数（用于多线程） ==============

def process_single_file(mask_filename, gdf_images, target_crs_string):
    """处理单个文件的函数，供线程池调用"""
    thread_id = threading.get_ident()
    print(f"\n[线程 {thread_id}] 开始处理: {mask_filename}")
    
    try:
        file_path = os.path.join(dir_roof_segment_masks, mask_filename)
        mask_id = mask_filename[:-4]
        
        gdf_segments = get_vector_labels_remapped(
            file_path, mask_id, gdf_images, segment_classes, a_min_segments, 
            dir_geotifs, mask_type='segment', bg_is_0=bg_is_0
        )
        
        if len(gdf_segments) == 0:
            print(f"[线程 {thread_id}] 警告：{mask_filename} 没有有效的分割区域，跳过")
            return [], [] if COMPARE_WITH_FIXED_30DEG else []
        
        gdf_segments = gdf_segments.set_geometry('geometry')
        gdf_segments = gdf_segments.set_crs(target_crs_string, allow_override=True)
        
        gdf_segments_copy = gpd.GeoDataFrame(
            gdf_segments.drop(columns=['geometry']),
            geometry=gdf_segments.geometry.copy(),
            crs=gdf_segments.crs
        )
        
        try:
            gdf_segments, _, _ = segment_simplify_and_add_azimuth(gdf_segments_copy)
            if gdf_segments.crs is None:
                gdf_segments = gdf_segments.set_geometry('geometry')
                gdf_segments = gdf_segments.set_crs(target_crs_string, allow_override=True)
        except Exception as e:
            print(f"[线程 {thread_id}] 错误：处理分割失败 - {e}")
            return [], [] if COMPARE_WITH_FIXED_30DEG else []
        
        file_path_super = os.path.join(dir_roof_superstructure_masks, mask_filename)
        gdf_superstructures = get_vector_labels_remapped(
            file_path_super, mask_id, gdf_images, superstructure_classes, 
            a_min_superstructures, dir_geotifs, mask_type='superstructure', bg_is_0=bg_is_0
        )
        
        ndsm_file = os.path.join(dir_ndsm, mask_filename.replace('.png', '.tif'))
        segment_mask_file = os.path.join(dir_roof_segment_masks, mask_filename)
        tif_file = os.path.join(dir_geotifs, mask_filename.replace('.png', '.tif'))
        
        # 存储当前文件的结果
        file_results_nosmooth = []
        file_results_rgb = []
        
        # ========== 分支1：无平滑方法 ==========
        print(f"[线程 {thread_id}] [无平滑方法] 计算真实坡度...")
        gdf_nosmooth = gdf_segments.copy()
        
        try:
            ndsm_data, _ = load_ndsm_with_georeference(ndsm_file, tif_file)
            segment_mask = cv2.imread(segment_mask_file, 0)
            segment_mask = remap_segment_mask(segment_mask)
            
            gdf_nosmooth["slopes"] = extract_slopes_no_smoothing(
                gdf_nosmooth, 
                ndsm_data,
                segment_mask
            )
            
            print(f"[线程 {thread_id}]   提取的坡度: {gdf_nosmooth['slopes'].tolist()}")
        except Exception as e:
            print(f"[线程 {thread_id}]   错误：{e}")
            return [], [] if COMPARE_WITH_FIXED_30DEG else []
        
        alignment_nosmooth, gdf_modules_v_nosmooth, gdf_modules_h_nosmooth, azimuth_nosmooth = module_placement(
            gdf_nosmooth,
            gdf_nosmooth["azimuth"],
            gdf_nosmooth["slopes"],
            gdf_superstructures,
            pv_module_height,
            pv_module_width
        )
        
        gdf_modules_nosmooth = create_pv_modules_gdf(alignment_nosmooth, gdf_modules_v_nosmooth, gdf_modules_h_nosmooth, azimuth_nosmooth)
        gdf_modules_nosmooth = gdf_modules_nosmooth.to_crs(epsg_metric_germany)
        
        gdf_nosmooth["pv_modules_per_segment"] = [len(mp.geometry.geoms) for mp in gdf_modules_nosmooth.iloc]
        gdf_nosmooth["pv_peak_power_per_segment"] = [len(mp.geometry.geoms) * pv_module_peak_power / 1000 for mp in gdf_modules_nosmooth.iloc]
        gdf_nosmooth["azimuth_incl_flat"] = azimuth_nosmooth
        
        gs_location = gpd.GeoSeries(gdf_nosmooth.unary_union.centroid)
        gs_location.crs = gdf_images.crs
        electricity_generations_nosmooth = pv_electricity_generation(
            location=gs_location,
            azimuths=gdf_nosmooth["azimuth_incl_flat"],
            slopes=gdf_nosmooth["slopes"],
            peak_powers=gdf_nosmooth["pv_peak_power_per_segment"],
            dir_pvgis_cache=dir_pvgis_cache
        )
        gdf_nosmooth["electricity_generations"] = [np.sum(e) for e in electricity_generations_nosmooth]
        
        for i in range(len(gdf_nosmooth)):
            file_results_nosmooth.append({
                'mask_id': mask_id,
                'segment_id': i,
                'label': gdf_nosmooth.iloc[i]['label'],
                'area_m2': gdf_nosmooth.iloc[i].geometry.area,
                'azimuth': gdf_nosmooth.iloc[i]['azimuth_incl_flat'],
                'slope_deg': gdf_nosmooth.iloc[i]['slopes'],
                'num_modules': gdf_nosmooth.iloc[i]['pv_modules_per_segment'],
                'peak_power_kw': gdf_nosmooth.iloc[i]['pv_peak_power_per_segment'],
                'annual_gen_kwh': gdf_nosmooth.iloc[i]['electricity_generations']
            })
        
        # ========== 分支2：RGB（默认30度）==========
        if COMPARE_WITH_FIXED_30DEG:
            print(f"[线程 {thread_id}] [对比基准] 计算RGB（默认{default_slope}度）...")
            gdf_rgb = gdf_segments.copy()
            gdf_rgb["slopes"] = [default_slope if not np.isnan(az) else 0 for az in gdf_rgb["azimuth"]]
            
            alignment_rgb, gdf_modules_v_rgb, gdf_modules_h_rgb, azimuth_rgb = module_placement(
                gdf_rgb,
                gdf_rgb["azimuth"],
                gdf_rgb["slopes"],
                gdf_superstructures,
                pv_module_height,
                pv_module_width
            )
            
            gdf_modules_rgb = create_pv_modules_gdf(alignment_rgb, gdf_modules_v_rgb, gdf_modules_h_rgb, azimuth_rgb)
            gdf_modules_rgb = gdf_modules_rgb.to_crs(epsg_metric_germany)
            
            gdf_rgb["pv_modules_per_segment"] = [len(mp.geometry.geoms) for mp in gdf_modules_rgb.iloc]
            gdf_rgb["pv_peak_power_per_segment"] = [len(mp.geometry.geoms) * pv_module_peak_power / 1000 for mp in gdf_modules_rgb.iloc]
            gdf_rgb["azimuth_incl_flat"] = azimuth_rgb
            
            electricity_generations_rgb = pv_electricity_generation(
                location=gs_location,
                azimuths=gdf_rgb["azimuth_incl_flat"],
                slopes=gdf_rgb["slopes"],
                peak_powers=gdf_rgb["pv_peak_power_per_segment"],
                dir_pvgis_cache=dir_pvgis_cache
            )
            gdf_rgb["electricity_generations"] = [np.sum(e) for e in electricity_generations_rgb]
            
            for i in range(len(gdf_rgb)):
                file_results_rgb.append({
                    'mask_id': mask_id,
                    'segment_id': i,
                    'label': gdf_rgb.iloc[i]['label'],
                    'area_m2': gdf_rgb.iloc[i].geometry.area,
                    'azimuth': gdf_rgb.iloc[i]['azimuth_incl_flat'],
                    'slope_deg': gdf_rgb.iloc[i]['slopes'],
                    'num_modules': gdf_rgb.iloc[i]['pv_modules_per_segment'],
                    'peak_power_kw': gdf_rgb.iloc[i]['pv_peak_power_per_segment'],
                    'annual_gen_kwh': gdf_rgb.iloc[i]['electricity_generations']
                })
        
        print(f"[线程 {thread_id}] 完成处理: {mask_filename}")
        return file_results_nosmooth, file_results_rgb if COMPARE_WITH_FIXED_30DEG else []
        
    except Exception as e:
        print(f"[线程 {thread_id}] 处理 {mask_filename} 时出错: {str(e)}")
        return [], [] if COMPARE_WITH_FIXED_30DEG else []


# ============== 主程序 ==============

if __name__ == "__main__":
    # 确保输出目录存在
    if not os.path.isdir(dir_pvgis_cache):
        os.makedirs(dir_pvgis_cache)
    if not os.path.isdir(dir_results):
        os.makedirs(dir_results)

    print(f"\n{'='*70}")
    print("无平滑坡度预测 - RID2数据集 (多线程版)")
    print(f"{'='*70}")
    print(f"方法: 无平滑（直接从原始nDSM提取坡度）")
    print(f"  线程数: {MAX_THREADS}")
    print(f"  Pixel Size: {PIXEL_SIZE}m")
    print(f"\n实验模式:")
    if COMPARE_WITH_FIXED_30DEG:
        print(f"  对比模式 - 将与固定{default_slope}度对比")
    else:
        print(f"  单一模式 - 仅运行无平滑方法")
    print(f"{'='*70}\n")

    gdf_images = get_image_gdf_in_directory(dir_geotifs)
    target_crs_string = f'EPSG:{epsg_metric_germany}'

    if gdf_images.crs is None:
        gdf_images = gdf_images.set_crs(epsg=epsg_metric_germany)
    else:
        gdf_images = gdf_images.to_crs(target_crs_string)

    mask_filenames = [f for f in os.listdir(dir_roof_segment_masks) if f.endswith('.png')]
    print(f"发现 {len(mask_filenames)} 个图片文件待处理")

    # --------------------------------------------------------------------------- #
    # 基准评估（可选）
    # --------------------------------------------------------------------------- #
    if EVALUATE_BASELINE:
        evaluate_baseline_methods(gdf_images, mask_filenames, max_images=None)

    # --------------------------------------------------------------------------- #
    # 使用线程池并行处理所有图片
    # --------------------------------------------------------------------------- #
    print("\n" + "="*70)
    print(f"开始并行处理 {len(mask_filenames)} 个文件，使用 {MAX_THREADS} 个线程")
    print("="*70)

    # 创建线程池
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        # 提交所有任务
        futures = {
            executor.submit(
                process_single_file, 
                mask_filename, 
                gdf_images, 
                target_crs_string
            ): mask_filename for mask_filename in mask_filenames
        }

        # 处理完成的任务并收集结果
        for future in as_completed(futures):
            mask_filename = futures[future]
            try:
                file_results_nosmooth, file_results_rgb = future.result()
                
                # 使用锁确保线程安全地更新结果列表
                with results_lock:
                    results_nosmooth.extend(file_results_nosmooth)
                    if COMPARE_WITH_FIXED_30DEG:
                        results_rgb.extend(file_results_rgb)
                        
            except Exception as e:
                print(f"处理 {mask_filename} 时发生异常: {str(e)}")

    # ========== 保存结果 ==========
    print("\n" + "="*70)
    print("保存结果...")
    print("="*70)

    df_nosmooth = pd.DataFrame(results_nosmooth)
    df_nosmooth.to_csv(os.path.join(dir_results, "gdf_segments_no_smoothing.csv"), index=False)
    print(f"  ✓ 无平滑结果: gdf_segments_no_smoothing.csv ({len(df_nosmooth)} 条记录)")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_txt_path = os.path.join(dir_results, f"results_summary_{timestamp}.txt")

    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("无平滑坡度预测 - 结果摘要 (多线程版)\n")
        f.write("="*70 + "\n\n")
        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"使用线程数: {MAX_THREADS}\n\n")
        
        f.write("方法: 无平滑（直接从原始nDSM提取坡度）\n")
        f.write(f"  Pixel Size: {PIXEL_SIZE}m\n\n")
        
        if COMPARE_WITH_FIXED_30DEG:
            df_rgb = pd.DataFrame(results_rgb)
            df_rgb.to_csv(os.path.join(dir_results, f"gdf_segments_rgb_fixed{default_slope}deg.csv"), index=False)
            print(f"  ✓ RGB结果: gdf_segments_rgb_fixed{default_slope}deg.csv ({len(df_rgb)} 条记录)")
            
            df_comparison = pd.merge(df_nosmooth, df_rgb, on=['mask_id', 'segment_id', 'label'], suffixes=('_nosmooth', '_rgb'))
            df_comparison['slope_diff'] = df_comparison['slope_deg_nosmooth'] - df_comparison['slope_deg_rgb']
            df_comparison['gen_diff_kwh'] = df_comparison['annual_gen_kwh_nosmooth'] - df_comparison['annual_gen_kwh_rgb']
            df_comparison['gen_diff_pct'] = (df_comparison['gen_diff_kwh'] / df_comparison['annual_gen_kwh_rgb']) * 100
            
            df_comparison.to_csv(os.path.join(dir_results, "comparison_nosmooth_vs_rgb.csv"), index=False)
            print(f"  ✓ 对比结果: comparison_nosmooth_vs_rgb.csv ({len(df_comparison)} 条记录)")
            
            f.write("实验模式: 对比模式\n")
            f.write(f"对比基准: 固定{default_slope}度\n\n")
            f.write("="*70 + "\n")
            f.write(f"无平滑方法 vs RGB（固定{default_slope}°）对比分析\n")
            f.write("="*70 + "\n\n")
            f.write(f"总屋顶分割数: {len(df_comparison)}\n\n")
            
            f.write("【坡度统计】\n")
            f.write(f"  RGB: 固定{default_slope}° (平屋顶0°)\n")
            f.write(f"  无平滑真实坡度范围: {df_comparison['slope_deg_nosmooth'].min():.0f}° - {df_comparison['slope_deg_nosmooth'].max():.0f}°\n")
            f.write(f"  无平滑平均坡度: {df_comparison['slope_deg_nosmooth'].mean():.2f}° ± {df_comparison['slope_deg_nosmooth'].std():.2f}°\n\n")
            
            f.write("【年发电量统计】\n")
            f.write(f"  RGB总发电量: {df_comparison['annual_gen_kwh_rgb'].sum():,.0f} kWh/年\n")
            f.write(f"  无平滑总发电量: {df_comparison['annual_gen_kwh_nosmooth'].sum():,.0f} kWh/年\n")
            f.write(f"  差异: {df_comparison['gen_diff_kwh'].sum():,.0f} kWh/年 ({df_comparison['gen_diff_kwh'].sum()/df_comparison['annual_gen_kwh_rgb'].sum()*100:+.2f}%)\n\n")
            
            print("\n" + "="*70)
            print(f"无平滑方法 vs RGB（固定{default_slope}°）对比分析")
            print("="*70)
            print(f"\n总屋顶分割数: {len(df_comparison)}")
            print(f"\n【坡度统计】")
            print(f"  RGB: 固定{default_slope}° (平屋顶0°)")
            print(f"  无平滑真实坡度范围: {df_comparison['slope_deg_nosmooth'].min():.0f}° - {df_comparison['slope_deg_nosmooth'].max():.0f}°")
            print(f"  无平滑平均坡度: {df_comparison['slope_deg_nosmooth'].mean():.2f}° ± {df_comparison['slope_deg_nosmooth'].std():.2f}°")
            print(f"\n【年发电量统计】")
            print(f"  RGB总发电量: {df_comparison['annual_gen_kwh_rgb'].sum():,.0f} kWh/年")
            print(f"  无平滑总发电量: {df_comparison['annual_gen_kwh_nosmooth'].sum():,.0f} kWh/年")
            print(f"  差异: {df_comparison['gen_diff_kwh'].sum():,.0f} kWh/年 ({df_comparison['gen_diff_kwh'].sum()/df_comparison['annual_gen_kwh_rgb'].sum()*100:+.2f}%)")
            
        else:
            f.write("实验模式: 单一模式\n")
            f.write("仅运行无平滑方法\n\n")
            f.write("="*70 + "\n")
            f.write("无平滑方法结果统计\n")
            f.write("="*70 + "\n\n")
            f.write(f"总屋顶分割数: {len(df_nosmooth)}\n\n")
            
            f.write("【坡度统计】\n")
            f.write(f"  坡度范围: {df_nosmooth['slope_deg'].min():.0f}° - {df_nosmooth['slope_deg'].max():.0f}°\n")
            f.write(f"  平均坡度: {df_nosmooth['slope_deg'].mean():.2f}° ± {df_nosmooth['slope_deg'].std():.2f}°\n")
            f.write(f"  中位数坡度: {df_nosmooth['slope_deg'].median():.0f}°\n\n")
            
            f.write("【发电量统计】\n")
            f.write(f"  总发电量: {df_nosmooth['annual_gen_kwh'].sum():,.0f} kWh/年\n")
            f.write(f"  平均发电量: {df_nosmooth['annual_gen_kwh'].mean():,.0f} kWh/年\n")
            f.write(f"  总装机容量: {df_nosmooth['peak_power_kw'].sum():,.1f} kW\n")
            f.write(f"  总光伏组件数: {df_nosmooth['num_modules'].sum():,.0f} 个\n\n")
            
            print("\n" + "="*70)
            print("无平滑方法结果统计")
            print("="*70)
            print(f"\n总屋顶分割数: {len(df_nosmooth)}")
            print(f"\n【坡度统计】")
            print(f"  坡度范围: {df_nosmooth['slope_deg'].min():.0f}° - {df_nosmooth['slope_deg'].max():.0f}°")
            print(f"  平均坡度: {df_nosmooth['slope_deg'].mean():.2f}° ± {df_nosmooth['slope_deg'].std():.2f}°")
            print(f"  中位数坡度: {df_nosmooth['slope_deg'].median():.0f}°")
            print(f"\n【发电量统计】")
            print(f"  总发电量: {df_nosmooth['annual_gen_kwh'].sum():,.0f} kWh/年")
            print(f"  平均发电量: {df_nosmooth['annual_gen_kwh'].mean():,.0f} kWh/年")
            print(f"  总装机容量: {df_nosmooth['peak_power_kw'].sum():,.1f} kW")
            print(f"  总光伏组件数: {df_nosmooth['num_modules'].sum():,.0f} 个")
        
        f.write("="*70 + "\n")
        f.write("完成!\n")
        f.write("="*70 + "\n")

    print(f"  ✓ 统计摘要: {os.path.basename(summary_txt_path)}")
    print("\n" + "="*70)
    print("完成!")
    print("="*70)
