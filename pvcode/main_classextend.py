import os
import sys
import threading
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from scipy.ndimage import distance_transform_edt
from matplotlib import pyplot as plt

from definitions import epsg_metric_germany

# 注意：这里建议使用 classextend 版本
from electricity_generation_classextend import pv_electricity_generation
from masks_to_vector_classextend import segment_simplify_and_add_azimuth
from module_placement_classextend import module_placement, create_pv_modules_gdf
from spatial_operations_classextend import get_image_gdf_in_directory


# =============================================================================
# 线程安全结果存储
# =============================================================================

results_nosmooth = []
results_rgb = []
results_lock = threading.Lock()


# =============================================================================
# 配置区域
# =============================================================================

dir_roof_segment_masks = './数据集/RID2_标准18类/gt_test_roof_segment/'
dir_roof_superstructure_masks = './数据集/RID2_标准数据集/gt_test_superstructure/'
dir_geotifs = './img_tif/'
dir_ndsm = './height_labels/'
dir_pvgis_cache = '/mnt/yjs/tmp/RID2/pvgis_cache'
dir_results = './结果/RID2_标准18类/'

# GT 数据目录，用于基准评估
dir_gt_segment_masks = './数据集/RID2_标准18类/gt_test_roof_segment/'

EVALUATE_BASELINE = False
HUBER_DELTA = 5.0

PIXEL_SIZE = 0.08
COMPARE_WITH_FIXED_30DEG = False
MAX_THREADS = 10

visualize = False
bg_is_0 = True

superstructure_classes = ['unknown', 'pvmodule']

a_min_segments = 2
a_min_superstructures = 0.5

pv_module_peak_power = 400
pv_module_height = 1.7
pv_module_width = 1

default_slope = 40


# =============================================================================
# 屋顶朝向类别模式
# =============================================================================
# 6  : 0 background + 5类  N E S W flat
# 10 : 0 background + 9类  N NE E SE S SW W NW flat
# 18 : 0 background + 17类 N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW flat
#
# 你生成的 18 类 mask 应该是：
# 0 background
# 1 N
# 2 NNE
# ...
# 17 flat
# =============================================================================

SEGMENT_CLASS_MODE = 18  # 可改成 6 / 10 / 18

SEGMENT_CLASSES_MAP = {
    6: [
        'N', 'E', 'S', 'W', 'flat'
    ],
    10: [
        'N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW', 'flat'
    ],
    18: [
        'N', 'NNE', 'NE', 'ENE',
        'E', 'ESE', 'SE', 'SSE',
        'S', 'SSW', 'SW', 'WSW',
        'W', 'WNW', 'NW', 'NNW',
        'flat'
    ]
}

if SEGMENT_CLASS_MODE not in SEGMENT_CLASSES_MAP:
    raise ValueError(f"SEGMENT_CLASS_MODE 必须是 6 / 10 / 18，当前是: {SEGMENT_CLASS_MODE}")

segment_classes = SEGMENT_CLASSES_MAP[SEGMENT_CLASS_MODE]

# 当前工程默认假设输入 mask 是：
# 0=background, 1..N 按 segment_classes 顺序编码
#
# 如果你使用的是 RID2 原始 6 类 PNG mask，它的 README 编码是：
# 0=N, 1=E, 2=S, 3=W, 4=flat, 5=background
# 这种情况下把下面改成 True。
USE_RID2_ORIGINAL_6CLASS_ENCODING = False


# =============================================================================
# 地理参考处理
# =============================================================================

def get_georeference_from_tif(tif_path):
    """从 TIF 文件读取地理参考信息。"""
    with rasterio.open(tif_path) as src:
        return {
            'bounds': src.bounds,
            'crs': src.crs,
            'transform': src.transform,
            'width': src.width,
            'height': src.height
        }


def load_ndsm_with_georeference(ndsm_path, tif_path):
    """加载 nDSM，并从对应 TIF 获取地理参考。"""
    ndsm_data = cv2.imread(ndsm_path, cv2.IMREAD_UNCHANGED)

    if ndsm_data is None:
        raise ValueError(f"无法读取 nDSM 文件: {ndsm_path}")

    if len(ndsm_data.shape) == 3:
        ndsm_data = ndsm_data[:, :, 0]

    ndsm_data = ndsm_data.astype(float)
    georeference = get_georeference_from_tif(tif_path)

    return ndsm_data, georeference


def raster_to_vector_with_georeference(raster_mask, georeference, label_classes, bg_is_0=True):
    """
    使用 TIF 的 transform / crs 将 mask 转为矢量。
    mask 编码约定：
    0 = background
    1..N = label_classes[value - 1]
    """
    transform = georeference['transform']
    crs = georeference['crs']

    geometries = []

    for geom, value in shapes(raster_mask.astype(np.int16), transform=transform):
        value = int(round(value))

        if bg_is_0 and value == 0:
            continue

        if 1 <= value <= len(label_classes):
            geometries.append({
                'geometry': shape(geom),
                'label': label_classes[value - 1],
                'value': value
            })

    if len(geometries) == 0:
        gdf = gpd.GeoDataFrame(columns=['geometry', 'label', 'value'], crs=crs)
    else:
        gdf = gpd.GeoDataFrame(geometries, crs=crs)

    return gdf


# =============================================================================
# mask 重映射
# =============================================================================

def remap_rid2_original_6class_mask(mask):
    """
    RID2 原始 6 类 PNG mask 的 README 编码：
    0=N, 1=E, 2=S, 3=W, 4=flat, 5=background

    转成项目统一编码：
    0=background, 1=N, 2=E, 3=S, 4=W, 5=flat
    """
    remapped = np.zeros_like(mask, dtype=np.uint8)

    remapped[mask == 5] = 0  # background
    remapped[mask == 0] = 1  # N
    remapped[mask == 1] = 2  # E
    remapped[mask == 2] = 3  # S
    remapped[mask == 3] = 4  # W
    remapped[mask == 4] = 5  # flat

    return remapped


def remap_segment_mask(mask):
    """
    通用屋顶朝向 mask 读取。

    默认输入编码：
    0=background, 1..N 按 segment_classes 顺序编码。

    例如：
    6类:
      0 background
      1 N
      2 E
      3 S
      4 W
      5 flat

    10类:
      0 background
      1 N
      2 NE
      3 E
      4 SE
      5 S
      6 SW
      7 W
      8 NW
      9 flat

    18类:
      0 background
      1 N
      2 NNE
      ...
      17 flat
    """
    if mask is None:
        raise ValueError("mask is None")

    mask = mask.astype(np.uint8)

    if USE_RID2_ORIGINAL_6CLASS_ENCODING:
        if SEGMENT_CLASS_MODE != 6:
            raise ValueError(
                "USE_RID2_ORIGINAL_6CLASS_ENCODING=True 只适用于 SEGMENT_CLASS_MODE=6。"
                "10/18 类 mask 应使用 0=background, 1..N 的标准编码。"
            )
        mask = remap_rid2_original_6class_mask(mask)

    valid_ids = set(range(len(segment_classes) + 1))
    unique_vals = set(np.unique(mask).astype(int).tolist())
    invalid_vals = unique_vals - valid_ids

    if invalid_vals:
        raise ValueError(
            f"分割掩码中存在非法类别值: {sorted(invalid_vals)}。"
            f" 当前 SEGMENT_CLASS_MODE={SEGMENT_CLASS_MODE},"
            f" 合法范围应为 0~{len(segment_classes)}。"
            f" 当前 unique={sorted(unique_vals)}"
        )

    return mask


def remap_superstructure_mask(mask):
    """
    附属结构二分类：
    0 background
    >0 obstacle
    """
    if mask is None:
        raise ValueError("superstructure mask is None")

    remapped = np.zeros_like(mask, dtype=np.uint8)
    remapped[mask > 0] = 1

    return remapped


def load_and_remap_mask(file_path, mask_type='segment'):
    """读取并重映射 mask。"""
    mask = cv2.imread(file_path, 0)

    if mask is None:
        raise ValueError(f"无法读取 mask 文件: {file_path}")

    if mask_type == 'segment':
        return remap_segment_mask(mask)
    elif mask_type == 'superstructure':
        return remap_superstructure_mask(mask)
    else:
        raise ValueError(f"未知 mask_type: {mask_type}")


def get_vector_labels_remapped(
    file_path,
    mask_filename,
    gdf_images,
    label_classes,
    area_threshold,
    tif_dir,
    mask_type='segment',
    bg_is_0=True
):
    """
    RID2 版本：
    读取 mask -> 重映射 -> 使用同名 TIF 的 georeference 转成矢量。
    """
    try:
        mask = load_and_remap_mask(file_path, mask_type=mask_type)
    except Exception as e:
        print(f"    错误：读取或重映射 mask 失败 {file_path}: {e}")
        return gpd.GeoDataFrame()

    tif_filename = mask_filename + '.tif'
    tif_path = os.path.join(tif_dir, tif_filename)

    if not os.path.exists(tif_path):
        print(f"    警告：找不到对应 TIF: {tif_path}")
        return gpd.GeoDataFrame()

    try:
        georeference = get_georeference_from_tif(tif_path)

        target_crs_string = f'EPSG:{epsg_metric_germany}'
        georeference['crs'] = target_crs_string

        gdf_labels = raster_to_vector_with_georeference(
            mask,
            georeference,
            label_classes,
            bg_is_0=bg_is_0
        )

    except Exception as e:
        print(f"    错误：mask 转矢量失败 {mask_filename}: {e}")
        return gpd.GeoDataFrame()

    if len(gdf_labels) == 0:
        return gdf_labels

    gdf_labels = gdf_labels[gdf_labels.geometry.area > area_threshold]
    gdf_labels = gdf_labels.reset_index(drop=True)

    return gdf_labels


# =============================================================================
# 信息保留度计算
# =============================================================================

def calculate_information_metrics(ndsm_original, ndsm_processed, segment_mask):
    """计算信息保留度指标。"""
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

    var_orig = np.var(original)
    var_proc = np.var(processed)
    variance_retention = var_proc / (var_orig + 1e-6)

    grad_y_orig, grad_x_orig = np.gradient(ndsm_original, PIXEL_SIZE)
    grad_y_proc, grad_x_proc = np.gradient(ndsm_processed, PIXEL_SIZE)

    grad_mag_orig = np.sqrt(grad_x_orig ** 2 + grad_y_orig ** 2)[roof_mask]
    grad_mag_proc = np.sqrt(grad_x_proc ** 2 + grad_y_proc ** 2)[roof_mask]

    gradient_retention = grad_mag_proc.mean() / (grad_mag_orig.mean() + 1e-6)

    hist_orig, _ = np.histogram(original, bins=50, density=True)
    hist_proc, _ = np.histogram(processed, bins=50, density=True)

    entropy_orig = -np.sum(hist_orig * np.log(hist_orig + 1e-10))
    entropy_proc = -np.sum(hist_proc * np.log(hist_proc + 1e-10))
    entropy_retention = entropy_proc / (entropy_orig + 1e-6)

    info_retention_avg = (
        variance_retention +
        gradient_retention +
        entropy_retention
    ) / 3

    return {
        'variance_retention': float(np.clip(variance_retention, 0, 1)),
        'gradient_retention': float(np.clip(gradient_retention, 0, 1)),
        'entropy_retention': float(np.clip(entropy_retention, 0, 1)),
        'info_retention_avg': float(np.clip(info_retention_avg, 0, 1))
    }


# =============================================================================
# 无平滑坡度提取
# =============================================================================

def extract_slopes_no_smoothing(gdf_segments, ndsm_data_or_path, segment_mask_data_or_path):
    """
    无平滑方法：
    直接从原始 nDSM 提取坡度，不做平滑。

    注意：
    label_to_id 根据当前 segment_classes 动态生成，
    因此支持 6 / 10 / 18 类。
    """
    slopes = []

    if isinstance(ndsm_data_or_path, np.ndarray):
        ndsm_data = ndsm_data_or_path
    else:
        if not os.path.exists(ndsm_data_or_path):
            print(f"    警告：nDSM 文件不存在: {ndsm_data_or_path}")
            return [
                default_slope if not np.isnan(seg.azimuth) else 0
                for seg in gdf_segments.itertuples()
            ]

        ndsm_data = cv2.imread(ndsm_data_or_path, cv2.IMREAD_UNCHANGED)

        if ndsm_data is None:
            print(f"    警告：无法读取 nDSM: {ndsm_data_or_path}")
            return [
                default_slope if not np.isnan(seg.azimuth) else 0
                for seg in gdf_segments.itertuples()
            ]

        if len(ndsm_data.shape) == 3:
            ndsm_data = ndsm_data[:, :, 0]

        ndsm_data = ndsm_data.astype(float)

    grad_y, grad_x = np.gradient(ndsm_data, PIXEL_SIZE)
    slope_angle = np.degrees(np.arctan(np.sqrt(grad_x ** 2 + grad_y ** 2)))

    if isinstance(segment_mask_data_or_path, np.ndarray):
        segment_mask = segment_mask_data_or_path
    else:
        if not os.path.exists(segment_mask_data_or_path):
            print(f"    警告：segment mask 不存在: {segment_mask_data_or_path}")
            segment_mask = None
        else:
            raw_mask = cv2.imread(segment_mask_data_or_path, 0)
            if raw_mask is None:
                print(f"    警告：无法读取 segment mask: {segment_mask_data_or_path}")
                segment_mask = None
            else:
                segment_mask = remap_segment_mask(raw_mask)

    label_to_id = {label: i + 1 for i, label in enumerate(segment_classes)}

    for idx, segment in gdf_segments.iterrows():
        try:
            label = segment['label']

            if label == 'flat' or np.isnan(segment.get('azimuth', np.nan)):
                slopes.append(0)
                continue

            if segment_mask is None:
                slopes.append(default_slope)
                continue

            label_id = label_to_id.get(label, 0)

            if label_id == 0:
                slopes.append(default_slope)
                continue

            mask = (segment_mask == label_id).astype(np.uint8)

            if mask.sum() == 0:
                slopes.append(default_slope)
                continue

            dist_transform = distance_transform_edt(mask)
            weights = dist_transform / (dist_transform.max() + 1e-6)

            segment_slopes = slope_angle[mask == 1]
            segment_weights = weights[mask == 1]

            if len(segment_slopes) > 10:
                q1, q3 = np.percentile(segment_slopes, [25, 75])
                iqr = q3 - q1

                lower_bound = q1 - 1.5 * iqr
                upper_bound = q3 + 1.5 * iqr

                valid_mask = (
                    (segment_slopes >= lower_bound) &
                    (segment_slopes <= upper_bound)
                )

                if valid_mask.sum() > 5:
                    avg_slope = np.average(
                        segment_slopes[valid_mask],
                        weights=segment_weights[valid_mask]
                    )
                else:
                    avg_slope = np.average(
                        segment_slopes,
                        weights=segment_weights
                    )

                avg_slope = int(np.clip(np.round(avg_slope), 0, 90))
                slopes.append(avg_slope)
            else:
                slopes.append(default_slope)

        except Exception as e:
            print(f"    区域 {idx} 坡度提取失败: {e}")
            slopes.append(default_slope)

    return slopes


# =============================================================================
# 基准评估
# =============================================================================

def evaluate_baseline_methods(gdf_images, mask_files, max_images=None):
    """
    基准评估：
    NoSmoothing vs FixedSlope。
    """
    if max_images is not None:
        mask_files = mask_files[:max_images]

    print(f"\n{'=' * 70}")
    print("基准评估：无平滑方法 vs 固定坡度方法")
    print(f"SEGMENT_CLASS_MODE: {SEGMENT_CLASS_MODE}")
    print(f"segment_classes: {segment_classes}")
    print(f"测试图片数: {len(mask_files)}")
    print(f"{'=' * 70}\n")

    if not os.path.exists(dir_gt_segment_masks):
        print(f"错误：GT 数据目录不存在: {dir_gt_segment_masks}")
        return

    huber_values_nosmooth = []

    print("\n第一步：计算 global_huber_max\n")

    for mask_idx, mask_filename in enumerate(mask_files):
        print(f"[步骤1-{mask_idx + 1}/{len(mask_files)}] {mask_filename}")

        mask_id = mask_filename[:-4]

        pred_mask_path = os.path.join(dir_roof_segment_masks, mask_filename)
        gt_mask_path = os.path.join(dir_gt_segment_masks, mask_filename)
        ndsm_path = os.path.join(dir_ndsm, mask_filename.replace('.png', '.tif'))
        tif_path = os.path.join(dir_geotifs, mask_filename.replace('.png', '.tif'))

        if (
            not os.path.exists(pred_mask_path) or
            not os.path.exists(gt_mask_path) or
            not os.path.exists(ndsm_path) or
            not os.path.exists(tif_path)
        ):
            print("  跳过：文件缺失")
            continue

        try:
            ndsm_data, _ = load_ndsm_with_georeference(ndsm_path, tif_path)

            gt_mask = cv2.imread(gt_mask_path, 0)
            pred_mask = cv2.imread(pred_mask_path, 0)

            gt_mask = remap_segment_mask(gt_mask)
            pred_mask = remap_segment_mask(pred_mask)

            gdf_gt = get_vector_labels_remapped(
                gt_mask_path,
                mask_id,
                gdf_images,
                segment_classes,
                a_min_segments,
                dir_geotifs,
                mask_type='segment',
                bg_is_0=True
            )

            if len(gdf_gt) == 0:
                print("  跳过：GT 无有效区域")
                continue

            gdf_gt, _, _ = segment_simplify_and_add_azimuth(
                gdf_gt,
                visualize=False
            )

            gdf_pred = get_vector_labels_remapped(
                pred_mask_path,
                mask_id,
                gdf_images,
                segment_classes,
                a_min_segments,
                dir_geotifs,
                mask_type='segment',
                bg_is_0=True
            )

            if len(gdf_pred) == 0:
                print("  跳过：Pred 无有效区域")
                continue

            gdf_pred, _, _ = segment_simplify_and_add_azimuth(
                gdf_pred,
                visualize=False
            )

            gt_slopes = extract_slopes_no_smoothing(
                gdf_gt,
                ndsm_data,
                gt_mask
            )
            pred_slopes = extract_slopes_no_smoothing(
                gdf_pred,
                ndsm_data,
                pred_mask
            )

            gt_mean = np.mean(gt_slopes)
            pred_mean = np.mean(pred_slopes)

            abs_error = abs(pred_mean - gt_mean)

            if abs_error <= HUBER_DELTA:
                huber = 0.5 * (abs_error ** 2)
            else:
                huber = HUBER_DELTA * (abs_error - 0.5 * HUBER_DELTA)

            huber_values_nosmooth.append(huber)

            print(f"  GT={len(gdf_gt)}, Pred={len(gdf_pred)}, Huber={huber:.4f}")

        except Exception as e:
            print(f"  跳过：处理失败 {e}")
            continue

    if len(huber_values_nosmooth) == 0:
        print("错误：没有成功处理任何图片，基准评估终止")
        return

    global_huber_max = max(huber_values_nosmooth)

    print(f"\n✓ global_huber_max = {global_huber_max:.4f}\n")

    results_eval_nosmooth = []
    results_eval_fixed = []

    print("\n第二步：完整评估\n")

    for mask_idx, mask_filename in enumerate(mask_files):
        print(f"[步骤2-{mask_idx + 1}/{len(mask_files)}] {mask_filename}")

        mask_id = mask_filename[:-4]

        pred_mask_path = os.path.join(dir_roof_segment_masks, mask_filename)
        gt_mask_path = os.path.join(dir_gt_segment_masks, mask_filename)
        ndsm_path = os.path.join(dir_ndsm, mask_filename.replace('.png', '.tif'))
        tif_path = os.path.join(dir_geotifs, mask_filename.replace('.png', '.tif'))

        if (
            not os.path.exists(pred_mask_path) or
            not os.path.exists(gt_mask_path) or
            not os.path.exists(ndsm_path) or
            not os.path.exists(tif_path)
        ):
            print("  跳过：文件缺失")
            continue

        try:
            ndsm_data, _ = load_ndsm_with_georeference(ndsm_path, tif_path)

            gt_mask = cv2.imread(gt_mask_path, 0)
            pred_mask = cv2.imread(pred_mask_path, 0)

            gt_mask = remap_segment_mask(gt_mask)
            pred_mask = remap_segment_mask(pred_mask)

            gdf_gt = get_vector_labels_remapped(
                gt_mask_path,
                mask_id,
                gdf_images,
                segment_classes,
                a_min_segments,
                dir_geotifs,
                mask_type='segment',
                bg_is_0=True
            )

            if len(gdf_gt) == 0:
                continue

            gdf_gt, _, _ = segment_simplify_and_add_azimuth(
                gdf_gt,
                visualize=False
            )

            gdf_pred = get_vector_labels_remapped(
                pred_mask_path,
                mask_id,
                gdf_images,
                segment_classes,
                a_min_segments,
                dir_geotifs,
                mask_type='segment',
                bg_is_0=True
            )

            if len(gdf_pred) == 0:
                continue

            gdf_pred, _, _ = segment_simplify_and_add_azimuth(
                gdf_pred,
                visualize=False
            )

            gt_slopes = extract_slopes_no_smoothing(
                gdf_gt,
                ndsm_data,
                gt_mask
            )
            pred_slopes = extract_slopes_no_smoothing(
                gdf_pred,
                ndsm_data,
                pred_mask
            )

            gt_mean = np.mean(gt_slopes)
            pred_mean = np.mean(pred_slopes)

            abs_error = abs(pred_mean - gt_mean)

            if abs_error <= HUBER_DELTA:
                huber_nosmooth = 0.5 * (abs_error ** 2)
            else:
                huber_nosmooth = HUBER_DELTA * (abs_error - 0.5 * HUBER_DELTA)

            info_gt = calculate_information_metrics(ndsm_data, ndsm_data.copy(), gt_mask)
            info_pred = calculate_information_metrics(ndsm_data, ndsm_data.copy(), pred_mask)

            info_avg = (
                info_gt['info_retention_avg'] +
                info_pred['info_retention_avg']
            ) / 2

            accuracy = 1 - (huber_nosmooth / global_huber_max)
            accuracy = np.clip(accuracy, 0, 1)

            error_norm = 1 - accuracy
            efficiency = info_avg / (error_norm + 0.1)

            results_eval_nosmooth.append({
                'image_id': mask_id,
                'method': 'NoSmoothing',
                'gt_mean_slope': gt_mean,
                'pred_mean_slope': pred_mean,
                'Huber': huber_nosmooth,
                'MAE': abs_error,
                'AvgInfoRetention': info_avg,
                'Efficiency': efficiency
            })

            gt_slopes_fixed = [
                default_slope if not np.isnan(az) else 0
                for az in gdf_gt["azimuth"]
            ]
            pred_slopes_fixed = [
                default_slope if not np.isnan(az) else 0
                for az in gdf_pred["azimuth"]
            ]

            gt_mean_fixed = np.mean(gt_slopes_fixed)
            pred_mean_fixed = np.mean(pred_slopes_fixed)

            abs_error_fixed = abs(pred_mean_fixed - gt_mean_fixed)

            if abs_error_fixed <= HUBER_DELTA:
                huber_fixed = 0.5 * (abs_error_fixed ** 2)
            else:
                huber_fixed = HUBER_DELTA * (abs_error_fixed - 0.5 * HUBER_DELTA)

            ndsm_fixed = np.full_like(ndsm_data, ndsm_data.mean())

            info_gt_fixed = calculate_information_metrics(ndsm_data, ndsm_fixed, gt_mask)
            info_pred_fixed = calculate_information_metrics(ndsm_data, ndsm_fixed, pred_mask)

            info_avg_fixed = (
                info_gt_fixed['info_retention_avg'] +
                info_pred_fixed['info_retention_avg']
            ) / 2

            accuracy_fixed = 1 - (huber_fixed / global_huber_max)
            accuracy_fixed = np.clip(accuracy_fixed, 0, 1)

            error_norm_fixed = 1 - accuracy_fixed
            efficiency_fixed = info_avg_fixed / (error_norm_fixed + 0.1)

            results_eval_fixed.append({
                'image_id': mask_id,
                'method': 'FixedSlope',
                'gt_mean_slope': gt_mean_fixed,
                'pred_mean_slope': pred_mean_fixed,
                'Huber': huber_fixed,
                'MAE': abs_error_fixed,
                'AvgInfoRetention': info_avg_fixed,
                'Efficiency': efficiency_fixed
            })

            print(
                f"  GT={len(gdf_gt)}, Pred={len(gdf_pred)}, "
                f"Huber={huber_nosmooth:.4f}, Fixed={huber_fixed:.4f}"
            )

        except Exception as e:
            print(f"  跳过：处理失败 {e}")
            continue

    df_nosmooth = pd.DataFrame(results_eval_nosmooth)
    df_fixed = pd.DataFrame(results_eval_fixed)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_txt = os.path.join(
        dir_results,
        f"baseline_evaluation_mode{SEGMENT_CLASS_MODE}_{timestamp}.txt"
    )

    with open(out_txt, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RID2 classextend 基准评估报告\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"SEGMENT_CLASS_MODE: {SEGMENT_CLASS_MODE}\n")
        f.write(f"segment_classes: {segment_classes}\n")
        f.write(f"global_huber_max: {global_huber_max:.4f}\n")
        f.write(f"测试图片数: {len(df_nosmooth)}\n\n")

        if len(df_nosmooth) > 0:
            f.write("[NoSmoothing]\n")
            f.write(f"平均 Huber: {df_nosmooth['Huber'].mean():.4f}\n")
            f.write(f"平均 MAE: {df_nosmooth['MAE'].mean():.4f}\n")
            f.write(f"平均信息保留: {df_nosmooth['AvgInfoRetention'].mean():.1%}\n")
            f.write(f"平均效率: {df_nosmooth['Efficiency'].mean():.4f}\n\n")

        if len(df_fixed) > 0:
            f.write("[FixedSlope]\n")
            f.write(f"平均 Huber: {df_fixed['Huber'].mean():.4f}\n")
            f.write(f"平均 MAE: {df_fixed['MAE'].mean():.4f}\n")
            f.write(f"平均信息保留: {df_fixed['AvgInfoRetention'].mean():.1%}\n")
            f.write(f"平均效率: {df_fixed['Efficiency'].mean():.4f}\n\n")

    print(f"\n✓ 基准评估完成: {out_txt}\n")


# =============================================================================
# 单文件处理
# =============================================================================

def process_single_file(mask_filename, gdf_images, target_crs_string):
    """处理单个 mask 文件，供线程池调用。"""
    thread_id = threading.get_ident()

    print(f"\n[线程 {thread_id}] 开始处理: {mask_filename}")

    try:
        file_path = os.path.join(dir_roof_segment_masks, mask_filename)
        mask_id = mask_filename[:-4]

        gdf_segments = get_vector_labels_remapped(
            file_path,
            mask_id,
            gdf_images,
            segment_classes,
            a_min_segments,
            dir_geotifs,
            mask_type='segment',
            bg_is_0=bg_is_0
        )

        if len(gdf_segments) == 0:
            print(f"[线程 {thread_id}] 警告：{mask_filename} 无有效 roof segment，跳过")
            return [], []

        gdf_segments = gdf_segments.set_geometry('geometry')
        gdf_segments = gdf_segments.set_crs(
            target_crs_string,
            allow_override=True
        )

        gdf_segments_copy = gpd.GeoDataFrame(
            gdf_segments.drop(columns=['geometry']),
            geometry=gdf_segments.geometry.copy(),
            crs=gdf_segments.crs
        )

        try:
            gdf_segments, _, _ = segment_simplify_and_add_azimuth(
                gdf_segments_copy,
                visualize=False
            )

            if gdf_segments.crs is None:
                gdf_segments = gdf_segments.set_geometry('geometry')
                gdf_segments = gdf_segments.set_crs(
                    target_crs_string,
                    allow_override=True
                )

        except Exception as e:
            print(f"[线程 {thread_id}] 错误：segment_simplify_and_add_azimuth 失败: {e}")
            return [], []

        file_path_super = os.path.join(dir_roof_superstructure_masks, mask_filename)

        gdf_superstructures = get_vector_labels_remapped(
            file_path_super,
            mask_id,
            gdf_images,
            superstructure_classes,
            a_min_superstructures,
            dir_geotifs,
            mask_type='superstructure',
            bg_is_0=bg_is_0
        )

        ndsm_file = os.path.join(dir_ndsm, mask_filename.replace('.png', '.tif'))
        segment_mask_file = os.path.join(dir_roof_segment_masks, mask_filename)
        tif_file = os.path.join(dir_geotifs, mask_filename.replace('.png', '.tif'))

        if not os.path.exists(ndsm_file):
            print(f"[线程 {thread_id}] nDSM 缺失: {ndsm_file}")
            return [], []

        if not os.path.exists(tif_file):
            print(f"[线程 {thread_id}] TIF 缺失: {tif_file}")
            return [], []

        ndsm_data, _ = load_ndsm_with_georeference(ndsm_file, tif_file)

        raw_segment_mask = cv2.imread(segment_mask_file, 0)
        if raw_segment_mask is None:
            print(f"[线程 {thread_id}] 无法读取 segment mask: {segment_mask_file}")
            return [], []

        segment_mask = remap_segment_mask(raw_segment_mask)

        file_results_nosmooth = []
        file_results_rgb = []

        # ---------------------------------------------------------------------
        # 分支 1：无平滑真实坡度
        # ---------------------------------------------------------------------
        print(f"[线程 {thread_id}] [无平滑] 计算真实坡度...")

        gdf_nosmooth = gdf_segments.copy()

        gdf_nosmooth["slopes"] = extract_slopes_no_smoothing(
            gdf_nosmooth,
            ndsm_data,
            segment_mask
        )

        print(f"[线程 {thread_id}] slopes={gdf_nosmooth['slopes'].tolist()}")

        alignment_nosmooth, gdf_modules_v_nosmooth, gdf_modules_h_nosmooth, azimuth_nosmooth = module_placement(
            gdf_nosmooth,
            gdf_nosmooth["azimuth"],
            gdf_nosmooth["slopes"],
            gdf_superstructures,
            pv_module_height,
            pv_module_width
        )

        gdf_modules_nosmooth = create_pv_modules_gdf(
            alignment_nosmooth,
            gdf_modules_v_nosmooth,
            gdf_modules_h_nosmooth,
            azimuth_nosmooth
        )

        gdf_modules_nosmooth = gdf_modules_nosmooth.to_crs(epsg_metric_germany)

        gdf_nosmooth["pv_modules_per_segment"] = [
            len(mp.geometry.geoms)
            for mp in gdf_modules_nosmooth.iloc
        ]

        gdf_nosmooth["pv_peak_power_per_segment"] = [
            len(mp.geometry.geoms) * pv_module_peak_power / 1000
            for mp in gdf_modules_nosmooth.iloc
        ]

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

        gdf_nosmooth["electricity_generations"] = [
            np.sum(e)
            for e in electricity_generations_nosmooth
        ]

        for i in range(len(gdf_nosmooth)):
            file_results_nosmooth.append({
                'mask_id': mask_id,
                'segment_id': i,
                'class_mode': SEGMENT_CLASS_MODE,
                'label': gdf_nosmooth.iloc[i]['label'],
                'area_m2': gdf_nosmooth.iloc[i].geometry.area,
                'azimuth': gdf_nosmooth.iloc[i]['azimuth_incl_flat'],
                'slope_deg': gdf_nosmooth.iloc[i]['slopes'],
                'num_modules': gdf_nosmooth.iloc[i]['pv_modules_per_segment'],
                'peak_power_kw': gdf_nosmooth.iloc[i]['pv_peak_power_per_segment'],
                'annual_gen_kwh': gdf_nosmooth.iloc[i]['electricity_generations']
            })

        # ---------------------------------------------------------------------
        # 分支 2：固定坡度 RGB baseline
        # ---------------------------------------------------------------------
        if COMPARE_WITH_FIXED_30DEG:
            print(f"[线程 {thread_id}] [固定坡度] 默认 {default_slope}°...")

            gdf_rgb = gdf_segments.copy()

            gdf_rgb["slopes"] = [
                default_slope if not np.isnan(az) else 0
                for az in gdf_rgb["azimuth"]
            ]

            alignment_rgb, gdf_modules_v_rgb, gdf_modules_h_rgb, azimuth_rgb = module_placement(
                gdf_rgb,
                gdf_rgb["azimuth"],
                gdf_rgb["slopes"],
                gdf_superstructures,
                pv_module_height,
                pv_module_width
            )

            gdf_modules_rgb = create_pv_modules_gdf(
                alignment_rgb,
                gdf_modules_v_rgb,
                gdf_modules_h_rgb,
                azimuth_rgb
            )

            gdf_modules_rgb = gdf_modules_rgb.to_crs(epsg_metric_germany)

            gdf_rgb["pv_modules_per_segment"] = [
                len(mp.geometry.geoms)
                for mp in gdf_modules_rgb.iloc
            ]

            gdf_rgb["pv_peak_power_per_segment"] = [
                len(mp.geometry.geoms) * pv_module_peak_power / 1000
                for mp in gdf_modules_rgb.iloc
            ]

            gdf_rgb["azimuth_incl_flat"] = azimuth_rgb

            electricity_generations_rgb = pv_electricity_generation(
                location=gs_location,
                azimuths=gdf_rgb["azimuth_incl_flat"],
                slopes=gdf_rgb["slopes"],
                peak_powers=gdf_rgb["pv_peak_power_per_segment"],
                dir_pvgis_cache=dir_pvgis_cache
            )

            gdf_rgb["electricity_generations"] = [
                np.sum(e)
                for e in electricity_generations_rgb
            ]

            for i in range(len(gdf_rgb)):
                file_results_rgb.append({
                    'mask_id': mask_id,
                    'segment_id': i,
                    'class_mode': SEGMENT_CLASS_MODE,
                    'label': gdf_rgb.iloc[i]['label'],
                    'area_m2': gdf_rgb.iloc[i].geometry.area,
                    'azimuth': gdf_rgb.iloc[i]['azimuth_incl_flat'],
                    'slope_deg': gdf_rgb.iloc[i]['slopes'],
                    'num_modules': gdf_rgb.iloc[i]['pv_modules_per_segment'],
                    'peak_power_kw': gdf_rgb.iloc[i]['pv_peak_power_per_segment'],
                    'annual_gen_kwh': gdf_rgb.iloc[i]['electricity_generations']
                })

        print(f"[线程 {thread_id}] 完成: {mask_filename}")

        return file_results_nosmooth, file_results_rgb

    except Exception as e:
        print(f"[线程 {thread_id}] 处理 {mask_filename} 出错: {e}")
        return [], []


# =============================================================================
# 主程序
# =============================================================================

if __name__ == "__main__":

    if not os.path.isdir(dir_pvgis_cache):
        os.makedirs(dir_pvgis_cache)

    if not os.path.isdir(dir_results):
        os.makedirs(dir_results)

    print(f"\n{'=' * 70}")
    print("RID2 无平滑坡度预测 - classextend 多线程版")
    print(f"{'=' * 70}")
    print(f"SEGMENT_CLASS_MODE: {SEGMENT_CLASS_MODE}")
    print(f"segment_classes: {segment_classes}")
    print(f"Pixel Size: {PIXEL_SIZE} m")
    print(f"线程数: {MAX_THREADS}")
    print(f"USE_RID2_ORIGINAL_6CLASS_ENCODING: {USE_RID2_ORIGINAL_6CLASS_ENCODING}")

    if COMPARE_WITH_FIXED_30DEG:
        print(f"实验模式: 对比固定坡度 {default_slope}°")
    else:
        print("实验模式: 仅无平滑方法")

    print(f"{'=' * 70}\n")

    print("读取 geotifs image footprints...")
    gdf_images = get_image_gdf_in_directory(dir_geotifs)

    target_crs_string = f'EPSG:{epsg_metric_germany}'

    if gdf_images.crs is None:
        gdf_images = gdf_images.set_crs(epsg=epsg_metric_germany)
    else:
        gdf_images = gdf_images.to_crs(target_crs_string)

    mask_filenames = sorted([
        f for f in os.listdir(dir_roof_segment_masks)
        if f.lower().endswith('.png')
    ])

    print(f"发现 {len(mask_filenames)} 个 segment mask 文件")

    if EVALUATE_BASELINE:
        evaluate_baseline_methods(
            gdf_images,
            mask_filenames,
            max_images=None
        )

    print("\n" + "=" * 70)
    print(f"开始并行处理 {len(mask_filenames)} 个文件，线程数={MAX_THREADS}")
    print("=" * 70)

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        futures = {
            executor.submit(
                process_single_file,
                mask_filename,
                gdf_images,
                target_crs_string
            ): mask_filename
            for mask_filename in mask_filenames
        }

        for future in as_completed(futures):
            mask_filename = futures[future]

            try:
                file_results_nosmooth, file_results_rgb = future.result()

                with results_lock:
                    results_nosmooth.extend(file_results_nosmooth)

                    if COMPARE_WITH_FIXED_30DEG:
                        results_rgb.extend(file_results_rgb)

            except Exception as e:
                print(f"处理 {mask_filename} 时发生异常: {e}")

    print("\n" + "=" * 70)
    print("保存结果")
    print("=" * 70)

    df_nosmooth = pd.DataFrame(results_nosmooth)

    out_csv_nosmooth = os.path.join(
        dir_results,
        f"gdf_segments_no_smoothing_RID2_mode{SEGMENT_CLASS_MODE}.csv"
    )

    df_nosmooth.to_csv(out_csv_nosmooth, index=False)

    print(f"✓ 无平滑结果: {out_csv_nosmooth}")
    print(f"  记录数: {len(df_nosmooth)}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    summary_txt_path = os.path.join(
        dir_results,
        f"results_summary_RID2_mode{SEGMENT_CLASS_MODE}_{timestamp}.txt"
    )

    with open(summary_txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("RID2 无平滑坡度预测 - classextend 结果摘要\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"SEGMENT_CLASS_MODE: {SEGMENT_CLASS_MODE}\n")
        f.write(f"segment_classes: {segment_classes}\n")
        f.write(f"PIXEL_SIZE: {PIXEL_SIZE}\n")
        f.write(f"MAX_THREADS: {MAX_THREADS}\n")
        f.write(f"USE_RID2_ORIGINAL_6CLASS_ENCODING: {USE_RID2_ORIGINAL_6CLASS_ENCODING}\n\n")

        if COMPARE_WITH_FIXED_30DEG:
            df_rgb = pd.DataFrame(results_rgb)

            out_csv_rgb = os.path.join(
                dir_results,
                f"gdf_segments_rgb_fixed{default_slope}deg_RID2_mode{SEGMENT_CLASS_MODE}.csv"
            )

            df_rgb.to_csv(out_csv_rgb, index=False)

            print(f"✓ 固定坡度结果: {out_csv_rgb}")
            print(f"  记录数: {len(df_rgb)}")

            df_comparison = pd.merge(
                df_nosmooth,
                df_rgb,
                on=['mask_id', 'segment_id', 'label', 'class_mode'],
                suffixes=('_nosmooth', '_rgb')
            )

            df_comparison['slope_diff'] = (
                df_comparison['slope_deg_nosmooth'] -
                df_comparison['slope_deg_rgb']
            )

            df_comparison['gen_diff_kwh'] = (
                df_comparison['annual_gen_kwh_nosmooth'] -
                df_comparison['annual_gen_kwh_rgb']
            )

            df_comparison['gen_diff_pct'] = (
                df_comparison['gen_diff_kwh'] /
                (df_comparison['annual_gen_kwh_rgb'] + 1e-6)
            ) * 100

            out_csv_cmp = os.path.join(
                dir_results,
                f"comparison_nosmooth_vs_fixed_RID2_mode{SEGMENT_CLASS_MODE}.csv"
            )

            df_comparison.to_csv(out_csv_cmp, index=False)

            print(f"✓ 对比结果: {out_csv_cmp}")
            print(f"  记录数: {len(df_comparison)}")

            f.write("实验模式: 对比模式\n")
            f.write(f"固定坡度: {default_slope}°\n\n")

            f.write("无平滑 vs 固定坡度\n")
            f.write("-" * 70 + "\n")

            if len(df_comparison) > 0:
                f.write(f"总屋顶分割数: {len(df_comparison)}\n")
                f.write(
                    f"无平滑坡度范围: "
                    f"{df_comparison['slope_deg_nosmooth'].min():.0f}° - "
                    f"{df_comparison['slope_deg_nosmooth'].max():.0f}°\n"
                )
                f.write(
                    f"无平滑平均坡度: "
                    f"{df_comparison['slope_deg_nosmooth'].mean():.2f}° ± "
                    f"{df_comparison['slope_deg_nosmooth'].std():.2f}°\n"
                )
                f.write(
                    f"固定坡度总发电量: "
                    f"{df_comparison['annual_gen_kwh_rgb'].sum():,.0f} kWh/年\n"
                )
                f.write(
                    f"无平滑总发电量: "
                    f"{df_comparison['annual_gen_kwh_nosmooth'].sum():,.0f} kWh/年\n"
                )
                f.write(
                    f"发电量差异: "
                    f"{df_comparison['gen_diff_kwh'].sum():,.0f} kWh/年\n"
                )

        else:
            f.write("实验模式: 单一模式，仅运行无平滑方法\n\n")

            if len(df_nosmooth) > 0:
                f.write("无平滑方法结果统计\n")
                f.write("-" * 70 + "\n")
                f.write(f"总屋顶分割数: {len(df_nosmooth)}\n")
                f.write(
                    f"坡度范围: "
                    f"{df_nosmooth['slope_deg'].min():.0f}° - "
                    f"{df_nosmooth['slope_deg'].max():.0f}°\n"
                )
                f.write(
                    f"平均坡度: "
                    f"{df_nosmooth['slope_deg'].mean():.2f}° ± "
                    f"{df_nosmooth['slope_deg'].std():.2f}°\n"
                )
                f.write(f"中位数坡度: {df_nosmooth['slope_deg'].median():.0f}°\n\n")
                f.write(f"总发电量: {df_nosmooth['annual_gen_kwh'].sum():,.0f} kWh/年\n")
                f.write(f"平均发电量: {df_nosmooth['annual_gen_kwh'].mean():,.0f} kWh/年\n")
                f.write(f"总装机容量: {df_nosmooth['peak_power_kw'].sum():,.1f} kW\n")
                f.write(f"总光伏组件数: {df_nosmooth['num_modules'].sum():,.0f} 个\n")
            else:
                f.write("没有有效结果。\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("完成\n")
        f.write("=" * 70 + "\n")

    print(f"✓ 统计摘要: {summary_txt_path}")

    if len(df_nosmooth) > 0:
        print("\n" + "=" * 70)
        print("无平滑方法结果统计")
        print("=" * 70)
        print(f"总屋顶分割数: {len(df_nosmooth)}")
        print(
            f"坡度范围: "
            f"{df_nosmooth['slope_deg'].min():.0f}° - "
            f"{df_nosmooth['slope_deg'].max():.0f}°"
        )
        print(
            f"平均坡度: "
            f"{df_nosmooth['slope_deg'].mean():.2f}° ± "
            f"{df_nosmooth['slope_deg'].std():.2f}°"
        )
        print(f"中位数坡度: {df_nosmooth['slope_deg'].median():.0f}°")
        print(f"总发电量: {df_nosmooth['annual_gen_kwh'].sum():,.0f} kWh/年")
        print(f"平均发电量: {df_nosmooth['annual_gen_kwh'].mean():,.0f} kWh/年")
        print(f"总装机容量: {df_nosmooth['peak_power_kw'].sum():,.1f} kW")
        print(f"总光伏组件数: {df_nosmooth['num_modules'].sum():,.0f} 个")
    else:
        print("警告：没有生成有效结果。请检查 mask 编码、TIF 文件名和路径是否一致。")

    print("\n" + "=" * 70)
    print("完成")
    print("=" * 70)