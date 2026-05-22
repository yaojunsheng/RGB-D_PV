import os
import sys
import time  # 用于API调用重试间隔
import geopandas as gpd
import numpy as np
import pandas as pd
import cv2
import psutil
from numba import jit
import multiprocessing as mp

# matplotlib设置
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()
matplotlib.rcParams['figure.max_open_warning'] = 0

from definitions import epsg_metric_germany
from electricity_generation import pv_electricity_generation
from masks_to_vector import raster_to_vector, segment_simplify_and_add_azimuth
from module_placement import module_placement, create_pv_modules_gdf
from spatial_operations import get_image_gdf_in_directory
from scipy.ndimage import distance_transform_edt
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc

# ============================================================================
# 配置参数（与标准版本保持一致）
# ============================================================================
dir_roof_segment_masks = '/home/yjs/tmp/Region-aware-Distribution-Contrast/RID1_德国/gt_test_roof_segment'
dir_roof_superstructure_masks = '/home/yjs/tmp/Region-aware-Distribution-Contrast/RID1_德国/gt_test_superstructure'
dir_geotifs = '/mnt/yjs/munihei/munich_data/dop20rgb/'
dir_ndsm = '/mnt/yjs/munihei/output_data/ndsm/'
dir_pvgis_cache = './pvgis_cache_munich/'
dir_results = './结果/德国慕尼黑v3/'

PIXEL_SIZE = 0.2
MAX_WORKERS = 64  # 最多进程数
BATCH_SIZE = 1  # 每个任务处理一张图片

# API调用重试配置（修改为无限重试，间隔1秒）
API_RETRY_DELAY = 1  # 重试间隔（秒）

bg_is_0 = True
segment_classes = ['flat', 'W', 'S', 'E', 'N']
superstructure_classes = ['unknown', 'pvmodule']
a_min_segments = 2
a_min_superstructures = 0.5
pv_module_peak_power = 400
pv_module_height = 1.7
pv_module_width = 1
default_slope = 30  # 与标准版本保持一致

# ============================================================================
# 工具函数（与标准版本处理逻辑保持一致）
# ============================================================================

# 保留Numba加速但确保计算逻辑与标准版本一致
@jit(nopython=True)
def fast_gradient_magnitude(ndsm_data, pixel_size):
    """使用Numba优化的梯度计算，确保结果与标准版本的np.gradient一致"""
    h, w = ndsm_data.shape
    grad_mag = np.zeros((h, w), dtype=np.float64)
    
    # 使用与标准版本相同的梯度计算方式
    for i in range(h):
        for j in range(w):
            # 边界处理与标准版本保持一致
            if i == 0:
                dy = (ndsm_data[i+1, j] - ndsm_data[i, j]) / pixel_size
            elif i == h-1:
                dy = (ndsm_data[i, j] - ndsm_data[i-1, j]) / pixel_size
            else:
                dy = (ndsm_data[i+1, j] - ndsm_data[i-1, j]) / (2 * pixel_size)
                
            if j == 0:
                dx = (ndsm_data[i, j+1] - ndsm_data[i, j]) / pixel_size
            elif j == w-1:
                dx = (ndsm_data[i, j] - ndsm_data[i, j-1]) / pixel_size
            else:
                dx = (ndsm_data[i, j+1] - ndsm_data[i, j-1]) / (2 * pixel_size)
                
            grad_mag[i, j] = np.sqrt(dx*dx + dy*dy)
    
    return np.degrees(np.arctan(grad_mag))

def load_and_validate_files(mask_filename):
    """预检查文件是否存在且有效"""
    mask_id = mask_filename.replace('.tif', '').replace('.png', '')
    
    files = {
        'pred_mask': os.path.join(dir_roof_segment_masks, mask_filename),
        'ndsm': os.path.join(dir_ndsm, mask_id + '.tif'),
        'super_mask': os.path.join(dir_roof_superstructure_masks, mask_filename)
    }
    
    # 检查必需文件
    if not os.path.exists(files['pred_mask']) or not os.path.exists(files['ndsm']):
        return None
    
    return mask_id, files

def remap_segment_mask(mask):
    """重映射屋顶分割掩码（与标准版本保持一致）"""
    remapped = np.zeros_like(mask, dtype=np.uint8)
    remapped[mask == 0] = 0
    remapped[mask == 1] = 5  # N
    remapped[mask == 2] = 4  # E
    remapped[mask == 3] = 3  # S
    remapped[mask == 4] = 2  # W
    remapped[mask == 5] = 1  # flat
    return remapped

def remap_superstructure_mask(mask):
    """重映射附属结构掩码（与标准版本保持一致）"""
    remapped = np.zeros_like(mask, dtype=np.uint8)
    remapped[mask == 0] = 0
    remapped[mask > 0] = 1
    return remapped

# ========== 修改：增加段级容错的坡度提取 ==========
def extract_slopes_optimized(gdf_segments, ndsm_data, segment_mask):
    """优化的坡度提取，增加段级容错"""
    slopes = []
    
    # 尝试计算整体梯度，如果失败则所有segment使用默认值
    try:
        slope_angle = fast_gradient_magnitude(ndsm_data, PIXEL_SIZE)
    except Exception as e:
        print(f"    ⚠ 梯度计算失败，所有坡度使用默认值: {str(e)[:50]}")
        return [default_slope if segment['label'] != 'flat' and not np.isnan(segment.get('azimuth', np.nan)) else 0 
                for _, segment in gdf_segments.iterrows()]
    
    label_to_id = {'flat': 1, 'W': 2, 'S': 3, 'E': 4, 'N': 5}
    failed_count = 0
    
    for idx, segment in gdf_segments.iterrows():
        try:
            # flat或无方位角时设为0
            if segment['label'] == 'flat' or np.isnan(segment.get('azimuth', np.nan)):
                slopes.append(0)
                continue
            
            label_id = label_to_id.get(segment['label'], 0)
            if label_id == 0:
                slopes.append(default_slope)
                continue
                
            # 提取对应区域
            mask = (segment_mask == label_id).astype('uint8')
            if not mask.any():
                slopes.append(default_slope)
                continue
            
            # 计算距离变换权重
            dist_transform = distance_transform_edt(mask)
            weights = (dist_transform / (dist_transform.max() + 1e-6))
            
            # 提取坡度值
            segment_slopes = slope_angle[mask == 1]
            segment_weights = weights[mask == 1]
            
            # 异常值过滤
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
            failed_count += 1
            slopes.append(default_slope)
    
    if failed_count > 0:
        print(f"    ⚠ {failed_count}个segment坡度计算失败，已使用默认值")
    
    return slopes

# 修改为无限重试机制的API调用函数
def call_pv_electricity_generation_with_retry(location, azimuths, slopes, peak_powers, dir_pvgis_cache):
    """调用光伏发电量计算API并实现无限重试机制，间隔1秒"""
    attempts = 0
    while True:  # 无限循环，直到成功
        try:
            # 尝试调用API
            result = pv_electricity_generation(
                location=location,
                azimuths=azimuths,
                slopes=slopes,
                peak_powers=peak_powers,
                dir_pvgis_cache=dir_pvgis_cache
            )
            # 调用成功，返回结果
            if attempts > 0:
                print(f"  ⚡ API调用成功（重试{attempts}次后）")
            return result
        except Exception as e:
            attempts += 1
            print(f"  ⚡ API调用失败（第{attempts}次尝试）: {str(e)[:100]}")
            print(f"  ⏳ 等待{API_RETRY_DELAY}秒后重试...")
            time.sleep(API_RETRY_DELAY)

# ============================================================================
# 单文件处理函数（进程版本，增加段级容错）
# ============================================================================

def process_single_file_optimized(args):
    """优化的单文件处理函数（用于进程池），增加段级容错"""
    mask_filename, gdf_images_data, gdf_images_crs, file_index, total_files = args
    
    try:
        # 打印当前处理的文件信息
        print(f"\n[进程 {os.getpid()}] 处理文件 {file_index}/{total_files}: {mask_filename}")
        
        # 重建gdf_images并恢复CRS
        gdf_images = gpd.GeoDataFrame.from_dict(gdf_images_data)
        gdf_images.crs = gdf_images_crs
        
        # 预检查文件
        file_info = load_and_validate_files(mask_filename)
        if file_info is None:
            print(f"  ✗ 文件无效或缺失关键文件")
            return [], None
        
        mask_id, files = file_info
        
        # ========== 1. 读取nDSM（保持原始精度）==========
        try:
            ndsm_data = cv2.imread(files['ndsm'], cv2.IMREAD_UNCHANGED)
            if ndsm_data is None:
                print(f"  ✗ 无法读取nDSM文件")
                return [], None
            
            if len(ndsm_data.shape) == 3:
                ndsm_data = ndsm_data[:, :, 0]
            
            # 保持原始数据类型精度
            ndsm_data = ndsm_data.astype(float)
            print(f"  ✓ nDSM: {ndsm_data.shape}, 高度范围[{ndsm_data.min():.1f}, {ndsm_data.max():.1f}]m")
            
        except Exception as e:
            print(f"  ✗ nDSM读取异常: {e}")
            return [], None
        
        # ========== 2. 读取segment mask ==========
        try:
            pred_segment_mask = cv2.imread(files['pred_mask'], 0)
            if pred_segment_mask is None:
                print(f"  ✗ 无法读取分割掩码")
                return [], None
            pred_segment_mask = remap_segment_mask(pred_segment_mask)
            print(f"  ✓ 分割掩码: {pred_segment_mask.shape}")
        except Exception as e:
            print(f"  ✗ 分割掩码读取异常: {e}")
            return [], None
        
        # ========== 3. 检查图像ID ==========
        if mask_id not in gdf_images['id'].values:
            print(f"  ✗ 图像ID不在边界框列表中")
            return [], None
        
        gdf_image = gdf_images[gdf_images['id'] == mask_id]
        if len(gdf_image) == 0:
            print(f"  ✗ 未找到对应的图像边界框")
            return [], None
        
        image_bbox = gdf_image.geometry.iloc[0]
        
        # ========== 4. 矢量化（与标准版本完全一致）==========
        try:
            gdf_segments = raster_to_vector(pred_segment_mask, mask_id, image_bbox, 
                                            segment_classes, bg_is_0=True)
            gdf_segments = gdf_segments[gdf_segments.geometry.area > a_min_segments]
            if len(gdf_segments) == 0:
                print(f"  ✗ 没有有效的屋顶分割区域")
                return [], None
            
            gdf_segments = gdf_segments.reset_index(drop=True)
            gdf_segments.crs = gdf_images.crs
            print(f"  ✓ 分割区域: {len(gdf_segments)}个")
            
        except Exception as e:
            print(f"  ✗ 矢量化异常: {e}")
            return [], None
        
        # ========== 5. 计算方位角（增加段级容错）==========
        try:
            gdf_segments_orig = gdf_segments.copy()
            gdf_segments, _, _ = segment_simplify_and_add_azimuth(gdf_segments, visualize=False)
            
            # 检查是否有segment在方位角计算中丢失
            if len(gdf_segments) < len(gdf_segments_orig):
                print(f"    ⚠ 方位角计算导致 {len(gdf_segments_orig) - len(gdf_segments)} 个segment丢失")
            
            # 如果方位角计算导致所有segment丢失，使用原始数据并设置默认方位角
            if len(gdf_segments) == 0:
                print(f"    ⚠ 所有segment方位角计算失败，使用默认方位角")
                gdf_segments = gdf_segments_orig
                gdf_segments['azimuth'] = [np.nan if seg['label'] == 'flat' else 0 
                                           for _, seg in gdf_segments.iterrows()]
            
            print(f"  ✓ 方位角计算完成（有效segment: {len(gdf_segments)}个）")
            
        except Exception as e:
            print(f"  ⚠ 方位角计算异常，使用默认方位角: {str(e)[:50]}")
            # 即使方位角计算失败，也继续处理，使用默认值
            gdf_segments['azimuth'] = [np.nan if seg['label'] == 'flat' else 0 
                                       for _, seg in gdf_segments.iterrows()]
        
        # 再次检查是否还有有效segment
        if len(gdf_segments) == 0:
            print(f"  ✗ 方位角计算后没有有效的屋顶分割区域")
            return [], None
        
        # ========== 6. 读取superstructure（与标准版本一致）==========
        gdf_superstructures = gpd.GeoDataFrame()
        if os.path.exists(files['super_mask']):
            try:
                super_mask = cv2.imread(files['super_mask'], 0)
                if super_mask is not None:
                    super_mask = remap_superstructure_mask(super_mask)
                    gdf_superstructures = raster_to_vector(super_mask, mask_id, image_bbox, 
                                                           superstructure_classes, bg_is_0=True)
                    gdf_superstructures = gdf_superstructures[
                        gdf_superstructures.geometry.area > a_min_superstructures
                    ].reset_index(drop=True)
                    gdf_superstructures.crs = gdf_images.crs
                    print(f"  ✓ 附属结构: {len(gdf_superstructures)}个")
            except:
                gdf_superstructures = gpd.GeoDataFrame()
                print(f"  ⚠ 附属结构读取失败，使用空数据")
        
        # ========== 7. 提取坡度（使用增强的容错函数）==========
        try:
            gdf_segments["slopes"] = extract_slopes_optimized(
                gdf_segments, ndsm_data, pred_segment_mask
            )
            slopes_summary = gdf_segments["slopes"].describe()
            print(f"  ✓ 坡度提取: 范围[{slopes_summary['min']:.0f}, {slopes_summary['max']:.0f}]°, "
                  f"平均{slopes_summary['mean']:.1f}°")
        except Exception as e:
            print(f"  ⚠ 坡度提取异常，使用默认值: {str(e)[:50]}")
            # 即使失败也继续，使用默认坡度
            gdf_segments["slopes"] = [default_slope if seg['label'] != 'flat' else 0 
                                      for _, seg in gdf_segments.iterrows()]
        
        # ========== 8. 光伏模块布置（增加段级容错）==========
        try:
            target_crs = gdf_images.crs if gdf_images.crs is not None else f'EPSG:{epsg_metric_germany}'
            
            if gdf_segments.crs is None:
                gdf_segments = gdf_segments.set_crs(target_crs)
            if len(gdf_superstructures) > 0 and gdf_superstructures.crs is None:
                gdf_superstructures = gdf_superstructures.set_crs(target_crs)
            
            alignment, gdf_modules_v, gdf_modules_h, azimuth = module_placement(
                gdf_segments, 
                gdf_segments["azimuth"], 
                gdf_segments["slopes"],
                gdf_superstructures, 
                pv_module_height, 
                pv_module_width
            )
            
            gdf_modules = create_pv_modules_gdf(alignment, gdf_modules_v, 
                                                gdf_modules_h, azimuth)
            
            if gdf_modules.crs is None:
                gdf_modules = gdf_modules.set_crs(target_crs)
            gdf_modules = gdf_modules.to_crs(epsg_metric_germany)
            
            gdf_segments["pv_modules_per_segment"] = [
                len(mp.geometry.geoms) for mp in gdf_modules.iloc
            ]
            gdf_segments["pv_peak_power_per_segment"] = [
                len(mp.geometry.geoms) * pv_module_peak_power / 1000 
                for mp in gdf_modules.iloc
            ]
            gdf_segments["azimuth_incl_flat"] = azimuth
            
            total_modules = gdf_segments["pv_modules_per_segment"].sum()
            total_power = gdf_segments["pv_peak_power_per_segment"].sum()
            print(f"  ✓ 光伏布置: {total_modules}个模块, {total_power:.1f}kW")
            
        except Exception as e:
            print(f"  ⚠ 光伏布置异常，部分segment可能失败: {str(e)[:100]}")
            # 即使失败也尝试继续，给失败的segment设置0模块
            if 'gdf_modules' not in locals() or len(gdf_modules) != len(gdf_segments):
                print(f"    ⚠ 使用空模块数据")
                gdf_segments["pv_modules_per_segment"] = 0
                gdf_segments["pv_peak_power_per_segment"] = 0.0
                gdf_segments["azimuth_incl_flat"] = gdf_segments["azimuth"]
        
        # ========== 9. 发电量计算（修复unary_union拓扑错误）==========
        print(f"  ⚡ 计算发电量（segment级独立计算）...")
        electricity_generations_list = []
        failed_segments = []
        successful_count = 0
        
        # ===== 修复：使用图像边界框中心，避免unary_union的拓扑错误 =====
        try:
            # 方法1：直接使用图像边界框中心（最稳定）
            gs_location = gpd.GeoSeries([image_bbox.centroid])
            if gs_location.crs is None:
                gs_location = gs_location.set_crs(gdf_images.crs)
            print(f"    ✓ 使用图像边界框中心作为location")
        except Exception as e:
            # 方法2：备用方案 - 使用第一个segment的中心
            print(f"    ⚠ 图像边界框方法失败: {str(e)[:50]}")
            try:
                gs_location = gpd.GeoSeries([gdf_segments.iloc[0].geometry.centroid])
                if gs_location.crs is None:
                    gs_location = gs_location.set_crs(target_crs)
                print(f"    ✓ 使用第一个segment中心作为location")
            except Exception as e2:
                print(f"    ✗ 无法创建location，跳过发电量计算: {str(e2)[:50]}")
                gdf_segments["electricity_generations"] = 0.0
                electricity_generations_list = [0.0] * len(gdf_segments)
                gs_location = None
        # ===== 修复结束 =====
        
        # 只有成功创建location才计算发电量
        if gs_location is not None:
            for segment_idx in range(len(gdf_segments)):
                segment = gdf_segments.iloc[segment_idx]
                
                if segment.get("pv_peak_power_per_segment", 0) <= 0:
                    electricity_generations_list.append(0.0)
                    continue
                
                try:
                    electricity_gen = call_pv_electricity_generation_with_retry(
                        location=gs_location,
                        azimuths=[segment["azimuth_incl_flat"]],
                        slopes=[segment["slopes"]],
                        peak_powers=[segment["pv_peak_power_per_segment"]],
                        dir_pvgis_cache=dir_pvgis_cache
                    )
                    
                    annual_gen = np.sum(electricity_gen[0])
                    electricity_generations_list.append(annual_gen)
                    successful_count += 1
                    
                except Exception as e:
                    failed_segments.append(segment_idx)
                    print(f"    ⚠ segment {segment_idx} ({segment['label']}) 发电量计算失败，使用0值: {str(e)[:50]}")
                    electricity_generations_list.append(0.0)
            
            gdf_segments["electricity_generations"] = electricity_generations_list
            
            with_modules = (gdf_segments["pv_peak_power_per_segment"] > 0).sum()
            
            if len(failed_segments) > 0:
                print(f"    ⚠ {len(failed_segments)}/{with_modules}个有效segment使用0值")
                print(f"    ✓ {successful_count}/{with_modules}个segment精确计算成功")
            else:
                print(f"    ✓ 所有{successful_count}个有效segment计算成功")
            
            total_generation = gdf_segments["electricity_generations"].sum()
            print(f"  ✓ 年发电量: {total_generation:,.0f} kWh")
        else:
            total_generation = 0.0
            print(f"  ✗ 无法计算发电量，所有segment使用0值")
        
        # ========== 10. 为GeoJSON模块添加完整属性（增加段级容错）==========
        module_geometries = []
        module_azimuths = []
        mask_id_list = []
        segment_id_list = []
        module_annual_gen_list = []
        module_peak_power_list = []
        slope_list = []
        roof_type_list = []
        efficiency_list = []
        
        valid_modules_count = 0
        failed_segments = []
        
        for segment_idx in range(len(gdf_segments)):
            try:
                segment = gdf_segments.iloc[segment_idx]
                num_modules = segment["pv_modules_per_segment"]
                
                # 只处理有模块的segment
                if num_modules > 0:
                    # 检查gdf_modules是否存在且有效
                    if 'gdf_modules' not in locals() or len(gdf_modules) <= segment_idx:
                        failed_segments.append(segment_idx)
                        continue
                    
                    segment_modules = gdf_modules.iloc[segment_idx]
                    segment_azimuth = segment["azimuth_incl_flat"]
                    
                    module_annual_gen = segment["electricity_generations"] / num_modules
                    module_peak_power = segment["pv_peak_power_per_segment"] / num_modules
                    module_efficiency = (module_annual_gen / (module_peak_power * 8760)) if module_peak_power > 0 else 0
                    
                    for module_geom in segment_modules.geometry.geoms:
                        module_geometries.append(module_geom)
                        module_azimuths.append(segment_azimuth)
                        mask_id_list.append(mask_id)
                        segment_id_list.append(segment_idx)
                        module_annual_gen_list.append(module_annual_gen)
                        module_peak_power_list.append(module_peak_power)
                        slope_list.append(segment["slopes"])
                        roof_type_list.append(segment["label"])
                        efficiency_list.append(module_efficiency)
                        valid_modules_count += 1
            
            except Exception as e:
                failed_segments.append(segment_idx)
                print(f"    ⚠ segment {segment_idx} 模块属性计算失败: {str(e)[:50]}")
                continue
        
        if failed_segments:
            print(f"    ⚠ {len(failed_segments)} 个segment模块属性计算失败，已跳过")
        
        # 创建GeoDataFrame（即使没有模块也创建空的）
        if valid_modules_count > 0:
            gdf_modules_complete = gpd.GeoDataFrame({
                'geometry': module_geometries,
                'azimuth': module_azimuths,
                'mask_id': mask_id_list,
                'segment_id': segment_id_list,
                'module_annual_gen': module_annual_gen_list,
                'module_peak_power': module_peak_power_list,
                'slope_deg': slope_list,
                'roof_type': roof_type_list,
                'efficiency': efficiency_list
            })
            
            gdf_modules_complete.crs = gdf_modules.crs if 'gdf_modules' in locals() else target_crs
            
            print(f"  ✓ 有效模块属性计算完成: {valid_modules_count}个模块")
            if valid_modules_count > 0:
                print(f"    - 平均单模块发电量: {np.mean(module_annual_gen_list):.1f} kWh")
                print(f"    - 平均效率: {np.mean(efficiency_list)*100:.1f}%")
        else:
            gdf_modules_complete = gpd.GeoDataFrame(columns=[
                'geometry', 'azimuth', 'mask_id', 'segment_id', 
                'module_annual_gen', 'module_peak_power', 'slope_deg', 
                'roof_type', 'efficiency'
            ])
            gdf_modules_complete.crs = target_crs
            print(f"  ⚠ 该图像没有有效的太阳能板模块")
        
        # ========== 11. 收集CSV结果（增加段级容错）==========
        file_results = []
        for i in range(len(gdf_segments)):
            try:
                segment = gdf_segments.iloc[i]
                file_results.append({
                    'mask_id': mask_id,
                    'segment_id': i,
                    'label': segment['label'],
                    'area_m2': segment.geometry.area,
                    'azimuth': segment.get('azimuth_incl_flat', segment.get('azimuth', np.nan)),
                    'slope_deg': segment.get('slopes', default_slope),
                    'num_modules': segment.get('pv_modules_per_segment', 0),
                    'peak_power_kw': segment.get('pv_peak_power_per_segment', 0.0),
                    'annual_gen_kwh': segment.get('electricity_generations', 0.0)
                })
            except Exception as e:
                print(f"    ⚠ segment {i} CSV结果收集失败: {str(e)[:50]}")
                # 添加一个带默认值的记录
                file_results.append({
                    'mask_id': mask_id,
                    'segment_id': i,
                    'label': 'unknown',
                    'area_m2': 0.0,
                    'azimuth': np.nan,
                    'slope_deg': 0,
                    'num_modules': 0,
                    'peak_power_kw': 0.0,
                    'annual_gen_kwh': 0.0
                })
        
        print(f"  ✓ 完成! CSV导出{len(file_results)}条记录, GeoJSON包含{valid_modules_count}个有效模块")
        
        # ========== 12. 转换完整的gdf_modules为可序列化格式 ==========
        if valid_modules_count > 0:
            modules_dict = gdf_modules_complete.to_dict('list')
        else:
            modules_dict = None
        
        # 清理内存
        del ndsm_data, pred_segment_mask, gdf_segments
        if 'gdf_modules' in locals():
            del gdf_modules
        if 'gdf_modules_complete' in locals():
            del gdf_modules_complete
        gc.collect()
        
        return file_results, modules_dict
        
    except Exception as e:
        print(f"  ✗ 整体处理失败: {str(e)[:100]}")
        import traceback
        print(f"  错误详情:\n{traceback.format_exc()}")
        return [], None

# ============================================================================
# 主程序（与标准版本保持一致）
# ============================================================================

if __name__ == "__main__":
    # 环境设置
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    # 创建目录
    for dir_path in [dir_pvgis_cache, dir_results]:
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
    
    # 获取系统信息
    memory = psutil.virtual_memory()
    cpu_count = psutil.cpu_count()
    
    print(f"\n{'='*70}")
    print("慕尼黑光伏潜力评估（健壮版本 - 段级容错）")
    print(f"{'='*70}")
    print(f"系统信息:")
    print(f"  CPU核心数: {cpu_count}")
    print(f"  总内存: {memory.total / (1024**3):.1f} GB")
    print(f"  可用内存: {memory.available / (1024**3):.1f} GB")
    print(f"\n参数配置:")
    print(f"  进程数: {MAX_WORKERS}")
    print(f"  每任务处理: 1张图片")
    print(f"  像素分辨率: {PIXEL_SIZE}m")
    print(f"  API调用策略: 无限重试，间隔{API_RETRY_DELAY}秒")
    print(f"  容错策略: 段级容错，只跳过失败的segment")
    print(f"{'='*70}\n")
    
    # 加载图像边界框
    print("加载图像边界框...")
    gdf_images = get_image_gdf_in_directory(dir_geotifs)
    target_crs_string = f'EPSG:{epsg_metric_germany}'
    
    if gdf_images.crs is None:
        gdf_images = gdf_images.set_crs(epsg=epsg_metric_germany)
    else:
        gdf_images = gdf_images.to_crs(target_crs_string)
    
    gdf_images_crs = gdf_images.crs
    gdf_images_data = gdf_images.to_dict()
    
    print(f"找到 {len(gdf_images)} 个图像边界框，CRS: {gdf_images_crs}\n")
    
    # 获取文件列表并预筛选
    print("预筛选文件...")
    mask_filenames = [f for f in os.listdir(dir_roof_segment_masks) 
                      if f.endswith(('.png', '.tif', '.tiff'))]
    
    # 预检查，过滤掉无效文件
    valid_files = []
    for filename in mask_filenames:
        if load_and_validate_files(filename) is not None:
            valid_files.append(filename)
    
    print(f"原始文件数: {len(mask_filenames)}")
    print(f"有效文件数: {len(valid_files)}")
    
    start_time = datetime.now()
    all_results = []
    all_modules_dicts = []
    processed_count = 0
    
    print("="*70)
    print(f"开始并行处理，使用 {MAX_WORKERS} 个进程")
    print("="*70)
    
    # 使用进程池处理
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {}
        for i, filename in enumerate(valid_files):
            future = executor.submit(
                process_single_file_optimized, 
                (filename, gdf_images_data, gdf_images_crs, i+1, len(valid_files))
            )
            futures[future] = filename
        
        # 处理完成的任务
        for future in as_completed(futures):
            filename = futures[future]
            try:
                file_results, modules_dict = future.result()
                
                # 即使某些segment失败，只要有结果就收集
                if len(file_results) > 0:
                    all_results.extend(file_results)
                
                if modules_dict is not None:
                    all_modules_dicts.append(modules_dict)
                
                processed_count += 1
                
                # 报告进度
                elapsed = (datetime.now() - start_time).total_seconds()
                progress = processed_count / len(valid_files) * 100
                speed = processed_count / elapsed if elapsed > 0 else 0
                remaining = (len(valid_files) - processed_count) / speed if speed > 0 else 0
                
                memory = psutil.virtual_memory()
                print(f"\n进度: {processed_count}/{len(valid_files)} ({progress:.1f}%) | "
                      f"速度: {speed:.2f} 张/秒 | "
                      f"剩余: {remaining/60:.1f}分钟 | "
                      f"内存: {memory.percent:.1f}% | "
                      f"已收集: {len(all_results)} 条记录")
                
            except Exception as e:
                print(f"\n✗ {filename} 处理异常: {str(e)[:50]}")
                processed_count += 1
    
    # 保存结果
    if len(all_results) == 0:
        print("\n✗ 没有成功处理任何文件")
        sys.exit(1)
    
    df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(dir_results, f"munich_pv_potential_robust_{timestamp}.csv")
    df.to_csv(output_csv, index=False)
    
    # 合并并保存geojson
    if len(all_modules_dicts) > 0:
        print(f"\n正在合并 {len(all_modules_dicts)} 个模块文件...")
        
        # 合并所有字典
        combined_dict = {key: [] for key in all_modules_dicts[0].keys()}
        for modules_dict in all_modules_dicts:
            for key in combined_dict.keys():
                combined_dict[key].extend(modules_dict[key])
        
        # 重建GeoDataFrame
        gdf_modules_all = gpd.GeoDataFrame.from_dict(combined_dict)
        
        # 保存geojson
        geojson_path = os.path.join(dir_results, f"gdf_modules_munich_robust_{timestamp}.json")
        gdf_modules_all.to_file(geojson_path, driver="GeoJSON")
        
        print(f"✓ geojson保存成功: {os.path.basename(geojson_path)}")
        print(f"  总模块数: {len(gdf_modules_all)}")
    else:
        print(f"\n⚠ 没有有效的模块数据")
    
    elapsed_time = (datetime.now() - start_time).total_seconds() / 60
    
    # 统计报告
    print(f"\n{'='*70}")
    print("处理完成 - 健壮版本统计")
    print(f"{'='*70}")
    
    print(f"\n[处理结果]")
    print(f"  总记录数: {len(df)}")
    print(f"  处理图片数: {df['mask_id'].nunique()}")
    print(f"  总耗时: {elapsed_time:.1f} 分钟")
    print(f"  平均速度: {len(valid_files)/elapsed_time:.2f} 张/分钟")
    
    # 坡度统计
    slopes = df['slope_deg'].values
    print(f"\n[坡度分布]")
    print(f"  平屋顶(0°): {(slopes==0).sum()}")
    print(f"  低坡度(1-15°): {((slopes>0) & (slopes<=15)).sum()}")
    print(f"  中坡度(16-30°): {((slopes>15) & (slopes<=30)).sum()}")
    print(f"  高坡度(>30°): {(slopes>30).sum()}")
    
    if (slopes > 0).sum() > 0:
        print(f"  非零坡度均值: {slopes[slopes>0].mean():.1f}°")
    
    # 光伏统计
    total_power = df['peak_power_kw'].sum()
    total_generation = df['annual_gen_kwh'].sum()
    print(f"\n[光伏潜力]")
    print(f"  总装机: {total_power:,.1f} kW")
    print(f"  年发电量: {total_generation:,.0f} kWh")
    if total_power > 0:
        print(f"  容量因子: {(total_generation/(total_power*8760))*100:.1f}%")
    
    print(f"\n[输出文件]")
    print(f"  CSV: {output_csv}")
    if len(all_modules_dicts) > 0:
        print(f"  GeoJSON: {geojson_path}")
    print(f"{'='*70}\n")
