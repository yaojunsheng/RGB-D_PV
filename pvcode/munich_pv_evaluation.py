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
dir_results = './结果/德国慕尼黑/'

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

# 修改坡度提取函数，使其与标准版本的extract_slopes_no_smoothing逻辑一致
def extract_slopes_optimized(gdf_segments, ndsm_data, segment_mask):
    """优化的坡度提取，与标准版本处理逻辑保持一致"""
    slopes = []
    
    # 使用与标准版本相同的梯度计算方式
    slope_angle = fast_gradient_magnitude(ndsm_data, PIXEL_SIZE)
    
    label_to_id = {'flat': 1, 'W': 2, 'S': 3, 'E': 4, 'N': 5}
    
    for idx, segment in gdf_segments.iterrows():
        try:
            # flat或无方位角时设为0（与标准版本一致）
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
            
            # 计算距离变换权重（与标准版本一致）
            dist_transform = distance_transform_edt(mask)
            weights = (dist_transform / (dist_transform.max() + 1e-6))
            
            # 提取坡度值（与标准版本一致）
            segment_slopes = slope_angle[mask == 1]
            segment_weights = weights[mask == 1]
            
            # 异常值过滤（与标准版本一致）
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
# 单文件处理函数（进程版本，与标准版本处理流程一致）
# ============================================================================

def process_single_file_optimized(args):
    """优化的单文件处理函数（用于进程池）"""
    mask_filename, gdf_images_data, file_index, total_files = args
    
    try:
        # 打印当前处理的文件信息
        print(f"\n[进程 {os.getpid()}] 处理文件 {file_index}/{total_files}: {mask_filename}")
        
        # 重建gdf_images（进程间不能直接传递GeoDataFrame）
        gdf_images = gpd.GeoDataFrame.from_dict(gdf_images_data)
        
        # 预检查文件
        file_info = load_and_validate_files(mask_filename)
        if file_info is None:
            print(f"  ✗ 文件无效或缺失关键文件")
            return [], None  # ========== 修改：返回tuple ==========
        
        mask_id, files = file_info
        
        # ========== 1. 读取nDSM（保持原始精度）==========
        try:
            ndsm_data = cv2.imread(files['ndsm'], cv2.IMREAD_UNCHANGED)
            if ndsm_data is None:
                print(f"  ✗ 无法读取nDSM文件")
                return [], None  # ========== 修改：返回tuple ==========
            
            if len(ndsm_data.shape) == 3:
                ndsm_data = ndsm_data[:, :, 0]
            
            # 保持原始数据类型精度
            ndsm_data = ndsm_data.astype(float)
            print(f"  ✓ nDSM: {ndsm_data.shape}, 高度范围[{ndsm_data.min():.1f}, {ndsm_data.max():.1f}]m")
            
        except Exception as e:
            print(f"  ✗ nDSM读取异常: {e}")
            return [], None  # ========== 修改：返回tuple ==========
        
        # ========== 2. 读取segment mask ==========
        try:
            pred_segment_mask = cv2.imread(files['pred_mask'], 0)
            if pred_segment_mask is None:
                print(f"  ✗ 无法读取分割掩码")
                return [], None  # ========== 修改：返回tuple ==========
            pred_segment_mask = remap_segment_mask(pred_segment_mask)
            print(f"  ✓ 分割掩码: {pred_segment_mask.shape}")
        except Exception as e:
            print(f"  ✗ 分割掩码读取异常: {e}")
            return [], None  # ========== 修改：返回tuple ==========
        
        # ========== 3. 检查图像ID ==========
        if mask_id not in gdf_images['id'].values:
            print(f"  ✗ 图像ID不在边界框列表中")
            return [], None  # ========== 修改：返回tuple ==========
        
        gdf_image = gdf_images[gdf_images['id'] == mask_id]
        if len(gdf_image) == 0:
            print(f"  ✗ 未找到对应的图像边界框")
            return [], None  # ========== 修改：返回tuple ==========
        
        image_bbox = gdf_image.geometry.iloc[0]
        
        # ========== 4. 矢量化（与标准版本处理一致）==========
        try:
            gdf_segments = raster_to_vector(pred_segment_mask, mask_id, image_bbox, 
                                            segment_classes, bg_is_0=True)
            gdf_segments = gdf_segments[gdf_segments.geometry.area > a_min_segments]
            if len(gdf_segments) == 0:
                print(f"  ✗ 没有有效的屋顶分割区域")
                return [], None  # ========== 修改：返回tuple ==========
            
            gdf_segments = gdf_segments.reset_index(drop=True)
            gdf_segments.crs = gdf_images.crs
            print(f"  ✓ 分割区域: {len(gdf_segments)}个")
            
        except Exception as e:
            print(f"  ✗ 矢量化异常: {e}")
            return [], None  # ========== 修改：返回tuple ==========
        
        # ========== 5. 计算方位角（与标准版本处理一致）==========
        try:
            gdf_segments, _, _ = segment_simplify_and_add_azimuth(gdf_segments, visualize=False)
            print(f"  ✓ 方位角计算完成")
        except Exception as e:
            print(f"  ✗ 方位角计算异常: {e}")
            return [], None  # ========== 修改：返回tuple ==========
        
        # ========== 6. 读取superstructure（如果存在）==========
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
        
        # ========== 7. 提取坡度（与标准版本处理一致）==========
        try:
            gdf_segments["slopes"] = extract_slopes_optimized(
                gdf_segments, ndsm_data, pred_segment_mask
            )
            slopes_summary = gdf_segments["slopes"].describe()
            print(f"  ✓ 坡度提取: 范围[{slopes_summary['min']:.0f}, {slopes_summary['max']:.0f}]°, "
                  f"平均{slopes_summary['mean']:.1f}°")
        except Exception as e:
            print(f"  ✗ 坡度提取异常: {e}")
            return [], None  # ========== 修改：返回tuple ==========
        
        # ========== 8. 光伏模块布置（与标准版本处理一致）==========
        try:
            # 确保gdf_segments和gdf_superstructures都有正确的CRS
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
            
            # 确保设置CRS后再转换
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
            print(f"  ✗ 光伏布置异常: {e}")
            return [], None  # ========== 修改：返回tuple ==========
        
        # ========== 9. 发电量计算（使用无限重试机制）==========
        try:
            gs_location = gpd.GeoSeries(gdf_segments.unary_union.centroid)
            # 确保location有正确的CRS设置
            if gs_location.crs is None:
                gs_location = gs_location.set_crs(target_crs)
            
            print(f"  ⚡ 计算发电量...")
            # 使用带无限重试机制的API调用函数
            electricity_generations = call_pv_electricity_generation_with_retry(
                location=gs_location,
                azimuths=gdf_segments["azimuth_incl_flat"],
                slopes=gdf_segments["slopes"],
                peak_powers=gdf_segments["pv_peak_power_per_segment"],
                dir_pvgis_cache=dir_pvgis_cache
            )
            
            gdf_segments["electricity_generations"] = [
                np.sum(e) for e in electricity_generations
            ]
            
            total_generation = gdf_segments["electricity_generations"].sum()
            print(f"  ✓ 年发电量: {total_generation:,.0f} kWh")
            
        except Exception as e:
            print(f"  ✗ 发电量计算异常: {e}")
            return [], None  # ========== 修改：返回tuple ==========
        
        # ========== 10. 收集结果 ==========
        file_results = []
        for i in range(len(gdf_segments)):
            file_results.append({
                'mask_id': mask_id,
                'segment_id': i,
                'label': gdf_segments.iloc[i]['label'],
                'area_m2': gdf_segments.iloc[i].geometry.area,
                'azimuth': gdf_segments.iloc[i]['azimuth_incl_flat'],
                'slope_deg': gdf_segments.iloc[i]['slopes'],
                'num_modules': gdf_segments.iloc[i]['pv_modules_per_segment'],
                'peak_power_kw': gdf_segments.iloc[i]['pv_peak_power_per_segment'],
                'annual_gen_kwh': gdf_segments.iloc[i]['electricity_generations']
            })
        
        print(f"  ✓ 完成! 导出{len(file_results)}条记录")
        
        # ========== 新增：转换gdf_modules为可序列化格式 ==========
        modules_dict = gdf_modules.to_dict('list')
        
        # 清理内存
        del ndsm_data, pred_segment_mask, gdf_segments
        # ========== 修改：不删除gdf_modules，已转为dict ==========
        gc.collect()
        
        return file_results, modules_dict  # ========== 修改：返回tuple ==========
        
    except Exception as e:
        print(f"  ✗ 处理失败: {str(e)[:100]}")
        return [], None  # ========== 修改：返回tuple ==========

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
    print("慕尼黑光伏潜力评估（与标准版本一致）")
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
    print(f"{'='*70}\n")
    
    # 加载图像边界框
    print("加载图像边界框...")
    gdf_images = get_image_gdf_in_directory(dir_geotifs)
    target_crs_string = f'EPSG:{epsg_metric_germany}'
    
    if gdf_images.crs is None:
        gdf_images = gdf_images.set_crs(epsg=epsg_metric_germany)
    else:
        gdf_images = gdf_images.to_crs(target_crs_string)
    
    # 转换为字典以便在进程间传递
    gdf_images_data = gdf_images.to_dict()
    
    print(f"找到 {len(gdf_images)} 个图像边界框\n")
    
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
    # ========== 新增：收集所有gdf_modules ==========
    all_modules_dicts = []
    processed_count = 0
    
    print("="*70)
    print(f"开始并行处理，使用 {MAX_WORKERS} 个进程")
    print("="*70)
    
    # 使用进程池处理
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # 提交所有任务
        futures = {}
        for i, filename in enumerate(valid_files):
            future = executor.submit(
                process_single_file_optimized, 
                (filename, gdf_images_data, i+1, len(valid_files))
            )
            futures[future] = filename
        
        # 处理完成的任务
        for future in as_completed(futures):
            filename = futures[future]
            try:
                # ========== 修改：接收两个返回值 ==========
                file_results, modules_dict = future.result()
                all_results.extend(file_results)
                # ========== 新增：收集modules_dict ==========
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
    output_csv = os.path.join(dir_results, f"munich_pv_potential_standard_{timestamp}.csv")
    df.to_csv(output_csv, index=False)
    
    # ========== 新增：合并并保存geojson ==========
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
        geojson_path = os.path.join(dir_results, f"gdf_modules_munich_{timestamp}.json")
        gdf_modules_all.to_file(geojson_path, driver="GeoJSON")
        
        print(f"✓ geojson保存成功: {os.path.basename(geojson_path)}")
        print(f"  总模块数: {len(gdf_modules_all)}")
    
    elapsed_time = (datetime.now() - start_time).total_seconds() / 60
    
    # 统计报告
    print(f"\n{'='*70}")
    print("处理完成 - 标准版本统计")
    print(f"{'='*70}")
    
    print(f"\n[处理结果]")
    print(f"  总记录数: {len(df)}")
    print(f"  处理图片数: {df['mask_id'].nunique()}")
    print(f"  总耗时: {elapsed_time:.1f} 分钟")
    print(f"  平均速度: {len(valid_files)/elapsed_time:.2f} 张/分钟")
    
    # 坡度统计（与标准版本输出格式一致）
    slopes = df['slope_deg'].values
    print(f"\n[坡度分布]")
    print(f"  平屋顶(0°): {(slopes==0).sum()}")
    print(f"  低坡度(1-15°): {((slopes>0) & (slopes<=15)).sum()}")
    print(f"  中坡度(16-30°): {((slopes>15) & (slopes<=30)).sum()}")
    print(f"  高坡度(>30°): {(slopes>30).sum()}")
    
    if (slopes > 0).sum() > 0:
        print(f"  非零坡度均值: {slopes[slopes>0].mean():.1f}°")
    
    # 光伏统计（与标准版本输出格式一致）
    total_power = df['peak_power_kw'].sum()
    total_generation = df['annual_gen_kwh'].sum()
    print(f"\n[光伏潜力]")
    print(f"  总装机: {total_power:,.1f} kW")
    print(f"  年发电量: {total_generation:,.0f} kWh")
    print(f"  容量因子: {(total_generation/(total_power*8760))*100:.1f}%")
    
    print(f"\n[输出文件]: {output_csv}")
    print(f"{'='*70}\n")