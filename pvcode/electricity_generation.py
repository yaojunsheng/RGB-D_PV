__author__ = "Sebastian Krapf"
__copyright__ = "Copyright 2023, Institute of Automotive Technology TUM"
__credits__ = []
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Sebastian Krapf"
__email__ = "sebastian.krapf@tum.de"
__status__ = "alpha"

import requests
import os
import json
import pandas as pd
import numpy as np
import time

from definitions import pv_system_loss, default_year


def pv_electricity_generation_per_kWp_hourly(longitude, latitude, azimuth, slope, peak_power=1, loss=14, year=2014, dir_pvgis_cache=""):
    """
    This function calls the PVGIS API to determine the hourly energy output of a roof segment for a
    year of weather data, default is 2014, a year with medium yearly solar radiation in bavaria.
    https://ec.europa.eu/jrc/en/PVGIS/docs/noninteractive

    :param longitude: float
        lonigtude of roof segment's location
    :param latitude: float
        latitude of roof segment's location
    :param azimuth: float
        azimuth of roof segment, between -180 (N) and 180 (N), with 0 being South. see PVGIS for definition
    :param slope: float
        roof tilt/slope angle between 0 and 90
    :param peak_power: float
        peak power of the PV system, default: 1 kWp
    :param loss: int
        loss of PV system in percent. default 14
    :param year: int
        year of radiation data. default is 2014

    :return: PV_E_gen_hourly pandas Dataframe
        pandas Dataframe with time and power as columns. Contains the hourly values of one year.
        power is in W. Will keep retrying until success.
    """

    # 【关键优化】量化坐标到0.01度（约1km精度）
    latitude = round(latitude, 2)
    longitude = round(longitude, 2)
    
    # round values integer to avoid weird errors
    loss = int(round(loss, 0))
    azimuth = int(round(azimuth, 0))
    slope = int(round(slope, 0))
    peak_power = round(peak_power, 2)

    # create PVGIS call string
    PVGIS_config = (f"lat={latitude: .8f}&lon={longitude: .8f}&pvcalculation=1&peakpower={peak_power}&angle={slope}"
                    f"&aspect={azimuth}&startyear={year}&endyear={year}&mountingplace=building&outputformat=json"
                    f"&loss={loss}")

    # check if result of a configuration has already been requested and saved before:
    pvgis_cache_filepath = os.path.join(dir_pvgis_cache, PVGIS_config + str(".json"))
    pvgis_result = None
    
    if os.path.isfile(pvgis_cache_filepath):
        try:
            with open(pvgis_cache_filepath, 'rb') as file:
                pvgis_result = json.load(file)
            
            # 验证JSON结构完整性
            if 'outputs' not in pvgis_result or 'hourly' not in pvgis_result['outputs']:
                raise ValueError("缓存文件缺少必要字段")
            
            # 验证数据不为空
            if not pvgis_result['outputs']['hourly']:
                raise ValueError("缓存文件hourly数据为空")
                
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"    警告：缓存文件损坏 ({type(e).__name__}: {e})，删除并重新请求...")
            try:
                os.remove(pvgis_cache_filepath)
            except:
                pass
            pvgis_result = None

    # ✅ 如果没有有效缓存，循环调用API直到成功
    if pvgis_result is None:
        query = ("https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?{}".format(PVGIS_config))
        
        attempt = 0
        while True:  # 无限循环直到成功
            attempt += 1
            try:
                if attempt > 1:
                    # 每3次失败后等待60秒
                    if (attempt - 1) % 3 == 0:
                        print(f"    [已失败{attempt-1}次] 等待60秒后继续尝试...")
                        time.sleep(60)
                    else:
                        # 常规等待5秒
                        print(f"    等待5秒后重试...")
                        time.sleep(5)
                
                print(f"    [API请求 第{attempt}次] lat={latitude:.2f}, lon={longitude:.2f}, az={azimuth}°, slope={slope}°")
                
                # API调用
                response = requests.get(query, timeout=30)
                
                # 检查HTTP状态码
                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")
                
                # 检查响应是否为空
                if not response.text or len(response.text) < 10:
                    raise Exception("API返回空响应")
                
                # 尝试解析JSON
                try:
                    pvgis_result = response.json()
                except json.JSONDecodeError as e:
                    raise Exception(f"JSON解析失败: {response.text[:200]}")
                
                # 验证API返回数据
                if 'outputs' not in pvgis_result or 'hourly' not in pvgis_result['outputs']:
                    raise ValueError(f"API返回格式错误: {pvgis_result.get('message', '数据不完整')}")
                
                if not pvgis_result['outputs']['hourly']:
                    raise ValueError("hourly数据为空")
                
                # 保存缓存
                try:
                    with open(pvgis_cache_filepath, 'w') as file:
                        json.dump(pvgis_result, file)
                    print(f"    [成功] 缓存已保存")
                except Exception as e:
                    print(f"    [警告] 无法保存缓存: {e}")
                
                # ✅ 成功，跳出循环
                break
                    
            except Exception as e:
                print(f"    [失败 第{attempt}次] {e}")
                # 继续下一次循环，不返回

    # 处理返回数据
    try:
        df = pd.DataFrame(pvgis_result['outputs']['hourly'])
        PV_E_gen_hourly = pd.DataFrame(columns=['time', 'power'])
        PV_E_gen_hourly.time = pd.to_datetime(df.time, format='%Y%m%d:%H%M')
        PV_E_gen_hourly.power = df.P
    except Exception as e:
        print(f"    错误：处理PVGIS数据失败 ({e})")
        raise  # 数据处理失败应该抛出异常，因为这不是网络问题

    return PV_E_gen_hourly


def pv_electricity_generation(location, azimuths, slopes, peak_powers, dir_pvgis_cache):
    """
    计算多个屋顶段的总发电量
    """
    # make sure location is in EPSG 4326
    if location.crs != 4326:
        location = location.to_crs(4326)
    
    # initialize result list
    E_gen_hourly_list = []
    
    total_configs = len(azimuths)
    print(f"\n开始处理 {total_configs} 个屋顶配置...")
    
    # request the electricity generation per kWp for each roof segment (azimuth)
    for i, azimuth in enumerate(azimuths):
        print(f"\n配置 {i+1}/{total_configs}:")
        
        # ✅ 现在这个函数会一直重试直到成功，不会返回None
        PV_E_gen_hourly_kWp = pv_electricity_generation_per_kWp_hourly(
            location.geometry.x.iloc[0],
            location.geometry.y.iloc[0],
            azimuth,
            slopes[i],
            peak_power=1,
            loss=pv_system_loss*100,
            year=default_year,
            dir_pvgis_cache=dir_pvgis_cache
        )
        
        # scale generation per kWp to segment's kWp
        PV_E_gen_hourly_segment = PV_E_gen_hourly_kWp.power * peak_powers[i] / 1000
        # electricity generation profile in kW
        E_gen_hourly_list.append(PV_E_gen_hourly_segment)
    
    print(f"\n✅ 所有 {total_configs} 个配置处理成功!\n")
    
    # if no electricity generation: fill e_gen with zeroes
    if len(E_gen_hourly_list) == 0:
        print("    [警告] 没有任何配置，使用零值填充")
        PV_E_gen_hourly_kWp = pv_electricity_generation_per_kWp_hourly(42, 11, 0, 30, 1, 14, 2014, dir_pvgis_cache)
        E_gen_hourly_list.append(PV_E_gen_hourly_kWp.power * 0)
    
    return E_gen_hourly_list