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
    year of weather data, default is 2014.
    """

    latitude = round(latitude, 2)
    longitude = round(longitude, 2)

    loss = int(round(loss, 0))
    azimuth = int(round(azimuth, 0))
    slope = int(round(slope, 0))
    peak_power = round(peak_power, 2)

    PVGIS_config = (
        f"lat={latitude: .8f}&lon={longitude: .8f}&pvcalculation=1&peakpower={peak_power}&angle={slope}"
        f"&aspect={azimuth}&startyear={year}&endyear={year}&mountingplace=building&outputformat=json"
        f"&loss={loss}"
    )

    pvgis_cache_filepath = os.path.join(dir_pvgis_cache, PVGIS_config + str(".json"))
    pvgis_result = None

    if os.path.isfile(pvgis_cache_filepath):
        try:
            with open(pvgis_cache_filepath, 'rb') as file:
                pvgis_result = json.load(file)

            if 'outputs' not in pvgis_result or 'hourly' not in pvgis_result['outputs']:
                raise ValueError("缓存文件缺少必要字段")

            if not pvgis_result['outputs']['hourly']:
                raise ValueError("缓存文件hourly数据为空")

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            print(f"    警告：缓存文件损坏 ({type(e).__name__}: {e})，删除并重新请求...")
            try:
                os.remove(pvgis_cache_filepath)
            except Exception:
                pass
            pvgis_result = None

    if pvgis_result is None:
        query = ("https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?{}".format(PVGIS_config))

        attempt = 0
        while True:
            attempt += 1
            try:
                if attempt > 1:
                    if (attempt - 1) % 3 == 0:
                        print(f"    [已失败{attempt - 1}次] 等待60秒后继续尝试...")
                        time.sleep(60)
                    else:
                        print(f"    等待5秒后重试...")
                        time.sleep(5)

                print(f"    [API请求 第{attempt}次] lat={latitude:.2f}, lon={longitude:.2f}, az={azimuth}°, slope={slope}°")

                response = requests.get(query, timeout=30)

                if response.status_code != 200:
                    raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")

                if not response.text or len(response.text) < 10:
                    raise Exception("API返回空响应")

                try:
                    pvgis_result = response.json()
                except json.JSONDecodeError:
                    raise Exception(f"JSON解析失败: {response.text[:200]}")

                if 'outputs' not in pvgis_result or 'hourly' not in pvgis_result['outputs']:
                    raise ValueError(f"API返回格式错误: {pvgis_result.get('message', '数据不完整')}")

                if not pvgis_result['outputs']['hourly']:
                    raise ValueError("hourly数据为空")

                try:
                    with open(pvgis_cache_filepath, 'w') as file:
                        json.dump(pvgis_result, file)
                    print(f"    [成功] 缓存已保存")
                except Exception as e:
                    print(f"    [警告] 无法保存缓存: {e}")

                break

            except Exception as e:
                print(f"    [失败 第{attempt}次] {e}")

    try:
        df = pd.DataFrame(pvgis_result['outputs']['hourly'])
        PV_E_gen_hourly = pd.DataFrame(columns=['time', 'power'])
        PV_E_gen_hourly.time = pd.to_datetime(df.time, format='%Y%m%d:%H%M')
        PV_E_gen_hourly.power = df.P
    except Exception as e:
        print(f"    错误：处理PVGIS数据失败 ({e})")
        raise

    return PV_E_gen_hourly


def pv_electricity_generation(location, azimuths, slopes, peak_powers, dir_pvgis_cache):
    """
    计算多个屋顶段的总发电量
    """
    if location.crs != 4326:
        location = location.to_crs(4326)

    E_gen_hourly_list = []

    total_configs = len(azimuths)
    print(f"\n开始处理 {total_configs} 个屋顶配置...")

    for i, azimuth in enumerate(azimuths):
        print(f"\n配置 {i + 1}/{total_configs}:")

        PV_E_gen_hourly_kWp = pv_electricity_generation_per_kWp_hourly(
            location.geometry.x.iloc[0],
            location.geometry.y.iloc[0],
            azimuth,
            slopes[i],
            peak_power=1,
            loss=pv_system_loss * 100,
            year=default_year,
            dir_pvgis_cache=dir_pvgis_cache
        )

        PV_E_gen_hourly_segment = PV_E_gen_hourly_kWp.power * peak_powers[i] / 1000
        E_gen_hourly_list.append(PV_E_gen_hourly_segment)

    print(f"\n✅ 所有 {total_configs} 个配置处理成功!\n")

    if len(E_gen_hourly_list) == 0:
        print("    [警告] 没有任何配置，使用零值填充")
        PV_E_gen_hourly_kWp = pv_electricity_generation_per_kWp_hourly(42, 11, 0, 30, 1, 14, 2014, dir_pvgis_cache)
        E_gen_hourly_list.append(PV_E_gen_hourly_kWp.power * 0)

    return E_gen_hourly_list