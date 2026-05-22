epsg_default = 4326  #定义默认的坐标参考系（CRS）为 EPSG:4326，即WGS84 坐标系（经纬度坐标系），是全球通用的地理坐标系统
epsg_metric_germany = 28992 # 荷兰坐标
#epsg_metric_germany = 25832 # the epsg in Germany
flat_roof_orientation_mode = 'alignment'  # options: 'south', 'east-west', 'alignment'    #定义平屋顶光伏组件的朝向模式为'alignment'（对齐模式）
flat_roof_space_util = 0.5  # factor 0-1 # own assumption        #定义平屋顶的空间利用率为 0.5（50%）
flat_roof_row_distance = 1  # in m # own assumption         #定义平屋顶光伏组件行间距为 1 米（自定义假设值）

pv_system_loss = 0.14             #定义光伏系统的总损耗率为 0.14（14%）
default_year = 2014 # the correponding reference year for PVGIS         #定义光伏系统发电量计算的参考年份为 2014 年，该年份在巴伐利亚州（德国）的太阳辐射量处于中等水平，适合作为典型年用于光伏潜力评估。