__author__ = "Sebastian Krapf"
__copyright__ = "Copyright 2023, "
__credits__ = ["Parts of code by Nils Kemmerzell"]
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Sebastian Krapf"
__email__ = "sebastian.krapf@tum.de"
__status__ = "alpha"

import rasterio
import cv2
import os

import numpy as np
import geopandas as gpd
import shapely
from shapely import MultiPolygon, Polygon, MultiLineString, LineString
from shapely.geometry import box

from utils_classextend import get_progress_string


def find_longest_line(polygon):
    """
    Finds longest line in roof segment to align solar panels.
    """
    ring_xy = polygon.exterior.xy
    x_coords = list(ring_xy[0])
    y_coords = list(ring_xy[1])

    longest_line_length = 0
    longest_line_coords = None

    for i, (x, y) in enumerate(zip(x_coords, y_coords)):
        try:
            next_x = x_coords[i + 1]
            next_y = y_coords[i + 1]
        except Exception:
            continue

        length = np.sqrt((next_x - x) ** 2 + (next_y - y) ** 2)
        if length > longest_line_length:
            longest_line_length = length
            longest_line_coords = [(x, y), (next_x, next_y)]

    return longest_line_coords


def azimuth_to_label_class(az, label_classes):
    """
    Convert azimuth to class label.
    label_classes should NOT include background, and should end with 'flat'.
    """
    directional_classes = label_classes[:-1]
    if az is None or np.isnan(az):
        az_class = "flat"
    else:
        surplus_angle = 360 / len(directional_classes) / 2
        az = az + 180 + surplus_angle
        if az > 360:
            az -= 360
        az_id = int(np.ceil(az / (360 / len(directional_classes))) - 1)
        az_class = directional_classes[az_id]
    return az_class


def label_class_to_azimuth(label_class):
    """
    Convert direction label to azimuth used by this project:
    South=0, East=-90, West=90, North=180
    """
    class_to_azimuth_map = {
        'N': 180,
        'NNE': -157.5,
        'NE': -135,
        'ENE': -112.5,
        'E': -90,
        'ESE': -67.5,
        'SE': -45,
        'SSE': -22.5,
        'S': 0,
        'SSW': 22.5,
        'SW': 45,
        'WSW': 67.5,
        'W': 90,
        'WNW': 112.5,
        'NW': 135,
        'NNW': 157.5,
        'flat': np.nan
    }
    if label_class not in class_to_azimuth_map:
        raise KeyError(f"Unknown label_class: {label_class}")
    return class_to_azimuth_map[label_class]


def filter_out_overlapping_polygons(gdf, percent_overlap=0.5, keep_mode="index"):
    keep_options = ["index", "larger"]
    assert keep_mode in keep_options, print(f"Keep_mode invalid. Please select {[k for k in keep_options]}")

    sindex = gdf.sindex

    intersecting_pairs = []
    for i, polygon in enumerate(gdf['geometry']):
        possible_matches_index = list(sindex.intersection(polygon.bounds))
        possible_matches = gdf.iloc[possible_matches_index]
        possible_matches = possible_matches[possible_matches.index != i]
        intersecting_pairs.extend(
            [(i, j) for j in possible_matches_index if polygon.overlaps(gdf['geometry'][j])]
        )

    overlapping_polygons = set()
    for pair in intersecting_pairs:
        intersection_area = gdf['geometry'][pair[0]].intersection(gdf['geometry'][pair[1]]).area
        if intersection_area / min(gdf['geometry'][pair[0]].area, gdf['geometry'][pair[1]].area) > percent_overlap:
            overlapping_polygons.add(pair[0])
            overlapping_polygons.add(pair[1])

    polygons_to_keep = set()
    for i in range(len(gdf)):
        if i not in overlapping_polygons:
            polygons_to_keep.add(i)

    for i, pair in enumerate(intersecting_pairs):
        progress_string = get_progress_string(round(i / len(intersecting_pairs), 2)) if len(intersecting_pairs) > 0 else ""
        print(f'Solving intersecting geom pairs {str(i)} {progress_string}')

        if pair[0] in overlapping_polygons and pair[1] in overlapping_polygons:
            if keep_mode == "index":
                overlapping_polygons_iter = set(pair)
                for other_pair in intersecting_pairs:
                    if pair[0] in other_pair or pair[1] in other_pair:
                        overlapping_polygons_iter.update(other_pair)
                polygons_to_keep.add(min(overlapping_polygons_iter))

            elif keep_mode == "larger":
                overlapping_polygons_iter = set(pair)
                for other_pair in intersecting_pairs:
                    if pair[0] in other_pair or pair[1] in other_pair:
                        overlapping_polygons_iter.update(other_pair)

                largest_polygon_index = max(overlapping_polygons_iter, key=lambda x: gdf['geometry'][x].area)
                polygons_to_keep.add(largest_polygon_index)

    gdf_segments_keep = gdf[gdf.index.isin(polygons_to_keep)]
    gdf_segments_to_drop = gdf[gdf.index.isin(polygons_to_keep) == False]

    return gdf_segments_keep, gdf_segments_to_drop


def save_img_as_geotiff(image, boundary, crs, save_path):
    height, width, bands = image.shape

    bbox = boundary.bounds
    transform = rasterio.transform.from_origin(
        bbox[0], bbox[3], (bbox[2] - bbox[0]) / width, (bbox[3] - bbox[1]) / height
    )

    meta = {
        'driver': 'GTiff',
        'count': bands,
        'width': width,
        'height': height,
        'crs': crs,
        'transform': transform,
        'dtype': str(image.dtype),
        'compress': 'lzw'
    }

    with rasterio.open(save_path, 'w', **meta) as dst:
        if bands == 4:
            dst.write(image[:, :, :3].transpose(2, 0, 1), indexes=[1, 2, 3])
            dst.write(image[:, :, 3] * 100, indexes=4)
        else:
            dst.write(image.transpose(2, 0, 1))

    return


def raster_to_vector(mask, id, image_bbox, CLASSES, bg_is_0=True):
    """
    Takes a mask as input and returns a GeoDataFrame of polygons.
    CLASSES should NOT include background.
    """
    label_list = []
    geometry_list = []
    image_shape = mask.shape

    for i, class_name in enumerate(CLASSES):
        prediction = np.zeros(image_shape)
        if bg_is_0:
            prediction[mask == i + 1] = 1
        else:
            prediction[mask == i] = 1
        prediction = prediction.astype(np.uint8)

        if np.sum(prediction) > 0:
            contours, _ = cv2.findContours(prediction, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            contours = np.array(contours, dtype=object)

            for cnt in contours:
                cnt = cnt.reshape(-1, 2)
                try:
                    shapely_poly = Polygon(cnt)
                except ValueError:
                    continue
                geometry_list.append(shapely_poly)
                label_list.append(CLASSES[i])

    image_bbox_px = box(0, 0, image_shape[0], image_shape[1])

    geometry_list = [
        convert_geocoord_and_pixel(geom, image_bbox_px, image_bbox, case='px_to_coord')
        for geom in geometry_list
    ]
    geometry_list = [Polygon(geometry) for geometry in geometry_list]

    gdf_labels = gpd.GeoDataFrame({
        'id': list([id]) * len(label_list),
        'label': label_list,
        'geometry': geometry_list
    })

    return gdf_labels


def switch_coordinates(x, y):
    return y, x


def convert_lonlat_to_latlon(obj):
    return shapely.ops.transform(switch_coordinates, obj)


def convert_points_geocoord_and_pixel(coords, img_box_orig, img_box_target):
    iox_min = img_box_orig.bounds[0]
    iox_max = img_box_orig.bounds[2]
    itx_min = img_box_target.bounds[0]
    itx_max = img_box_target.bounds[2]

    ioy_min = img_box_orig.bounds[1]
    ioy_max = img_box_orig.bounds[3]
    ity_min = img_box_target.bounds[1]
    ity_max = img_box_target.bounds[3]

    point_list = []
    for x, y in coords:
        x_new = itx_min + ((x - iox_min) / (iox_max - iox_min) * (itx_max - itx_min))
        y_new = ity_min + ((ioy_max - y) / (ioy_max - ioy_min) * (ity_max - ity_min))
        point_list.append((x_new, y_new))
    return point_list


def convert_geocoord_and_pixel(geom, img_box_px, img_box_latlon, case='coord_to_px'):
    if case == 'px_to_coord':
        img_box_orig = img_box_px
        img_box_target = img_box_latlon
    elif case == 'coord_to_px':
        img_box_orig = img_box_latlon
        img_box_target = img_box_px
    else:
        print('transformation case not covered: try case=px_to_latlon ')
        return geom

    if type(geom) == MultiPolygon:
        pol_list = []
        for pol in geom.geoms:
            p_list = convert_points_geocoord_and_pixel(pol.exterior.coords, img_box_orig, img_box_target)
            pol_list.append(Polygon(p_list))
        converted_geom = MultiPolygon(pol_list)
    elif type(geom) == MultiLineString:
        linestring_list = []
        for ls in geom.geoms:
            p_list = convert_points_geocoord_and_pixel(ls.coords, img_box_orig, img_box_target)
            linestring_list.append(LineString(p_list))
        converted_geom = MultiLineString(linestring_list)
    elif type(geom) == Polygon:
        coords = geom.exterior.coords
        point_list = convert_points_geocoord_and_pixel(coords, img_box_orig, img_box_target)
        converted_geom = Polygon(point_list)
    elif type(geom) == LineString:
        coords = geom.coords
        point_list = convert_points_geocoord_and_pixel(coords, img_box_orig, img_box_target)
        converted_geom = LineString(point_list)
    else:
        converted_geom = geom
    return converted_geom


def simplify_polygon(polygon, tolerance=1):
    simplified_polygon = polygon.simplify(tolerance=tolerance, preserve_topology=False)
    if type(simplified_polygon) == MultiPolygon:
        simplified_polygon = simplified_polygon.geoms[0]
    if simplified_polygon.is_empty:
        simplified_polygon = polygon.simplify(tolerance=0.5, preserve_topology=False)
        if type(simplified_polygon) == MultiPolygon:
            simplified_polygon = simplified_polygon.geoms[0]
        if simplified_polygon.is_empty:
            simplified_polygon = polygon
    return simplified_polygon


def calculate_rotation_angle(line):
    """
    Calculate angle to rotate modules.
    """
    try:
        slope = (line[1][1] - line[0][1]) / (line[1][0] - line[0][0])
        angle_rad = np.arctan(slope)
        angle_deg = angle_rad * 180 / np.pi
    except ZeroDivisionError:
        angle_deg = 90
    return angle_deg


def opposite_angle(angle):
    opposite = (angle + 180) % 360
    if opposite > 180:
        opposite -= 360
    return opposite


def _circular_angle_diff(a, b):
    """
    Smallest angular difference in degrees.
    Works for angles in [-180, 180] or [0, 360].
    """
    d = abs(a - b) % 360
    return min(d, 360 - d)


def select_azimuth(orientation, angle1, angle2, orientation_mapping=None):
    """
    Select the azimuth that better matches the segment label.
    Supports 6-class / 10-class / 18-class direction labels.

    orientation examples:
    N, NE, E, SE, S, SW, W, NW,
    NNE, ENE, ESE, SSE, SSW, WSW, WNW, NNW, flat
    """
    orig_angle1 = angle1
    orig_angle2 = angle2

    if orientation == "flat":
        return np.nan

    if orientation_mapping is None:
        target_angle = label_class_to_azimuth(orientation)
    else:
        target_angle = orientation_mapping[orientation]

    diff1 = _circular_angle_diff(orig_angle1, target_angle)
    diff2 = _circular_angle_diff(orig_angle2, target_angle)

    selected_angle = orig_angle1 if diff1 < diff2 else orig_angle2
    return selected_angle


def get_image_gdf_in_directory(DIR_IMAGES_GEOTIFF, save_to_png_path=[]):
    image_id_list = [id[:-4] for id in os.listdir(DIR_IMAGES_GEOTIFF) if id[-4:] == '.tif']

    raster_srcs = [rasterio.open(os.path.join(DIR_IMAGES_GEOTIFF, str(image_id) + ".tif")) for image_id in image_id_list]
    image_bbox_list = []
    image_width_px = []
    image_height_px = []
    print('')

    for i, raster_src in enumerate(raster_srcs):
        progress_string = get_progress_string(round(i / len(raster_srcs), 2)) if len(raster_srcs) > 0 else ""
        print('Loading geo_tiffs: ' + progress_string, end="\r")

        if len(save_to_png_path) > 0:
            filename_mask = os.path.join(save_to_png_path, str(image_id_list[i]) + '.png')
            data = raster_src.read()
            img = np.dstack((data[0, :, :], data[1, :, :], data[2, :, :]))
            cv2.imwrite(filename_mask, img)

        band_shape = raster_src.shape
        image_width_px.append(band_shape[1])
        image_height_px.append(band_shape[0])

        transform = raster_src.transform
        ulx, uly = transform * (0, 0)
        lrx, lry = transform * (raster_src.width, raster_src.height)

        image_bbox = shapely.geometry.box(ulx, lry, lrx, uly)
        image_bbox_list.append(image_bbox)

    gdf_images = gpd.GeoDataFrame({
        'id': image_id_list,
        'geometry': image_bbox_list,
        'image_width_px': image_width_px,
        'image_height_px': image_height_px,
    })
    gdf_images.crs = raster_srcs[0].crs.to_dict()
    return gdf_images