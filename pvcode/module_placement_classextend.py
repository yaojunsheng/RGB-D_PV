__author__ = "Sebastian Krapf"
__copyright__ = "Copyright 2023, "
__credits__ = ["Parts of code by Nils Kemmerzell"]
__license__ = "GNU GPLv3"
__version__ = "0.1"
__maintainer__ = "Sebastian Krapf"
__email__ = "sebastian.krapf@tum.de"
__status__ = "alpha"

import math

import geopandas as gpd
import numpy as np

from shapely.affinity import rotate
from shapely.errors import TopologicalError
from shapely.ops import cascaded_union, transform
from shapely.geometry import Point, MultiPolygon, Polygon, LineString, GeometryCollection

from definitions import (
    epsg_default,
    epsg_metric_germany,
    flat_roof_orientation_mode,
    flat_roof_space_util,
    flat_roof_row_distance
)
from spatial_operations_classextend import calculate_rotation_angle


def create_pv_modules_gdf(alignment_list, gdf_modules_vertical, gdf_modules_horizontal, azimuth_list):
    modules_placed_list = []
    for i, alignment in enumerate(alignment_list):
        if alignment == "vertical":
            modules_placed_list.append(gdf_modules_vertical.geometry.iloc[i])
        elif alignment == "horizontal":
            modules_placed_list.append(gdf_modules_horizontal.geometry.iloc[i])
    gdf_modules = gpd.GeoDataFrame({
        "azimuth": azimuth_list,
        "geometry": modules_placed_list
    })
    gdf_modules.crs = epsg_default
    return gdf_modules


def correct_modules(modules, roof_shape, obstacles):
    """
    Check if module is within the roof segment and does not intersect with a superstructure.
    """
    module_list = []

    for module in modules.geoms:
        if obstacles:
            if isinstance(obstacles, Polygon):
                obstacles = MultiPolygon([obstacles])
            elif isinstance(obstacles, LineString):
                print('this should not be the case')
            else:
                try:
                    obstacles = MultiPolygon(obstacles)
                except Exception:
                    pass
            try:
                if module.within(roof_shape) and not module.intersects(obstacles):
                    module_list.append(module)
            except Exception:
                pass
        else:
            if module.within(roof_shape):
                module_list.append(module)
    modules_mp = MultiPolygon(p for p in module_list)

    return modules_mp


def module_dimensions_distortion(module_height, module_width, slope):
    def correct_length(length, angle):
        angle = angle / 360 * 2 * np.pi
        return np.round(length * np.cos(angle), 2)

    module_height = correct_length(module_height, slope)
    module_width = correct_length(module_width, slope)
    return module_height, module_width


def link_superstructure_to_segment(list_obstacles, list_segments):
    dictionary = {}
    for i, segment in enumerate(list_segments):
        belonging_obstacles = []
        for superstructure in list_obstacles:
            if segment.contains(superstructure):
                belonging_obstacles.append(superstructure)
            elif segment.overlaps(superstructure):
                intersection = segment.intersection(superstructure)
                belonging_obstacles.append(intersection)

        dictionary[i] = belonging_obstacles

    return dictionary


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


def modules_shape(module_width, module_height, num_mod_x, num_mod_y, origin):
    x_corners = module_width * np.arange(0, num_mod_x + 0.5, 1) + origin[0] + 1
    y_corners = module_height * np.arange(0, num_mod_y + 0.5, 1) + origin[1] + 1
    module_list = []
    for i, _ in enumerate(x_corners[1:]):
        for j, _ in enumerate(y_corners[1:]):
            module_list.append(Polygon([
                [x_corners[i], y_corners[j]],
                [x_corners[i], y_corners[j + 1]],
                [x_corners[i + 1], y_corners[j + 1]],
                [x_corners[i + 1], y_corners[j]]
            ]))
    shape = MultiPolygon(p for p in module_list)
    return shape


def place_modules(roof, obstacles, module_height, module_width, slope):
    """
    Places modules of a given height and width into a Polygon.
    """
    def move_to_origin(x_coordinate, y_coordinate):
        return x_coordinate - delta_x, y_coordinate - delta_y

    def move_back(x_coordinate, y_coordinate):
        return x_coordinate + delta_x, y_coordinate + delta_y

    roof = Polygon(roof.exterior.coords)

    delta_x = roof.bounds[0]
    delta_y = roof.bounds[1]

    roof_transformed = transform(move_to_origin, roof)
    longest_line = find_longest_line(roof_transformed)
    rotation_angle = calculate_rotation_angle(longest_line)
    rotation_center = roof_transformed.centroid
    rot_poly = rotate(roof_transformed, -rotation_angle, rotation_center)
    rot_bb = rot_poly.envelope
    x_roof, y_roof = rot_bb.exterior.xy

    if not obstacles.is_empty:
        obstacles_transformed = transform(move_to_origin, obstacles)
        obstacles_trans_rot = rotate(obstacles_transformed, -rotation_angle, rotation_center)
    else:
        obstacles_trans_rot = MultiPolygon()

    module_height_distorted, module_width_distorted = module_dimensions_distortion(module_height, module_width, slope)

    num_mod_x = np.ceil((max(x_roof) - min(x_roof)) / module_width)
    num_mod_y = np.ceil((max(y_roof) - min(y_roof)) / module_height_distorted)
    modules_vert = modules_shape(
        module_width, module_height_distorted, num_mod_x, num_mod_y, (rot_bb.bounds[0], rot_bb.bounds[1])
    )

    num_mod_x = np.ceil((max(x_roof) - min(x_roof)) / module_height)
    num_mod_y = np.ceil((max(y_roof) - min(y_roof)) / module_width_distorted)
    modules_hor = modules_shape(
        module_height, module_width_distorted, num_mod_x, num_mod_y, (rot_bb.bounds[0], rot_bb.bounds[1])
    )

    modules_vert_corrected = correct_modules(modules_vert, rot_poly, obstacles_trans_rot)
    modules_hor_corrected = correct_modules(modules_hor, rot_poly, obstacles_trans_rot)

    if len(modules_vert_corrected.geoms) >= len(modules_hor_corrected.geoms):
        best_modules = modules_vert_corrected
        alignment = 'vertical'
    else:
        best_modules = modules_hor_corrected
        alignment = 'horizontal'

    best_modules_or = transform(move_back, best_modules)
    best_modules_rot = rotate(best_modules_or, rotation_angle, origin=roof.centroid)

    best_modules_vertical_original = transform(move_back, modules_vert_corrected)
    best_modules_vertical_rot = rotate(best_modules_vertical_original, rotation_angle, origin=roof.centroid)

    best_modules_horizontal_original = transform(move_back, modules_hor_corrected)
    best_modules_horizontal_rot = rotate(best_modules_horizontal_original, rotation_angle, origin=roof.centroid)

    modules_possible_vertical = best_modules_vertical_rot
    modules_possible_horizontal = best_modules_horizontal_rot

    return alignment, modules_possible_vertical, modules_possible_horizontal


def place_modules_flatroof(
        roof,
        obstacles,
        module_height,
        module_width,
        slope,
        mode,
        space_util,
        row_distance):
    """
    Places modules of a given height and width into a Polygon for flat roofs.
    """
    assert mode in ['south', 'east-west', 'alignment']

    def move_to_origin(x_coordinate, y_coordinate):
        return x_coordinate - delta_x, y_coordinate - delta_y

    def move_back(x_coordinate, y_coordinate):
        return x_coordinate + delta_x, y_coordinate + delta_y

    roof = Polygon(roof.exterior.coords)

    delta_x = roof.bounds[0]
    delta_y = roof.bounds[1]

    roof_transformed = transform(move_to_origin, roof)

    longest_line = find_longest_line(roof_transformed)
    rotation_angle = calculate_rotation_angle(longest_line)
    if mode in ['south', 'east-west']:
        rotation_angle = 0
    rotation_center = roof_transformed.centroid
    rot_poly = rotate(roof_transformed, -rotation_angle, origin='centroid')
    rot_bb = rot_poly.envelope
    x_roof, y_roof = rot_bb.exterior.xy

    if not obstacles.is_empty:
        obstacles_transformed = transform(move_to_origin, obstacles)
        obstacles_trans_rot = rotate(obstacles_transformed, -rotation_angle, rotation_center)
    else:
        obstacles_trans_rot = MultiPolygon()

    num_mod_x = math.ceil((max(x_roof) - min(x_roof)) / module_width)
    num_mod_y = math.ceil((max(y_roof) - min(y_roof)) / correct_length(module_height, slope))
    num_mod_x = max(num_mod_x, num_mod_y)
    num_mod_y = num_mod_x
    modules_vert = modules_on_flatroof(
        module_width,
        correct_length(module_height, slope),
        num_mod_x,
        num_mod_y,
        (rot_bb.bounds[0], rot_bb.bounds[1]),
        mode,
        'vertical',
        space_util,
        row_distance
    )

    modules_vert2 = modules_on_flatroof(
        module_width,
        correct_length(module_height, slope),
        num_mod_x,
        num_mod_y,
        (rot_bb.bounds[0], rot_bb.bounds[1]),
        mode,
        'horizontal',
        space_util,
        row_distance
    )

    num_mod_x = math.ceil((max(x_roof) - min(x_roof)) / module_height)
    num_mod_y = math.ceil((max(y_roof) - min(y_roof)) / correct_length(module_width, slope))
    num_mod_x = max(num_mod_x, num_mod_y)
    num_mod_y = num_mod_x
    modules_hor = modules_on_flatroof(
        module_height,
        correct_length(module_width, slope),
        num_mod_x,
        num_mod_y,
        (rot_bb.bounds[0], rot_bb.bounds[1]),
        mode,
        'vertical',
        space_util,
        row_distance
    )

    modules_hor2 = modules_on_flatroof(
        module_height,
        correct_length(module_width, slope),
        num_mod_x,
        num_mod_y,
        (rot_bb.bounds[0], rot_bb.bounds[1]),
        mode,
        'horizontal',
        space_util,
        row_distance
    )

    modules_vert_corrected = correct_modules(modules_vert, rot_poly, obstacles_trans_rot)
    modules_vert_corrected2 = correct_modules(modules_vert2, rot_poly, obstacles_trans_rot)
    modules_hor_corrected = correct_modules(modules_hor, rot_poly, obstacles_trans_rot)
    modules_hor_corrected2 = correct_modules(modules_hor2, rot_poly, obstacles_trans_rot)

    best_modules_horizontal = MultiPolygon(
        max([modules_hor_corrected.geoms, modules_hor_corrected2.geoms], key=lambda mp: len(mp))
    )
    best_modules_horizontal_original = transform(move_back, best_modules_horizontal)
    best_modules_horizontal_rot = rotate(best_modules_horizontal_original, rotation_angle, origin=roof.centroid)

    best_modules_vertical = MultiPolygon(
        max([modules_vert_corrected.geoms, modules_vert_corrected2.geoms], key=lambda mp: len(mp))
    )
    best_modules_vertical_original = transform(move_back, best_modules_vertical)
    best_modules_vertical_rot = rotate(best_modules_vertical_original, rotation_angle, origin=roof.centroid)

    if len(best_modules_vertical.geoms) >= len(best_modules_horizontal.geoms):
        alignment = 'vertical'
    else:
        alignment = 'horizontal'

    if mode == 'alignment':
        azimuth = rotation_angle
    elif mode == 'south':
        azimuth = 0
    else:
        azimuth = 90

    modules_possible_vertical = best_modules_vertical_rot
    modules_possible_horizontal = best_modules_horizontal_rot
    return alignment, modules_possible_vertical, modules_possible_horizontal, azimuth


def modules_on_flatroof(
        module_width,
        module_height,
        num_mod_x,
        num_mod_y,
        origin,
        mode,
        stack_orientation,
        space_util=0.5,
        row_distance=1):
    """
    Place modules on a flat roof with space between rows.
    """
    module_list = []
    if mode == 'south':
        alpha = 36 / 180 * math.pi
        distance = np.round(module_height * (1 / (math.cos(alpha) * space_util) - 1), 1)
        x_corners = np.arange(0, module_width * (num_mod_x + 1), module_width) + origin[0] + 1
        y_corners_lower = np.arange(
            0, (module_height + distance) * (num_mod_y + 1), module_height + distance
        ) + origin[0] + 1
        y_corners_upper = np.arange(
            module_height, (module_height + distance) * (num_mod_y + 1), module_height + distance
        ) + origin[0] + 1

        for j in range(len(y_corners_lower) - 1):
            for i in range(len(x_corners) - 1):
                module_list.append(Polygon([
                    [x_corners[i], y_corners_lower[j]],
                    [x_corners[i], y_corners_upper[j]],
                    [x_corners[i + 1], y_corners_upper[j]],
                    [x_corners[i + 1], y_corners_lower[j]]
                ]))

    else:
        if mode == 'east-west':
            stack_orientation = 'horizontal'

        if stack_orientation == 'horizontal':
            row_distance *= module_width

            x_corners = np.array([
                module_width * x + (row_distance - module_width) * (x // 3)
                for x in range(num_mod_y)
            ]) + origin[0] + 1

            y_corners = module_height * np.arange(0, (num_mod_x + 1), 1) + origin[0] + 1
            for j in range(len(y_corners) - 1):
                for i in range(len(x_corners) - 1):
                    if (i + 1) % 3 != 0:
                        module_list.append(Polygon([
                            [x_corners[i], y_corners[j]],
                            [x_corners[i], y_corners[j + 1]],
                            [x_corners[i + 1], y_corners[j + 1]],
                            [x_corners[i + 1], y_corners[j]]
                        ]))
        else:
            row_distance = module_height
            x_corners = module_width * np.arange(0, (num_mod_x + 1), 1) + origin[0] + 1
            y_corners = np.array([
                module_height * x + (row_distance - module_height) * (x // 3)
                for x in range(num_mod_y)
            ]) + origin[0] + 1

            for j in range(len(y_corners) - 1):
                for i in range(len(x_corners) - 1):
                    if (j + 1) % 3 != 0:
                        module_list.append(Polygon([
                            [x_corners[i], y_corners[j]],
                            [x_corners[i], y_corners[j + 1]],
                            [x_corners[i + 1], y_corners[j + 1]],
                            [x_corners[i + 1], y_corners[j]]
                        ]))

    shape = MultiPolygon(module_list)
    return shape


def correct_length(length, angle):
    angle = angle / 360 * 2 * math.pi
    return np.round(length * math.cos(angle), 2)


def check_obstacles(modules, obstacles):
    """
    Check if obstacles intersect with modules.
    """
    module_list = []
    for module in modules.geoms:
        try:
            obstacles = MultiPolygon(obstacles)
        except ValueError:
            for ss in obstacles:
                if ss.geom_type == 'MultiPolygon':
                    obstacles.remove(ss)
                    obstacles += [polygon for polygon in ss]
            obstacles = MultiPolygon(obstacles)
        try:
            if not module.intersects(obstacles):
                module_list.append(module)
        except TopologicalError:
            intersection = False
            for s in obstacles:
                intersection = module.intersects(s) or intersection
            if not intersection:
                module_list.append(module)
    modules_mp = MultiPolygon(module_list)

    return modules_mp


def drop_ignored_modules(modules_shape_list, modules_ignore_list):
    id_count = 0
    modules_shape_list_filtered = []
    for module_shapes in modules_shape_list:
        module_polygon_list = [
            module_shapes[id - id_count] for id in np.arange(id_count, id_count + len(module_shapes))
            if id not in modules_ignore_list
        ]
        module_multipolygon = MultiPolygon(module_polygon_list)
        modules_shape_list_filtered.append(module_multipolygon)
        id_count += len(module_shapes)

    num_modules_list = [len(modules) for modules in modules_shape_list_filtered]
    return modules_shape_list_filtered, num_modules_list


def drop_ignored_segments(drop_id, old_list):
    filtered_list = [old_list[id] for id in np.arange(0, len(old_list)) if id not in drop_id]
    return filtered_list


def module_selection(is_single_segment, modules_shape_list, num_modules_list, E_gen_rank_index_list):
    modules_shape_list_M = modules_shape_list[E_gen_rank_index_list[0]]
    mplist = []
    for m in modules_shape_list:
        for p in m:
            mplist.append(p)
    modules_shape_list_L = MultiPolygon(mplist)

    modules_shape_list_selected = [None]
    modules_shape_list_selected.append(modules_shape_list_M)
    if is_single_segment is True:
        modules_shape_list_selected.append(modules_shape_list_M)
    else:
        modules_shape_list_selected.append(modules_shape_list_L)

    num_modules_list_S = list(np.zeros(len(num_modules_list)))
    num_modules_list_S[0] = 2
    num_modules_list_M = list(np.zeros(len(num_modules_list)))
    num_modules_list_M[E_gen_rank_index_list[0]] = num_modules_list[E_gen_rank_index_list[0]]
    num_modules_list_L = num_modules_list

    num_modules_list_selected = list([num_modules_list_S])
    num_modules_list_selected.append(num_modules_list_M)

    if is_single_segment is True:
        num_modules_list_selected.append(num_modules_list_M)
    else:
        num_modules_list_selected.append(num_modules_list_L)

    return modules_shape_list_selected, num_modules_list_selected


def module_placement(gdf_segments, azimuth_list, slope_list, gdf_superstructures, module_height, module_width):
    """
    Main module placement pipeline.
    """
    gdf_segments = gdf_segments.to_crs(epsg_metric_germany)
    gdf_superstructures = gdf_superstructures.to_crs(epsg_metric_germany)

    modules_possible_vertical_list = []
    modules_possible_horizontal_list = []
    alignment_list = []
    azimuths = []

    for i, roof_segment in enumerate(gdf_segments.iloc):
        if isinstance(roof_segment.geometry, MultiPolygon):
            segment_shape = roof_segment.geometry.geoms[0]
        else:
            segment_shape = roof_segment.geometry

        if len(gdf_superstructures) > 0:
            superstructures = MultiPolygon([[] if pol.is_empty else pol for pol in gdf_superstructures.geometry.iloc])
        else:
            superstructures = MultiPolygon()

        if slope_list[i] == 0:
            alignment, modules_possible_vertical, modules_possible_horizontal, azimuth = \
                place_modules_flatroof(
                    roof=segment_shape,
                    obstacles=superstructures,
                    module_height=module_height,
                    module_width=module_width,
                    slope=slope_list[i],
                    mode=flat_roof_orientation_mode,
                    space_util=flat_roof_space_util,
                    row_distance=flat_roof_row_distance
                )
        else:
            alignment, modules_possible_vertical, modules_possible_horizontal = \
                place_modules(
                    roof=segment_shape,
                    obstacles=superstructures,
                    module_height=module_height,
                    module_width=module_width,
                    slope=slope_list[i]
                )
            azimuth = azimuth_list[i]

        modules_possible_vertical_list.append(modules_possible_vertical)
        modules_possible_horizontal_list.append(modules_possible_horizontal)
        alignment_list.append(alignment)
        azimuths.append(azimuth)

    gdf_modules_vertical = gpd.GeoDataFrame({"geometry": modules_possible_vertical_list})
    gdf_modules_vertical.crs = epsg_metric_germany
    gdf_modules_vertical = gdf_modules_vertical.to_crs(epsg_default)

    gdf_modules_horizontal = gpd.GeoDataFrame({"geometry": modules_possible_horizontal_list})
    gdf_modules_horizontal.crs = epsg_metric_germany
    gdf_modules_horizontal = gdf_modules_horizontal.to_crs(epsg_default)

    azimuth_list = azimuths

    return alignment_list, gdf_modules_vertical, gdf_modules_horizontal, azimuth_list


def drop_modules_outside_outline(modules_placed_list, outline):
    """
    :param modules_placed_list: list of MultiPolygons
    :param outline: GeoSeries with crs set
    :return: update list of MultiPolygons that are within outline
    """
    if outline.crs != 4326:
        outline = outline.to_crs(4326)

    modules_placed_list_new = []
    modules_dropped = []
    for i in np.arange(0, len(modules_placed_list)):
        mpol_modules = modules_placed_list[i]

        assert isinstance(mpol_modules, MultiPolygon), print("Program expects modules in MultiPolygon format.")
        modules_within = [module for module in mpol_modules.geoms if module.within(outline.geometry.iloc[0])]
        mpol_modules_new = MultiPolygon(modules_within)
        modules_placed_list_new.append(mpol_modules_new)

        modules_outside = [module for module in mpol_modules.geoms if not module.within(outline.geometry.iloc[0])]
        modules_dropped.append(MultiPolygon(modules_outside))
    return modules_placed_list_new, modules_dropped


def create_single_module_gdf(gs_modules, building_id):
    """
    Takes the placed module result as GeoSeries and creates a GeoDataFrame with module polygons
    """
    modules = []
    building_ids = []
    segment_ids = []

    for seg_id, mpol_modules in enumerate(list(gs_modules)):
        [modules.append(module) for module in mpol_modules.geoms]
        [building_ids.append(building_id) for _ in mpol_modules.geoms]
        [segment_ids.append(seg_id) for _ in mpol_modules.geoms]

    gdf_modules = gpd.GeoDataFrame({
        "building_id": building_ids,
        "segment_id": segment_ids,
        "geometry": modules
    })
    gdf_modules.crs = gs_modules.crs

    return gdf_modules