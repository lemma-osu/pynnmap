import os
import sys

from pynnmap.misc import geoprocessor
from pynnmap.parser import xml_stand_metadata_parser as xsmp


def main(parser):
    """
    This script runs all of the post-processing operations that use
    the arcgis geoprocessor
    """

    # Load grids into lists corresponding to type
    axes = []
    distances = []
    neighbors = []

    frmt_dict = {
        'GIO': '',
        'GTiff': '.tif',
        'ENVI': '.bsq',
        'HFA': '.img',
    }

    def build_name(model_dir, prefix, number, frmt):
        base_name = ''.join((prefix, str(number), frmt_dict[frmt]))
        return os.path.join(model_dir, base_name)

    md = parser.model_directory
    of = parser.output_format
    for n in range(1, int(parser.write_axes) + 1):
        axis_grid = build_name(md, parser.axes_file, n, of)
        axes.append(axis_grid)

    for n in range(1, int(parser.write_neighbors) + 1):
        neighbor_grid = build_name(md, parser.neighbor_file, n, of)
        neighbors.append(neighbor_grid)

    for n in range(1, int(parser.write_distances) + 1):
        distance_grid = build_name(md, parser.distance_file, n, of)
        distances.append(distance_grid)

    # Define projections on all grids
    define_projections(axes, parser.projection_file)
    define_projections(neighbors, parser.projection_file)
    define_projections(distances, parser.projection_file)

    # Convert axis and distance grids to integers (scaled by 100.0)
    integerize_rasters(axes)
    integerize_rasters(distances)

    # Copy the neighbor rasters to force recalculation of statistics
    copy_rasters(neighbors)

    # Join attributes to original neighbor grids
    attribute_join_field = parser.plot_id_field
    for neighbor in neighbors:
        join_attributes(
            neighbor, 'VALUE', parser.stand_attribute_file,
            attribute_join_field, parser.stand_metadata_file
        )

    # Copy, clip and mask neighbor grids & join attributes
    if parser.mask_raster != '':
        for i, neighbor in enumerate(neighbors):
            masked_neighbor = '%s/mr%d_nnmsk%d' % (
                parser.model_directory, parser.model_region, i + 1)
            create_masked_raster(
                neighbor, parser.boundary_raster, parser.mask_raster,
                masked_neighbor)
            join_attributes(
                masked_neighbor, 'VALUE', parser.stand_attribute_file,
                attribute_join_field, parser.stand_metadata_file)


def get_path(file_name):
    """
    Return the path portion of the file_name
    If no path is given, return the current working directory

    Parameters
    ----------
    file_name : str
        name of file and optional path

    Returns
    -------
    path : str

    """
    path = os.path.dirname(file_name)
    if path == '':
        path = os.getcwd()
    return path


def build_vats(rasters):
    """
    Build value attribute tables

    Parameters:
    -----------
    rasters : list
        list of rasters to build vats (or update if vat already built)

    Returns:
    --------
    None
    """
    for raster in rasters:
        try:
            model_dir = get_path(raster)
            gp = geoprocessor.Geoprocessor(model_dir)
            gp.build_vat(raster)
        except:
            print sys.exc_info()
            # If there was an error, continue with the next raster
            continue


def create_masked_raster(in_raster, boundary_raster, mask_raster, out_raster):
    """
    Create a masked raster

    Parameters:
    -----------
    in_raster : str
        input raster to apply mask
    boundary_raster : str
        boundary raster used to clip masked output
    mask_raster: str
        raster to use as a mask
    out_raster: str
        name of output masked raster

    Returns:
    --------
    None
    """
    model_dir = get_path(in_raster)
    gp = geoprocessor.Geoprocessor(model_dir)
    gp.create_clipped_masked_raster(
        in_raster, boundary_raster, mask_raster, out_raster)


def define_projections(rasters, projection_file):
    """
    Define projections on rasters

    Parameters:
    -----------
    rasters : list
        list of rasters to define projections on
    projection_file : str
        name and full path of file containing projection parameters

    Returns:
    --------
    None
    """
    for raster in rasters:
        try:
            model_dir = get_path(raster)
            gp = geoprocessor.Geoprocessor(model_dir)
            gp.define_projection(raster, projection_file)
        except:
            print sys.exc_info()
            continue


def integerize_rasters(rasters):
    """
    Convert floating point rasters to integer rasters.  The rasters are
    overwritten to the same input names

    Parameters:
    -----------
    rasters : list
        list of rasters to convert to integers

    Returns:
    --------
    None
    """
    for raster in rasters:
        try:
            model_dir = get_path(raster)
            gp = geoprocessor.Geoprocessor(model_dir)
            gp.overwrite(gp.convert_to_integer, raster)
        except:
            print sys.exc_info()
            continue


def copy_rasters(rasters):
    """
    Copy rasters without the attribute table.  The rasters are overwritten
    to the same input names

    Parameters:
    -----------
    rasters : list
        list of rasters to copy

    Returns:
    --------
    None
    """
    for raster in rasters:
        try:
            model_dir = get_path(raster)
            gp = geoprocessor.Geoprocessor(model_dir)
            gp.overwrite(gp.copy_raster_no_attributes, raster)
        except:
            print sys.exc_info()
            continue


def join_attributes(
        raster, raster_join_field, attribute_file, attribute_join_field,
        attribute_metadata_file):
    """
    Join attributes to a raster

    Parameters:
    -----------
    raster : str
        raster to join attributes to
    raster_join_field : str
        name of join field in raster
    attribute_file: str
        name of file with attributes to join to raster
    attribute_join_field: str
        name of join field in attribute file
    attribute_metadata_file: str
        name of file with attribute metadata to decide what variables to
        drop from join file

    Returns:
    --------
    None
    """
    model_dir = get_path(raster)
    gp = geoprocessor.Geoprocessor(model_dir)

    # create list of attributes to drop from join file - we only want a
    # subset of all variables joined to the NN grids (PROJECT_ATTR = 1), so
    # we need to specify all variables that have PROJECT_ATTR = 0 in the
    # metadata
    mp = xsmp.XMLStandMetadataParser(attribute_metadata_file)
    drop_fields = [x.field_name for x in mp.attributes if x.project_attr == 0]

    try:
        gp.join_attributes(
            raster, raster_join_field, attribute_file, attribute_join_field,
            drop_fields=drop_fields)
    except:
        print sys.exc_info()


if __name__ == '__main__':
    main(sys.argv[1:])
