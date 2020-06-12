import os
import subprocess

import click
import numpy as np
import pandas as pd
import rasterio
from affine import Affine
from rasterio.windows import Window

from pynnmap.misc import utilities
from pynnmap.parser import parameter_parser_factory as ppf
from pynnmap.parser import xml_stand_metadata_parser as xsmp
from . import scalars


# Weights
K7_WEIGHTS = np.array([0.6321, 0.2325, 0.0855, 0.0315, 0.0116, 0.0043, 0.0016,])
K7_WEIGHTS /= K7_WEIGHTS.sum()

WEIGHTS = {1: np.array([1.0]), 7: K7_WEIGHTS}


def get_weights(k):
    return WEIGHTS[k]


def create_neighbor_rasters(p):
    # TODO: This is a total hack to hard code the parameter file name.  We
    # TODO: only have the parameter parser object at this point.
    # TODO: Because we are still calling the C program, we're still
    # TODO: requiring the model filename as input.
    md = p.model_directory
    model_fn = os.path.join(md, "model.xml")
    os.chdir(md)
    cmd = " ".join(("gnnrun", model_fn))
    subprocess.call(cmd)


def get_attribute_df(csv_fn, attr):
    return pd.read_csv(csv_fn, usecols=["FCID", attr.upper()])


def get_attribute_array(csv_fn, attr):
    df = get_attribute_df(csv_fn, attr)
    sparse_df = pd.DataFrame({"FCID": np.arange(1, df.FCID.max() + 1)})
    sparse_df = sparse_df.merge(df, on="FCID", how="left").fillna(0.0)
    sparse_attr_arr = sparse_df.values
    return np.insert(sparse_attr_arr, 0, [0, 0], axis=0)


def open_output_raster(profile, window, scalar, out_fn):
    c, r, w, h = window.flatten()
    transform = profile["transform"] * Affine.translation(c, r)
    profile.update(
        dtype=rasterio.int32,
        count=1,
        compress="lzw",
        transform=transform,
        nodata=-32768,
        height=h,
        width=w,
    )
    dst = rasterio.open(out_fn, "w", **profile)
    dst.update_tags(SCALAR=scalar)
    return dst


def get_nn_arrays(rasters, window):
    return [x.read(1, window=window, masked=True) for x in rasters]


def get_raster(fn):
    return rasterio.open(fn)


def get_array(raster, window):
    return raster.read(1, window=window, masked=True)


def weighted(id_arrs, attr_arr, weights):
    attr_stack = np.dstack([np.take(attr_arr[:, 1], x) for x in id_arrs])
    return (attr_stack * weights).sum(axis=2)


def generate_row_blocks(src, bounds):
    w = src.window(*bounds)
    c, r, w, h = map(int, w.flatten())
    return (
        Window.from_slices((r + x, r + x + 1), (c, c + w)) for x in range(h)
    )


def process_raster(
    nn_rasters,
    mask_raster,
    nf_raster,
    attr_arr,
    window,
    weights,
    scalar,
    out_raster,
):
    # Get the bounding coordinates of the window relative to the first
    # neighbor raster and generate row blocks for each of the datasets
    bounds = nn_rasters[0].window_bounds(window)
    nn_windows = generate_row_blocks(nn_rasters[0], bounds)
    nf_windows = generate_row_blocks(nf_raster, bounds)
    mask_windows = generate_row_blocks(mask_raster, bounds)

    # Metadata from output raster
    dtype = out_raster.dtypes[0]
    nd = out_raster.nodata

    # Iterate over rows
    zipped = zip(nn_windows, nf_windows, mask_windows)
    for nn_window, nf_window, mask_window in zipped:
        # Extract the arrays
        nn_arrs = get_nn_arrays(nn_rasters, nn_window)
        nf_arr = get_array(nf_raster, nf_window).astype(np.bool)
        mask_arr = get_array(mask_raster, mask_window)

        # Calculate the weighted value
        out_arr = weighted(nn_arrs, attr_arr, weights=weights)

        # Apply the scalar and round
        out_arr = np.round(out_arr * scalar)

        # Burn in nonforest areas as -1
        out_arr = np.where(nf_arr, out_arr, -1)

        # Mask and fill
        nn_mask = np.any([x.mask for x in nn_arrs], axis=0)
        mask = np.logical_and(nf_arr, nn_mask)
        mask = np.logical_or(mask, mask_arr.mask)
        out_arr = np.ma.masked_array(out_arr, mask=mask)

        # Write out array
        out_raster.write(
            out_arr.filled(nd).astype(dtype), 1, window=mask_window
        )


@click.command(
    name="build-attribute-raster",
    short_help="Build raster for numerical attributes",
)
@click.argument("parameter-file", type=click.Path(exists=True), required=True)
@click.argument("attribute", type=click.STRING, required=True)
def main(parameter_file, attribute):
    # Read in the parameters
    p = ppf.get_parameter_parser(parameter_file)

    # Get the value of k
    attr_name = attribute.lower()
    k = scalars.get_k(attr_name.upper())

    # Build the neighbor rasters if not present
    nn_files = [p.get_neighbor_file(idx) for idx in range(1, k + 1)]
    try:
        _ = [rasterio.open(x) for x in nn_files]
    except rasterio.errors.RasterioIOError:
        create_neighbor_rasters(p)

    # Get the metadata parser and get the project area attributes
    mp = xsmp.XMLStandMetadataParser(p.stand_metadata_file)
    attrs = mp.get_area_attrs()

    # Find the attribute in the list
    try:
        _ = [x for x in attrs if x.field_name.lower() == attr_name][0]
    except IndexError:
        msg = "Could not find {} in valid area attributes".format(attr_name)
        raise ValueError(msg)

    # Obtain the weights
    weights = get_weights(k)

    # Get the lookup table
    attr_arr = get_attribute_array(p.stand_attribute_file, attr_name)

    # Open the neighbor rasters
    nn_rasters = [rasterio.open(x) for x in nn_files]

    # Bring in the boundary raster and the nonforest mask raster
    mask_raster = rasterio.open(p.boundary_raster)
    nf_raster = rasterio.open(p.mask_raster)

    # Retrieve the profile from the first neighbor raster
    profile = nn_rasters[0].profile

    # Determine the sub-window of the mask raster based on the
    # neighbor rasters.
    window = nn_rasters[0].window(*mask_raster.bounds)

    # Get the scalar for this variable
    scalar = scalars.get_scalar(attr_name.upper())

    # Open the output image for writing
    out_fn = "attribute_rasters/{}.tif".format(attr_name)
    out_raster = open_output_raster(profile, window, scalar, out_fn)

    # Run the process
    process_raster(
        nn_rasters,
        mask_raster,
        nf_raster,
        attr_arr,
        window,
        weights,
        scalar,
        out_raster,
    )
