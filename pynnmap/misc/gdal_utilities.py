import math

from osgeo import gdal


class RasterBlock:
    """
    Class to represent a chunk of a GDAL Band.  Typically, these are generated
    from the generator get_blocks() which reads on block boundaries.
    However, it should be generic enough to use with any window.
    """

    def __init__(self, band, x_size, y_size, x_offset, y_offset):
        """
        Create an instance using a valid GDAL Band and window information.
        Stores the pixel information in the self.data property

        Parameters
        ----------
        band : gdal.Band
            A raster band from a valid gdal.Dataset

        x_size : int
            Number of columns to extract

        y_size : int
            Number of rows to extract

        x_offset : int
            Column offset from upper-left corner for extraction window

        y_offset : int
            Row offset from upper-left corner for extraction window
        """
        self.data = band.ReadAsArray(x_offset, y_offset, x_size, y_size)
        self.x_size = x_size
        self.y_size = y_size
        self.x_offset = x_offset
        self.y_offset = y_offset

    def __repr__(self):
        """
        Return a string representation of this object
        """
        return_str = ''
        return_str += '\nBlock origin: ('
        return_str += repr(self.x_offset) + ', ' + repr(self.y_offset) + ')'
        return_str += '\nBlock size: ('
        return_str += repr(self.x_size) + ', ' + repr(self.y_size) + ')'
        return_str += '\nData:'
        return_str += '\n' + repr(self.data)
        return return_str


def get_blocks(band):
    """
    A generator which returns all blocks from a gdal.Band using block
    boundaries for efficient reads

    Parameters
    ----------
    band : gdal.Band
        A raster band from a valid gdal.Dataset

    Yields
    -------
    out : RasterBlock instance
        The current raster block within the band
    """
    # Get the block size for this band
    (x_block_size, y_block_size) = band.GetBlockSize()

    # Calculate how many blocks are needed to cover the band
    x_blocks = (band.XSize + (x_block_size - 1)) / x_block_size
    y_blocks = (band.YSize + (y_block_size - 1)) / y_block_size

    # Get the size of the band in pixels
    x_grid_size = band.XSize
    y_grid_size = band.YSize

    for j in range(y_blocks):
        for k in range(x_blocks):
            if (k + 1) * x_block_size <= x_grid_size:
                x_size = x_block_size
            else:
                x_size = x_grid_size - (k * x_block_size)

            if (j + 1) * y_block_size <= y_grid_size:
                y_size = y_block_size
            else:
                y_size = y_grid_size - (j * y_block_size)

            x_offset = k * x_block_size
            y_offset = j * y_block_size

            yield RasterBlock(band, x_size, y_size, x_offset, y_offset)


def get_chunks(band, max_size=250000000):
    """
    A generator which returns all 'chunks' for a gdal.Band using whole
    rows.  The largest group of rows that is under max_size is
    returned

    Parameters
    ----------
    band : gdal.Band
        A raster band from a valid gdal.Dataset

    Keywords
    --------
    max_size : int
        The maximum size (in bytes) of the chunk to be returned

    Yields
    ------
    out : RasterBlock instance
        The current raster chunk within the band
    """
    # Calculate the size of a row's worth of data.  All chunks should be
    # complete rows.
    x_chunk_size = band.XSize
    x_offset = 0
    num_bytes_row = x_chunk_size * gdal.GetDataTypeSize(band.DataType)

    # Now calculate how many rows to retrieve per call
    y_chunk_size = int(math.floor(float(max_size) / num_bytes_row))

    # Figure out how many batches this represents
    y_grid_size = band.YSize
    y_chunks = (y_grid_size + (y_chunk_size - 1)) / y_chunk_size

    for j in range(y_chunks):
        if (j + 1) * y_chunk_size <= y_grid_size:
            y_size = y_chunk_size
        else:
            y_size = y_grid_size - (j * y_chunk_size)
        y_offset = j * y_chunk_size

        yield RasterBlock(band, x_chunk_size, y_size, x_offset, y_offset)


def get_chunks_from_raster_envelope(rds, re, max_size=250000000):
    """
    A generator which returns all 'chunks' for a RasterDataset within a
    raster envelope using whole rows.  The largest group of rows
    that is under max_size is returned

    Parameters
    ----------
    rds : RasterDataset
        A raster band from a valid gdal.Dataset

    re : RasterEnvelope
        A raster envelope which should be a subset of the band window

    Keywords
    --------
    max_size : int
        The maximum size (in bytes) of the chunk to be returned

    Yields
    ------
    out : RasterBlock instance
        The current raster chunk within the band
    """

    # Get the offsets from the rds envelope to the re envelope
    rd_env = rds.env
    rd_x_off = int((re.x_min - rd_env.x_min) / rd_env.cell_size)
    rd_y_off = int((rd_env.y_max - re.y_max) / rd_env.cell_size)

    # Calculate the size of a row's worth of data.  All chunks should be
    # complete rows.
    x_chunk_size = re.n_cols
    x_offset = rd_x_off
    num_bytes_row = x_chunk_size * gdal.GetDataTypeSize(rds.data_type)

    # Now calculate how many rows to retrieve per call
    y_chunk_size = int(math.floor(float(max_size) / num_bytes_row))

    # Figure out how many batches this represents
    y_grid_size = re.n_rows
    y_chunks = (y_grid_size + (y_chunk_size - 1)) / y_chunk_size

    for j in range(y_chunks):
        if (j + 1) * y_chunk_size <= y_grid_size:
            y_size = y_chunk_size
        else:
            y_size = y_grid_size - (j * y_chunk_size)
        y_offset = rd_y_off + (j * y_chunk_size)

        yield RasterBlock(rds.rb, x_chunk_size, y_size, x_offset, y_offset)


def compare_datasets(ds_01, ds_02, skip=()):
    """
    Compare features of two different GDAL datasets in order to determine
    whether or not they can be processed together.  Optional keyword
    argument skip takes a tuple or list of features not to test.  These
    should be the following keyword arguments:

      'geotransform' : Don't compare geotransforms between datasets
      'projection' : Don't compare projections between datasets
      'dimension' : Don't compare dimensions (rows, columns) between datasets
      'data_type' : Don't compare data types between datasets
      'nodata_value' : Don't compare nodata values between datasets

    Parameters
    ----------
    ds_01 : gdal.Dataset
        Dataset #1 to compare

    ds_02 : gdal.Dataset
        Dataset #2 to compare

    skip : tuple (or list) of strings
        The set of tests to skip.  Defaults to 'None'

    Returns
    -------
    compare : bool
        The 'AND'ed comparison of all tests run

    failed_tests : tuple of strings
        If any tests fail, the tuple contains the set of failed tests.
        Returns empty if compare returns True
    """

    all_tests = {
        'geotransform': compare_geotransform,
        'projection': compare_projection,
        'dimension': compare_dimension,
        'data_type': compare_data_type,
        'nodata_value': compare_nodata_value,
    }

    failed_tests = []
    for (kw, fcn) in all_tests.iteritems():
        if kw not in skip:
            result = fcn(ds_01, ds_02)
            if not result:
                failed_tests.append(kw)

    return len(failed_tests) == 0, tuple(failed_tests)


def compare_geotransform(ds_01, ds_02):
    """
    Compare geotransforms between two datasets
    """
    return ds_01.GetGeoTransform() == ds_02.GetGeoTransform()


def compare_projection(ds_01, ds_02):
    """
    Compare projections between two datasets
    """
    return ds_01.GetProjection() == ds_02.GetProjection()


def compare_dimension(ds_01, ds_02):
    """
    Compare dimensions between two datasets
    """
    xsize_01, ysize_01 = ds_01.RasterXSize, ds_01.RasterYSize
    xsize_02, ysize_02 = ds_02.RasterXSize, ds_02.RasterYSize
    return xsize_01 == xsize_02 and ysize_01 == ysize_02


def compare_data_type(ds_01, ds_02):
    """
    Compare data types between two datasets.  This compares the first band
    from each dataset with the assumption that data type stays consistent
    between bands
    """
    rb_01 = ds_01.GetRasterBand(1)
    rb_02 = ds_02.GetRasterBand(1)
    return rb_01.DataType == rb_02.DataType


def compare_nodata_value(ds_01, ds_02):
    """
    Compare no data values between two datasets.  This compares the first band
    from each dataset with the assumption that nodata value stays consistent
    between bands
    """
    rb_01 = ds_01.GetRasterBand(1)
    rb_02 = ds_02.GetRasterBand(1)
    return rb_01.GetNoDataValue() == rb_02.GetNoDataValue()
