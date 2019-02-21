import re

import numpy as np

from pynnmap.misc import parser


class FootprintError(Exception):
    pass


class Footprint(object):

    def __init__(self, key, kernel, cell_size):
        self.key = key
        kernel = np.array(kernel)
        self.cell_size = cell_size
        self.n_rows, self.n_cols = kernel.shape

        # Get the location of the index pixel and raise an exception if one
        # isn't found
        try:
            self.index = np.transpose(np.nonzero(kernel == 2))
            assert len(self.index) == 1
            self.index = self.index[0]
        except AssertionError:
            try:
                self.index = np.transpose(np.nonzero(kernel == 3))
                assert len(self.index) == 1
                self.index = self.index[0]
            except AssertionError:
                err_msg = 'Incorrect number of index pixels in kernel'
                raise FootprintError(err_msg)

        # Get offsets into array
        self.offsets = np.nonzero(np.logical_or(kernel == 1, kernel == 2))
        self.offsets = np.transpose(self.offsets)

    def __str__(self):
        out_str = ''
        out_str += self.__class__.__name__ + '\n'
        out_str += 'Key = "' + self.key + '"\n'
        out_str += 'Cellsize = %.2f\n' % self.cell_size
        out_str += 'Number of rows = %d\n' % self.n_rows
        out_str += 'Number of columns = %d\n' % self.n_cols
        out_str += str(self.kernel())
        return out_str

    def kernel(self):
        """
        Based on the offsets, construct a two-dimensional numpy array
        representing the kernel for this footprint

        Returns
        -------
        kernel : np.array
            A 2D array of the footprint configuration
        """

        # Create a blank kernel the appropriate size
        kernel = np.zeros((self.n_rows, self.n_cols), dtype=np.int)

        # Iterate through the offsets, turning on the correct pixels
        for offset in self.offsets:
            row, col = offset
            if np.all(offset == self.index):
                kernel[row, col] = 2
            else:
                kernel[row, col] = 1

        # Ensure that the index pixel is not zero for footprints where the
        # index pixel is not part of the footprint
        if kernel[self.index[0], self.index[1]] == 0:
            kernel[self.index[0], self.index[1]] = 3
        return kernel

    def coords(self, point):
        """
        Given a 2D coordinate, return center points of all pixels in the
        footprint

        Parameters
        ----------
        point : tuple
            A tuple or list of the original 2D point

        Returns
        -------
        fp_coords : list of tuples
            A set of points representing the center points of footprint pixels
        """

        ref_x, ref_y = point
        index_row, index_col = self.index
        fp_coords = []
        for offset in self.offsets:
            x_off = (offset[1] - index_col) * self.cell_size + ref_x
            y_off = (index_row - offset[0]) * self.cell_size + ref_y
            fp_coords.append((x_off, y_off))
        return fp_coords

    def window(self, point):
        """
        Given a 2D coordinate, return the upper left corner coordinate and
        footprint window size.  This is used for more efficient extraction of
        the footprint from a raster

        Parameters
        ----------
        point : tuple
            A tuple or list of the original 2D point

        Returns
        -------
        window_params : tuple
            A tuple representing, in order, x_min, y_max, x_size, y_size
        """

        # Offsets from index pixel
        ref_x, ref_y = point
        index_row, index_col = self.index
        x_min = ref_x - (index_col * self.cell_size)
        y_max = ref_y + (index_row * self.cell_size)
        return x_min, y_max, self.n_cols, self.n_rows


class FootprintParser(parser.Parser):

    def __init__(self):
        super(FootprintParser, self).__init__()

    def parse(self, fp_file):
        """
        Parse a footprint file and return the set of all footprints

        Parameters
        ----------
        fp_file : str
            File containing all footprint configurations

        Returns
        -------
        fp_dict : dict of Footprints
        """

        # Open the footprint file and read in all the lines
        fp_fh = open(fp_file, 'r')
        all_lines = fp_fh.readlines()
        fp_fh.close()

        # Regular expression to match footprint specification starting lines
        fp_start = re.compile('^[A-Za-z0-9_]+\s+\d+\s+\d+\s+(\d+\.*\d*)$')

        # Get all footprints from and write them to individual Footprint
        # instances.  Push each of these to a footprint dictionary (fp_dict)
        fp_dict = {}
        chunks = self.read_chunks(
            all_lines, fp_start, self.blank_re, skip_lines=0, flush=True)
        for chunk in chunks:
            pixels = []
            for (i, line) in enumerate(chunk):
                if i == 0:
                    (key, n_rows, n_cols, cell_size) = line.strip().split()
                else:
                    pixels.append([int(x) for x in line.strip().split()])
            fp_dict[key] = Footprint(key, np.array(pixels), float(cell_size))

        # Return this dictionary
        return fp_dict
