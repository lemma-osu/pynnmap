import re

import numpy as np
from numpy.typing import NDArray

from ..misc import parser


class FootprintError(Exception):
    pass


class Footprint:
    def __init__(self, key: str, kernel: NDArray, cell_size: float):
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
            except AssertionError as e:
                err_msg = "Incorrect number of index pixels in kernel"
                raise FootprintError(err_msg) from e

        # Get offsets into array
        self.offsets = np.nonzero(np.logical_or(kernel == 1, kernel == 2))
        self.offsets = np.transpose(self.offsets)

    def __str__(self):
        out_str = ""
        out_str += self.__class__.__name__ + "\n"
        out_str += f'Key = "{self.key}"\n'
        out_str += f"Cellsize = {self.cell_size:.2f}\n"
        out_str += f"Number of rows = {self.n_rows}\n"
        out_str += f"Number of columns = {self.n_cols}\n"
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
        kernel = np.zeros((self.n_rows, self.n_cols), dtype=int)

        # Iterate through the offsets, turning on the correct pixels
        for offset in self.offsets:
            row, col = offset
            kernel[row, col] = 2 if np.all(offset == self.index) else 1

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

    def window(self, point: tuple[float, float]) -> tuple[float, float, int, int]:
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
    def parse(self, fp_file: str) -> dict[str, Footprint]:
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

        with open(fp_file) as fp_fh:
            all_lines = fp_fh.readlines()

        # Regular expression to match footprint specification starting lines
        fp_start = re.compile(r"^[A-Za-z0-9_]+\s+\d+\s+\d+\s+(\d+\.*\d*)$")

        # Get all footprints from and write them to individual Footprint
        # instances.  Push each of these to a footprint dictionary (fp_dict)
        fp_dict = {}
        chunks = self.read_chunks(
            all_lines, fp_start, self.blank_re, skip_lines=0, flush=True
        )
        for chunk in chunks:
            pixels = []
            key = None
            cell_size = None
            for i, line in enumerate(chunk):
                if i == 0:
                    parts = line.strip().split()
                    if len(parts) != 4:
                        raise ValueError(f"Invalid header format in chunk: {line}")
                    key, _, _, cell_size_str = parts
                    key = str(key)
                    try:
                        cell_size = float(cell_size_str)
                    except ValueError as exc:
                        raise ValueError(f"Invalid cell size: {cell_size_str}") from exc

                else:
                    pixels.append([int(x) for x in line.strip().split()])
            if key is None or cell_size is None:
                raise ValueError("Missing key or cell size for this chunk.")

            fp_dict[key] = Footprint(key, np.array(pixels), cell_size)

        return fp_dict
