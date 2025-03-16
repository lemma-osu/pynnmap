from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
from numpy.typing import NDArray
from rasterio.windows import Window

from ..misc import footprint
from ..ordination_parser import lemma_ordination_parser
from . import get_id_year_crosswalk
from .imputation_model import ImputationModel


@dataclass
class DatasetStatus:
    """
    Simple dataclass to store the handle of a rasterio.Dataset and
    whether it has been processed for all footprints.
    """

    dataset: rasterio.DatasetReader
    processed: bool


def get_year_id_crosswalk(
    id_x_year: dict[int, int],
) -> tuple[NDArray, NDArray, dict[int, list[int]]]:
    """
    Create a dictionary of all plots associated with each model year.
    """
    id_vals, years = id_x_year.keys(), id_x_year.values()
    id_vals, years = map(lambda x: np.unique(list(x)), (id_vals, years))
    year_ids = defaultdict(list)
    for id_val, year in id_x_year.items():
        year_ids[year].append(id_val)
    return id_vals, years, year_ids


def get_model_plots(
    coordinate_file: str, id_vals: NDArray, id_field: str
) -> pd.DataFrame:
    """
    Get the coordinates for the plots in id_vals.
    """
    coords = pd.read_csv(coordinate_file)
    id_arr = getattr(coords, id_field)
    return coords[np.isin(id_arr, id_vals)]


def parse_footprints(
    coordinate_df: pd.DataFrame, footprint_path: str, id_field: str
) -> dict[int, tuple[float, float, int, int]]:
    """
    Parse the footprint file and return a dictionary of footprint windows
    in the format of (x_min, y_max, x_size, y_size) keyed by ID.
    """
    footprint_parser = footprint.FootprintParser()
    footprint_dict = footprint_parser.parse(footprint_path)
    footprint_windows = {}
    for rec in coordinate_df.itertuples():
        id_val = getattr(rec, id_field)
        ds, x, y = rec.DATA_SOURCE, rec.X_COORD, rec.Y_COORD
        footprint_windows[id_val] = footprint_dict[ds].window((x, y))
    return footprint_windows


def get_variable_info(
    parser, years: NDArray
) -> tuple[
    dict[int, dict[str, str]],
    dict[str, int],
    dict[str, DatasetStatus],
]:
    """
    Get information about the ordination variables for each year.
    """
    # This section extracts the ordination variable information from the
    # model XML files and creates a dict of year/variable combinations.
    # Once this dict is created, we only need to extract the spatial data
    # from the unique set of values in this dict and use this crosswalk
    # to get to those values.  This should be efficient from a spatial processing
    # perspective to avoid cache thrashing.
    #
    # However, because we don't need all ordination variable's values for
    # all plots (ie. temporally varying ordination variables), at this
    # point we only want to extract footprints for those variables that are
    # common across all years.  We track the count of times a variable
    # appears across all lists (raster_counts) and if equal to
    # len(years), we extract footprints at this point.
    #
    # For all other variables, we wait until we have a subset of the coords
    # to extract the spatial data
    ordination_year_variable_dict: dict[int, dict[str, str]] = {}
    raster_counts: dict[str, int] = {}
    raster_dict: dict[str, DatasetStatus] = {}

    for year in years:
        ordination_year_variable_dict[year] = {}

        # Get the ordination variables specialized for this year
        ord_vars = parser.get_ordination_variables(year)

        for var, path in ord_vars:
            # For this year, variable combination, store the path to the
            # variable
            ordination_year_variable_dict[year][var] = path

            # Record this variable in the counts and push to the raster
            # list if it's a new variable
            try:
                raster_counts[path] += 1
            except KeyError:
                ds = rasterio.open(path)
                raster_dict[path] = DatasetStatus(ds, False)
                raster_counts[path] = 1
    return ordination_year_variable_dict, raster_counts, raster_dict


def get_footprint_values(
    ds: rasterio.DatasetReader,
    windows: dict[int, tuple[float, float, int, int]],
    band: int = 1,
) -> dict[int, NDArray]:
    """
    Get the footprint values for a given dataset and set of windows.
    """
    return {k: get_footprint_value(v, ds, band=band) for k, v in windows.items()}


def get_footprint_value(
    window: tuple[float, float, int, int], ds: rasterio.DatasetReader, band: int = 1
) -> NDArray:
    """
    Get the footprint value for a given dataset and window.
    """
    x_min, y_max, x_size, y_size = window
    row, col = ds.index(x_min, y_max)
    return ds.read(band, window=Window(col, row, x_size, y_size))


class EnvironmentalVector:
    def __init__(self, d: dict[str, float]):
        self.d = d

    def __repr__(self):
        return "\n".join((f"{k}: {v}" for k, v in self.d.items()))


def extract_footprints(
    raster_counts: dict[str, int],
    raster_dict: dict[str, DatasetStatus],
    years: NDArray,
    fp_windows: dict[int, tuple[float, float, int, int]],
    year_ids: dict[int, list[int]],
    ord_year_var_dict: dict[int, dict[str, str]],
) -> dict[int, list[EnvironmentalVector]]:
    """
    Extract the footprints for all environmental variables across all plot
    windows.
    """
    # Create a reverse dictionary of path name to variable name / year.
    path_to_var = defaultdict(set)
    for year, variable_information in ord_year_var_dict.items():
        for variable, path_name in variable_information.items():
            path_to_var[path_name].add((variable, year))

    # Extract footprint information for every ordination variable that is
    # common to all years and store in a dict keyed by ID and raster
    # file name
    fp_value_dict: dict[int, dict[str, float]] = defaultdict(dict)
    for fn, count in raster_counts.items():
        if count == len(years):
            print(fn)

            # Get the footprint window values for this dataset
            fp_values = get_footprint_values(raster_dict[fn].dataset, fp_windows)

            # Change the processed flag for this dataset to True
            raster_dict[fn].processed = True

            # Store these footprint values in a dictionary keyed by
            # id and variable file name
            unique_variables = {x[0] for x in path_to_var[fn]}
            for variable in unique_variables:
                for id_val, fp in fp_values.items():
                    fp_value_dict[id_val][variable] = fp

            # Close this dataset - no longer needed
            raster_dict[fn].dataset.close()

    # Main loop to iterate over all years
    for year in years:
        print(year)

        # Get the subset of footprint offsets and windows for this year
        windows = {x: fp_windows[x] for x in year_ids[year]}

        # Extract footprints for any variables that are not common to all
        # years, but specialized for this year
        for fn in ord_year_var_dict[year].values():
            if not raster_dict[fn].processed:
                print(fn)

                # Extract footprint values for this dataset
                fp_values = get_footprint_values(raster_dict[fn].dataset, windows)

                # Store these values - it's possible that a path is used more
                # than once for different variables.  Ensure that the values
                # extracted are used only for the correct variable and year
                unique_variables = {x[0] for x in path_to_var[fn]}
                for variable in unique_variables:
                    variable_years = {x[1] for x in path_to_var[fn] if x[0] == variable}
                    if year in variable_years:
                        for id_val, fp in fp_values.items():
                            fp_value_dict[id_val][variable] = fp

    # Close the temporally varying datasets
    datasets = [x.dataset for x in raster_dict.values()]
    for dataset in datasets:
        if not dataset.closed:
            dataset.close()

    # Reorder this such that each ID has a list of EnvironmentalVectors
    # associated with it keyed by variable name
    env_dict = defaultdict(list)
    for id_val, val in fp_value_dict.items():
        pixel_data: dict[int, dict[str, float]] = defaultdict(dict)
        for var, fp in val.items():
            arr = fp.flatten()
            for i, pixel_val in enumerate(arr):
                pixel_data[i][var] = pixel_val
        for d in pixel_data.values():
            env_dict[id_val].append(EnvironmentalVector(d))
    return env_dict


class NNPixel:
    def __init__(self, neighbors: NDArray, distances: NDArray) -> None:
        self.neighbors = np.copy(neighbors)
        self.distances = np.copy(distances)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}\n"
            f"(neighbors={self.neighbors[:5]}\n"
            f"distances={self.distances[:5]}\n)"
        )


class NNFootprint:
    def __init__(self, id_val: int) -> None:
        self.id = id_val
        self.pixels: list[NNPixel] = []

    def __repr__(self):
        return "\n".join([repr(x) for x in self.pixels])

    def append(self, pixel: NNPixel) -> None:
        self.pixels.append(pixel)


class NNFinder(ABC):
    def __init__(self, parameter_parser):
        self.parameter_parser = p = parameter_parser

        # Create a container for a dictionary of plot IDs (keys) to
        # complete neighbors and distances (values).
        self.neighbor_data = {}

        # Ensure the parameter parser is not a PROTOTYPE
        if p.parameter_set not in ("FULL", "MINIMUM"):
            err_msg = 'Parameter set must be "FULL" or "MINIMUM"'
            raise ValueError(err_msg)

        # Check ID field
        if p.plot_id_field not in ("FCID", "PLTID"):
            err_msg = p.id_field + " accuracy assessment is not currently "
            err_msg += "supported"
            raise NotImplementedError(err_msg)

    def calculate_neighbors_cross_validation(self) -> dict[int, NNFootprint]:
        """
        Wrapper around get_predicted_neighbors_at_ids optimized for cross-
        validation (ie. using plots that went into model development).
        """
        # Alias for self.parameter_parser
        p = self.parameter_parser

        # Subset the plot ID list down to just those plots that went into
        # imputation.  This may be a subset of the plots that are in the
        # environmental matrix file based on running GNN in a unique way.
        # This requires parsing the model and extracting just the plot IDs
        lop = lemma_ordination_parser.LemmaOrdinationParser()
        ord_model = lop.parse(p.get_ordination_file())
        plot_ids = ord_model.plot_ids

        # Call the main function
        return self.calculate_neighbors_at_ids(plot_ids)

    def calculate_neighbors_at_ids(self, plot_ids: NDArray) -> dict[int, NNFootprint]:
        p = self.parameter_parser

        # Get the ordination model
        lop = lemma_ordination_parser.LemmaOrdinationParser()
        ordination_model = lop.parse(p.get_ordination_file())

        # Create the imputation model based on the ordination model and the
        # imputation parameters
        imputation_model = ImputationModel(
            ordination_model,
            n_axes=p.number_axes,
            use_weightings=p.use_axis_weighting,
            max_neighbors=p.max_neighbors,
        )

        # Get the environmental data associated with all plots
        env_data = self.get_environmental_data(plot_ids)

        # Get neighbors for all plots
        variable_names = [str(x) for x in ordination_model.var_names]
        return self.get_neighbors(env_data, imputation_model, variable_names)

    @abstractmethod
    def get_environmental_data(
        self, plot_ids: NDArray
    ) -> dict[int, list[EnvironmentalVector]]:
        pass

    @staticmethod
    def get_neighbors(
        env_data: dict[int, list[EnvironmentalVector]],
        imputation_model: ImputationModel,
        variable_names: list[str],
    ) -> dict[int, NNFootprint]:
        """
        Return neighbors and distances for all pixels in env_data given
        the imputation model.
        """
        neighbor_data = {}
        for id_val in sorted(env_data.keys()):
            # Set up an output instance to capture each pixel's neighbors
            # and distances
            obj = NNFootprint(id_val)

            # Run over observations for this plot
            for ev in env_data[id_val]:
                v = np.array([ev.d[x] for x in variable_names])
                v = v[np.newaxis, :]

                # Run the imputation
                nn_ids, nn_dists = imputation_model.get_neighbors(v, id_val=id_val)

                # Append this pixel to the NNFootprint object
                obj.append(NNPixel(nn_ids, nn_dists))

            # Store the neighbor information
            neighbor_data[id_val] = copy.deepcopy(obj)
        return neighbor_data


class PixelNNFinder(NNFinder):
    def get_environmental_data(
        self, plot_ids: NDArray
    ) -> dict[int, list[EnvironmentalVector]]:
        """
        Get the environmental data for each plot ID at the pixel scale by
        extracting the footprint data for each plot from model covariates.
        """
        p = self.parameter_parser

        # Crosswalk of ID to year
        id_x_year = get_id_year_crosswalk(p)
        id_x_year = {i: id_x_year[i] for i in id_x_year if i in plot_ids}

        # Get the list of IDs, years, and the crosswalk of years to IDs
        id_vals, years, year_ids = get_year_id_crosswalk(id_x_year)

        # Retrieve coordinates from the model plots
        coord_list = get_model_plots(p.coordinate_file, id_vals, p.plot_id_field)

        # Retrieve the footprint configurations.  Footprint offsets store the
        # row and column tuples of each pixel within a given footprint.
        # Footprint windows store the upper left coordinate and window size for
        # extraction from GDAL datasets
        fp_windows = parse_footprints(coord_list, p.footprint_file, p.plot_id_field)

        # Get information about environmental variables by year
        ord_year_var_dict, raster_counts, raster_dict = get_variable_info(p, years)

        # Extract environmental variable footprint data
        return extract_footprints(
            raster_counts,
            raster_dict,
            years,
            fp_windows,
            year_ids,
            ord_year_var_dict,
        )


class PlotNNFinder(NNFinder):
    def get_environmental_data(
        self, plot_ids: NDArray
    ) -> dict[int, list[EnvironmentalVector]]:
        """
        Get the environmental data for each plot ID at the plot scale by
        using the values in the environmental matrix file.
        """
        # Get the environmental matrix and subset down to plot_ids
        p = self.parameter_parser
        env_df = pd.read_csv(p.environmental_matrix_file)
        plot_df = pd.DataFrame({p.plot_id_field: plot_ids})
        env_df = plot_df.merge(env_df, on=p.plot_id_field)

        # Create an EnvironmentalVector for each ID
        return {
            row[p.plot_id_field]: [EnvironmentalVector(row)]
            for row in env_df.to_dict(orient="records")
        }
