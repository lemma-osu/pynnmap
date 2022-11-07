import copy
from abc import ABC, abstractmethod
from collections import defaultdict

import numpy as np
import pandas as pd
from osgeo import gdal, gdalconst

from pynnmap.core import get_id_year_crosswalk
from pynnmap.core import imputation_model as im
from pynnmap.misc import footprint
from pynnmap.ordination_parser import lemma_ordination_parser


def get_year_id_crosswalk(id_x_year):
    # Create a dictionary of all plots associated with each model year
    id_vals, years = id_x_year.keys(), id_x_year.values()
    id_vals, years = map(lambda x: np.unique(list(x)), (id_vals, years))
    year_ids = defaultdict(list)
    for id_val, year in id_x_year.items():
        year_ids[year].append(id_val)
    return id_vals, years, year_ids


def get_model_plots(coordinate_file, id_vals, id_field):
    coords = pd.read_csv(coordinate_file)
    id_arr = getattr(coords, id_field)
    return coords[np.isin(id_arr, id_vals)]


def parse_footprints(coord_list, fp_file, id_field):
    fp_parser = footprint.FootprintParser()
    fp_dict = fp_parser.parse(fp_file)
    fp_offsets = {}
    fp_windows = {}
    for rec in coord_list.itertuples():
        id_val = getattr(rec, id_field)
        ds, x, y = rec.DATA_SOURCE, rec.X_COORD, rec.Y_COORD
        fp_offsets[id_val] = fp_dict[ds].offsets
        fp_windows[id_val] = fp_dict[ds].window((x, y))
    return fp_offsets, fp_windows


def get_variable_info(parser, years):
    # This section extracts the ordination variable information from the
    # model XML files and creates a dict of year/variable combinations.
    # Once this dict is created, we only need to extract the spatial data
    # from the unique set of values in this dict and use this crosswalk
    # to get to those values.  This should be efficient from GDAL's
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
    ord_year_var_dict = {}
    raster_counts = {}
    raster_dict = {}

    for year in years:
        ord_year_var_dict[year] = {}

        # Get the ordination variables specialized for this year
        ord_vars = parser.get_ordination_variables(year)

        for var, path in ord_vars:
            # For this year, variable combination, store the path to the
            # variable
            ord_year_var_dict[year][var] = path

            # Record this variable in the counts and push to the raster
            # list if it's a new variable
            try:
                raster_counts[path] += 1
            except KeyError:
                ds = gdal.Open(path, gdalconst.GA_ReadOnly)
                raster_dict[path] = [ds, False]
                raster_counts[path] = 1
    return ord_year_var_dict, raster_counts, raster_dict


def get_footprint_values(ds, windows, band=1):
    gt = ds.GetGeoTransform()
    band = ds.GetRasterBand(band)
    return {k: get_footprint_value(v, band, gt) for k, v in windows.items()}


def get_footprint_value(window, band, gt):
    x_min, y_max, x_size, y_size = window
    col = int((x_min - gt[0]) / gt[1])
    row = int((y_max - gt[3]) / gt[5])
    return band.ReadAsArray(col, row, x_size, y_size)


def extract_footprints(
    raster_counts,
    raster_dict,
    years,
    fp_windows,
    year_ids,
    ord_year_var_dict,
):
    # Create a reverse dictionary of path name to variable name
    # There will be multiple identical entries for non-temporally varying
    # attributes - they will be overwritten on each pass
    path_to_var = defaultdict(str)
    for year, d in ord_year_var_dict.items():
        for var, fn in d.items():
            path_to_var[fn] = var

    # Extract footprint information for every ordination variable that is
    # common to all years and store in a dict keyed by ID and raster
    # file name
    fp_value_dict = defaultdict(dict)
    for fn, count in raster_counts.items():
        var = path_to_var[fn]
        if count == len(years):
            print(fn)
            ds, processed = raster_dict[fn]

            # Get the footprint window values for this dataset
            fp_values = get_footprint_values(ds, fp_windows)

            # Change the flag for this dataset to 'processed'
            raster_dict[fn][1] = True

            # Store these footprint values in a dictionary keyed by
            # id and variable file name
            for id_val, fp in fp_values.items():
                fp_value_dict[id_val][var] = fp

            # Close this dataset - no longer needed
            raster_dict[fn][0] = None

    # Main loop to iterate over all years
    for year in years:
        print(year)

        # Get the subset of footprint offsets and windows for this year
        windows = {x: fp_windows[x] for x in year_ids[year]}

        # Extract footprints for any variables that are not common to all
        # years, but specialized for this year
        for _, fn in ord_year_var_dict[year].items():
            var = path_to_var[fn]
            ds, processed = raster_dict[fn]
            if not processed:
                print(fn)

                # Extract footprint values for this dataset
                fp_values = get_footprint_values(ds, windows)

                # Set the processed flag to True
                raster_dict[fn][1] = True

                # Store these values
                for id_val, fp in fp_values.items():
                    fp_value_dict[id_val][var] = fp

                # Close the dataset - no longer needed
                raster_dict[fn][0] = None

    # Reorder this such that each ID has a list of EnvironmentalVectors
    # associated with it keyed by variable name
    env_dict = defaultdict(list)
    for id_val, val in fp_value_dict.items():
        pixel_data = defaultdict(dict)
        for var, fp in val.items():
            arr = fp.flatten()
            for i, pixel_val in enumerate(arr):
                pixel_data[i][var] = pixel_val
        for i, d in pixel_data.items():
            env_dict[id_val].append(EnvironmentalVector(d))
    return env_dict


class EnvironmentalVector:
    def __init__(self, d):
        self.d = d

    def __repr__(self):
        return "\n".join((f"{k}: {v}" for k, v in self.d.items()))


class NNPixel(object):
    def __init__(self, neighbors, distances):
        self.neighbors = np.copy(neighbors)
        self.distances = np.copy(distances)

    def __repr__(self):
        return "{kls}(\n neighbors={n}\n distances={d}\n)".format(
            kls=self.__class__.__name__,
            n=self.neighbors[:5],
            d=self.distances[:5],
        )


class NNFootprint(object):
    def __init__(self, id_val):
        self.id = id_val
        self.pixels = []

    def __repr__(self):
        return "\n".join([repr(x) for x in self.pixels])

    def append(self, pixel):
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

    def calculate_neighbors_cross_validation(self):
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

    def calculate_neighbors_at_ids(self, plot_ids):
        p = self.parameter_parser

        # Get the ordination model
        lop = lemma_ordination_parser.LemmaOrdinationParser()
        ord_model = lop.parse(p.get_ordination_file())

        # Create the imputation model based on the ordination model and the
        # imputation parameters
        imp_model = im.ImputationModel(
            ord_model,
            n_axes=p.number_axes,
            use_weightings=p.use_axis_weighting,
            max_neighbors=p.max_neighbors,
        )

        # Get the environmental data associated with all plots
        env_data = self.get_environmental_data(plot_ids)

        # Get neighbors for all plots
        return self.get_neighbors(env_data, ord_model, imp_model)

    @abstractmethod
    def get_environmental_data(self, plot_ids):
        pass

    @staticmethod
    def get_neighbors(env_data, ord_model, imp_model):
        var_names = ord_model.var_names
        neighbor_data = {}
        for id_val in sorted(env_data.keys()):
            # Set up an output instance to capture each pixel's neighbors
            # and distances
            obj = NNFootprint(id_val)

            # Run over observations for this plot
            for ev in env_data[id_val]:
                v = np.array([ev.d[x] for x in var_names])
                v = v[np.newaxis, :]

                # Run the imputation
                nn_ids, nn_dists = imp_model.get_neighbors(v, id_val=id_val)

                # Append this pixel to the NNFootprint object
                obj.append(NNPixel(nn_ids, nn_dists))

            # Store the neighbor information
            neighbor_data[id_val] = copy.deepcopy(obj)
        return neighbor_data


class PixelNNFinder(NNFinder):
    def get_environmental_data(self, plot_ids):
        p = self.parameter_parser

        # Crosswalk of ID to year
        id_x_year = get_id_year_crosswalk(p)
        id_x_year = {i: id_x_year[i] for i in id_x_year.keys() if i in plot_ids}

        # Get the list of IDs, years, and the crosswalk of years to IDs
        id_vals, years, year_ids = get_year_id_crosswalk(id_x_year)

        # Retrieve coordinates from the model plots
        coord_list = get_model_plots(
            p.coordinate_file, id_vals, p.plot_id_field
        )

        # Retrieve the footprint configurations.  Footprint offsets store the
        # row and column tuples of each pixel within a given footprint.
        # Footprint windows store the upper left coordinate and window size for
        # extraction from GDAL datasets
        fp_offsets, fp_windows = parse_footprints(
            coord_list, p.footprint_file, p.plot_id_field
        )

        # Get information about environmental variables by year
        ord_year_var_dict, raster_counts, raster_dict = get_variable_info(
            p, years
        )

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
    def get_environmental_data(self, plot_ids):
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
