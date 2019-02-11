import copy
from collections import defaultdict

import numpy as np
import pandas as pd
from osgeo import gdal, gdalconst

from pynnmap.core import imputation_model as im
from pynnmap.misc import footprint
from pynnmap.ordination_parser import lemma_ordination_parser


class NNPixel(object):
    def __init__(self, neighbors, distances):
        self.neighbors = np.copy(neighbors)
        self.distances = np.copy(distances)

    def __repr__(self):
        return '{kls}(\n neighbors={n}\n distances={d}\n)'.format(
            kls=self.__class__.__name__,
            n=self.neighbors[0:5],
            d=self.distances[0:5]
        )


class NNFootprint(object):
    def __init__(self, id_val):
        self.id = id_val
        self.pixels = []

    def __repr__(self):
        return '\n'.join([repr(x) for x in self.pixels])

    def append(self, pixel):
        self.pixels.append(pixel)


class NNFinder(object):
    def __init__(self, parameter_parser):
        self.parameter_parser = p = parameter_parser

        # Create a container for a dictionary of plot IDs (keys) to
        # complete neighbors and distances (values).
        self.neighbor_data = {}

        # Ensure the parameter parser is not a PROTOTYPE
        if p.parameter_set not in ('FULL', 'MINIMUM'):
            err_msg = 'Parameter set must be "FULL" or "MINIMUM"'
            raise ValueError(err_msg)

        # Check ID field
        if p.plot_id_field not in ('FCID', 'PLTID'):
            err_msg = p.id_field + ' accuracy assessment is not currently '
            err_msg += 'supported'
            raise NotImplementedError(err_msg)

        # Get footprint file
        self.fp_file = p.footprint_file

    def calculate_neighbors_at_ids(self, id_x_year):
        """
        Run ordination model over the list of IDs sent in and return neighbors
        and distances for each plot

        Parameters
        ----------
        id_x_year : dict
            Dictionary of plot IDs to associated imagery year to know what
            year to run the model

        Returns
        -------
        neighbor_data : dict
            Neighbor predictions
        """
        p = self.parameter_parser

        # Get the list of IDs, years, and the crosswalk of years to IDs
        id_vals, years, year_ids = self._get_year_id_crosswalk(id_x_year)

        # Retrieve coordinates from the model plots
        coord_list = self._get_model_plots(
            p.coordinate_file, id_vals, p.plot_id_field)

        # Retrieve the footprint configurations.  Footprint offsets store the
        # row and column tuples of each pixel within a given footprint.
        # Footprint windows store the upper left coordinate and window size for
        # extraction from GDAL datasets
        fp_offsets, fp_windows = self._parse_footprints(
            coord_list, p.footprint_file, p.plot_id_field)

        # Get the ordination model and read it in
        ord_file = p.get_ordination_file()
        lop = lemma_ordination_parser.LemmaOrdinationParser()
        ord_model = lop.parse(ord_file, delimiter=',')

        # Create the imputation model based on the ordination model and the
        # imputation parameters
        imp_model = im.ImputationModel(
            ord_model, n_axes=p.number_axes,
            use_weightings=p.use_axis_weighting, max_neighbors=p.max_neighbors
        )

        # Get information about environmental variables by year
        ord_year_var_dict, raster_counts, raster_dict = self._get_variable_info(
            years
        )

        # Extract environmental variable footprint data
        fp_value_dict = self._extract_footprints(
            raster_counts, raster_dict, years, fp_windows, year_ids,
            ord_year_var_dict)

        # Get neighbors for all plots
        return self._get_neighbors(
            years, fp_windows, fp_offsets, year_ids, ord_model,
            ord_year_var_dict, fp_value_dict, imp_model)

    def calculate_neighbors_cross_validation(self):
        """
        Wrapper around get_predicted_neighbors_at_ids optimized for cross-
        validation (ie. using plots that went into model development).
        """
        # Alias for self.parameter_parser
        p = self.parameter_parser

        # ID field
        id_field = p.plot_id_field

        # Get the plot/year crosswalk file and read the plot IDs
        # and image years into a dictionary
        xwalk_df = pd.read_csv(p.plot_year_crosswalk_file, low_memory=False)

        # Associate each plot with a model year; this is either the year the
        # model is associated with (for models that use imagery), or the
        # model_year (for models that don't use imagery)
        if p.model_type in p.imagery_model_types:
            s = pd.Series(xwalk_df.IMAGE_YEAR.values, index=xwalk_df[id_field])
        else:
            s = pd.Series(p.model_year, index=xwalk_df[id_field])
        id_x_year = s.to_dict()

        # Subset the plot ID list down to just those plots that went into
        # imputation.  This may be a subset of the plots that are in the
        # environmental matrix file based on running GNN in a unique way.
        # This requires parsing the model and extracting just the plot IDs
        ord_file = p.get_ordination_file()
        lop = lemma_ordination_parser.LemmaOrdinationParser()
        ord_model = lop.parse(ord_file, delimiter=',')
        plot_ids = ord_model.plot_ids
        id_x_year = dict(
            (i, id_x_year[i]) for i in id_x_year.keys() if i in plot_ids)

        # Call the main function
        return self.calculate_neighbors_at_ids(id_x_year)

    @staticmethod
    def _get_year_id_crosswalk(id_x_year):
        id_vals = np.unique(list(id_x_year.keys()))
        years = np.unique(list(id_x_year.values()))

        # Create a dictionary of all plots associated with each model year
        year_ids = defaultdict(list)
        for id_val, year in id_x_year.items():
            year_ids[year].append(id_val)
        return id_vals, years, year_ids

    # Retrieve all coordinates records as a data frame
    @staticmethod
    def _get_model_plots(coordinate_file, id_vals, id_field):
        coords = pd.read_csv(coordinate_file)
        id_arr = getattr(coords, id_field)
        return coords[np.isin(id_arr, id_vals)]

    @staticmethod
    def _parse_footprints(coord_list, fp_file, id_field):
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

    def _get_variable_info(self, years):
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
            ord_vars = self.parameter_parser.get_ordination_variables(year)

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

    def _extract_footprints(self, raster_counts, raster_dict, years, fp_windows,
                            year_ids, ord_year_var_dict):
        # Extract footprint information for every ordination variable that is
        # common to all years and store in a dict keyed by ID and raster
        # file name
        fp_value_dict = {}
        for fn, count in raster_counts.items():
            if count == len(years):
                print(fn)
                ds, processed = raster_dict[fn]

                # Get the footprint window values for this dataset
                fp_values = self._get_footprint_values(ds, fp_windows)

                # Change the flag for this dataset to 'processed'
                raster_dict[fn][1] = True

                # Store these footprint values in a dictionary keyed by
                # id and variable file name
                for (id_val, fp) in fp_values.items():
                    try:
                        fp_value_dict[id_val][fn] = fp
                    except KeyError:
                        fp_value_dict[id_val] = {}
                        fp_value_dict[id_val][fn] = fp

                # Close this dataset - no longer needed
                raster_dict[fn][0] = None

        # Main loop to iterate over all years
        for year in years:
            print(year)

            # Get the subset of footprint offsets and windows for this year
            windows = dict((x, fp_windows[x]) for x in year_ids[year])

            # Extract footprints for any variables that are not common to all
            # years, but specialized for this year
            for (var, fn) in ord_year_var_dict[year].items():
                ds, processed = raster_dict[fn]
                if not processed:
                    print(fn)

                    # Extract footprint values for this dataset
                    fp_values = self._get_footprint_values(ds, windows)

                    # Set the processed flag to True
                    raster_dict[fn][1] = True

                    # Store these values
                    for (id_val, fp) in fp_values.items():
                        try:
                            fp_value_dict[id_val][fn] = fp
                        except KeyError:
                            fp_value_dict[id_val] = {}
                            fp_value_dict[id_val][fn] = fp

                    # Close the dataset - no longer needed
                    raster_dict[fn][0] = None

        return fp_value_dict

    def _get_neighbors(self, years, fp_windows, fp_offsets, year_ids,
                       ord_model, ord_year_var_dict, fp_value_dict, imp_model):
        neighbor_data = {}
        for year in years:
            print(year)

            # Get the subset of footprint offsets and windows for this year
            offsets = dict((x, fp_offsets[x]) for x in year_ids[year])
            windows = dict((x, fp_windows[x]) for x in year_ids[year])

            # At this point, we have all the footprint information needed for
            # this year stored in fp_value_dict.  Now, iterate over each plot
            # in this year and run the imputation for each pixel.  Output is
            # captured at the pixel scale (within zonal_pixel_dict) and
            # for each attribute at the plot scale (within predicted_dict).
            for id_val in sorted(windows.keys()):

                # Get the footprint values for this plot
                fp_values = []
                for var in ord_model.var_names:
                    fn = ord_year_var_dict[year][var]
                    fp_values.append(fp_value_dict[id_val][fn])

                # Set up an output instance to capture each pixel's neighbors
                # and distances
                obj = NNFootprint(id_val)

                # Run the imputation for each pixel in the footprint
                for o in offsets[id_val]:
                    # Get the ordination variable values for this offset
                    # Store in (1xv) array
                    v = np.array(self._get_values_from_offset(fp_values, o))
                    v = v[np.newaxis, :]

                    # Run the imputation
                    nn_ids, nn_dists = imp_model.get_neighbors(v, id_val=id_val)

                    # Append this pixel to the NNFootprint object
                    obj.append(NNPixel(nn_ids, nn_dists))

                # Store the neighbor information
                neighbor_data[id_val] = copy.deepcopy(obj)

        return neighbor_data

    def _get_footprint_values(self, ds, windows, band=1):
        """
        Given a GDAL dataset and a dictionary of Footprint windows, extract
        all footprint values

        Parameters
        ----------
        ds : gdal.Dataset
            Ordination variable from which to extract footprint windows
        windows : dict
            Dict of footprint IDs (keys) to window specifications (values)
        band : int
            Band number of ds to extract (defaults to 1)

        Returns
        -------
        out_dict : dict
            Dict of footprint IDs (keys) to footprint values stored in 2d
            numpy arrays (values)
        """
        out_dict = {}
        gt = ds.GetGeoTransform()
        band = ds.GetRasterBand(band)
        for (k, v) in windows.items():
            out_dict[k] = self._get_footprint_value(v, band, gt)
        return out_dict

    @staticmethod
    def _get_footprint_value(window, band, gt):
        """
        Extract a footprint window from a GDAL band

        Parameters
        ----------
        window : tuple
            Window specification of upper left corner and footprint
            window size
        band : gdal.Band
            Band from which to extract data
        gt : tuple
            Geo-transform of band to go from (x, y) -> (row, col)

        Returns
        -------
        value : numpy 2d array
            Pixel values within window stored as 2d numpy array
        """
        (x_min, y_max, x_size, y_size) = window
        col = int((x_min - gt[0]) / gt[1])
        row = int((y_max - gt[3]) / gt[5])
        value = band.ReadAsArray(col, row, x_size, y_size)
        return value

    @staticmethod
    def _get_values_from_offset(footprints, offset):
        """
        Given a set of footprints representing ordination variables, extract
        the values associated with the given offset into the footprint window

        Parameters
        ----------
        footprints : list of numpy 2d arrays
            Footprint values for each ordination variables

        offset : tuple
            Row and column of offset for this pixel

        Returns
        -------
        out_list : list of
            Vector of ordination variable values for this pixel
        """
        out_list = []
        for fp in footprints:
            out_list.append(fp[offset[0], offset[1]])
        return out_list
