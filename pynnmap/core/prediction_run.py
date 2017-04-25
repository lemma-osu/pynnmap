import copy

import numpy as np
from osgeo import gdal, gdalconst

from pynnmap.core import imputation_model as im
from pynnmap.misc import footprint
from pynnmap.misc import utilities
from pynnmap.ordination_parser import lemma_ordination_parser
from pynnmap.parser import xml_stand_metadata_parser as xsmp

# Minimum distance constant
MIN_DIST = 0.000000000001


class NNPixel(object):

    def __init__(self, neighbors, distances):
        self.neighbors = np.copy(neighbors)
        self.distances = np.copy(distances)


class NNFootprint(object):

    def __init__(self, id_val):
        self.id = id_val
        self.pixels = []

    def append(self, pixel):
        self.pixels.append(pixel)


class PixelPrediction(object):
    """
    Class to hold a given pixel's prediction including neighbor IDs, distances
    and predicted values for each continuous attribute.
    """
    def __init__(self, id_val, pixel_number, k):
        self.id = id_val
        self.pixel_number = pixel_number
        self.k = k
        self._predicted_attr = {}
        self.neighbors = None
        self.distances = None

    def neighbors(self, neighbors):
        self.neighbors = neighbors

    def distances(self, distances):
        self.distances = distances

    def get_predicted_attr(self, attr):
        return self._predicted_attr[attr]

    def set_predicted_attr(self, attr, value):
        self._predicted_attr[attr] = value


class PredictionOutput(object):

    def __init__(self, prediction_run):
        self.prediction_run = prediction_run
        self.parameter_parser = prediction_run.parameter_parser
        self.id_field = self.parameter_parser.plot_id_field

    def open_prediction_files(self, zonal_pixel_file, predicted_file):
        """
        Open the prediction files and write the header lines.  Return
        file handles to the caller

        Parameters
        ----------
        zonal_pixel_file : str
            Name of the zonal_pixel_file to open

        predicted_file : str
            Name of the predicted file to open

        Returns
        -------
        zonal_pixel_fh : file
            File handle of the zonal_pixel_file

        predicted_fh : file
            File handle of the predicted file
        """

        # Open the zonal pixel file and write the header line
        zonal_pixel_fh = open(zonal_pixel_file, 'w')
        header_fields = (
            self.id_field,
            'PIXEL_NUMBER',
            'NEIGHBOR',
            'NEIGHBOR_ID',
            'DISTANCE',
        )
        zonal_pixel_fh.write(','.join(header_fields) + '\n')

        # Open the prediction file and write the header line
        pr = self.prediction_run
        predicted_fh = open(predicted_file, 'w')
        predicted_fh.write(self.id_field + ',' + ','.join(pr.attrs) + '\n')

        return zonal_pixel_fh, predicted_fh


class PredictionRun(object):

    def __init__(self, parameter_parser):
        self.parameter_parser = parameter_parser

        # Create a container for a dictionary of plot IDs (keys) to
        # complete neighbors and distances (values).
        self.neighbor_data = {}

        # Retrieve the attribute data
        self.stand_attr_data, self.attrs = self._get_attribute_data()

    def _get_attribute_data(self):
        """
        Retrieve the attribute data for which predictions will be made.  This
        should be called one time, after which it is stored in an instance-
        level variable along with the attribute names (self.attrs)

        Returns
        -------
        stand_attr_data : numpy recarray
            Recarray with all stand attributes
        attrs : list of strs
            List of all continuous variables in stand_attr_data
        """

        # Get the stand attribute table and read into a recarray
        p = self.parameter_parser
        stand_attr_file = p.stand_attribute_file
        stand_attr_data = utilities.csv2rec(stand_attr_file)

        # Get the stand attribute metadata and retrieve only the
        # continuous accuracy attributes
        stand_metadata_file = p.stand_metadata_file
        mp = xsmp.XMLStandMetadataParser(stand_metadata_file)
        attrs = [
            x.field_name
            for x in mp.attributes
            if x.field_type == 'CONTINUOUS' and x.accuracy_attr == 1
        ]

        return stand_attr_data, attrs

    def get_footprint_values(self, ds, windows, band=1):
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
        for (k, v) in windows.iteritems():
            out_dict[k] = self.get_footprint_value(v, band, gt)
        return out_dict

    def get_footprint_value(self, window, band, gt):
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

    def get_values_from_offset(self, footprints, offset):
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

    def calculate_neighbors_at_ids(self, id_x_year, id_field='FCID'):
        """
        Run ordination model over the list of IDs sent in and return neighbors
        and distances for each plot

        Parameters
        ----------
        id_x_year : dict
            Dictionary of plot IDs to associated imagery year to know what
            year to run the model

        id_field : str
            Name of the ID field - should be either 'FCID' or 'PLTID'.
            Defaults to 'FCID'

        Returns
        -------
        None (neighbor data stored as self attribute)
        """

        # Alias for self.parameter_parser
        p = self.parameter_parser

        # Ensure the parameter parser is not a PROTOTYPE
        if p.parameter_set not in ('FULL', 'MINIMUM'):
            err_msg = 'Parameter set must be "FULL" or "MINIMUM"'
            raise ValueError(err_msg)

        # Get footprint file
        fp_file = p.footprint_file

        # Check ID field
        if id_field not in ('FCID', 'PLTID'):
            err_msg = id_field + ' accuracy assessment is not currently '
            err_msg += 'supported'
            raise NotImplementedError(err_msg)

        # Get a list of the unique IDs
        ids = np.unique(id_x_year.keys())

        # Get a list of the years over which we need to run models
        years = np.unique(id_x_year.values())

        # Create a dictionary of all plots associated with each model year
        year_ids = {}
        for (k, v) in id_x_year.iteritems():
            try:
                year_ids[v].append(k)
            except KeyError:
                year_ids[v] = [k]

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
            ord_vars = p.get_ordination_variables(year)

            for (var, path) in ord_vars:
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

        # Retrieve all coordinates records as a recarray
        coords = utilities.csv2rec(p.coordinate_file)

        # Subset this list to just those plots in the model
        id_arr = getattr(coords, id_field)
        coord_list = coords[np.in1d(id_arr, ids)]

        # Retrieve the footprint configurations.  Footprint offsets store the
        # row and column tuples of each pixel within a given footprint.
        # Footprint windows store the upper left coordinate and window size for
        # extraction from GDAL datasets
        fp_parser = footprint.FootprintParser()
        fp_dict = fp_parser.parse(fp_file)
        fp_offsets = {}
        fp_windows = {}
        for (id_val, data_source, x, y) in coord_list:
            fp_offsets[id_val] = fp_dict[data_source].offsets
            fp_windows[id_val] = fp_dict[data_source].window((x, y))

        # Extract footprint information for every ordination variable that is
        # common to all years and store in a dict keyed by ID and raster
        # file name
        fp_value_dict = {}
        for (fn, count) in raster_counts.iteritems():
            if count == len(years):
                print fn
                ds, processed = raster_dict[fn]

                # Get the footprint window values for this dataset
                fp_values = self.get_footprint_values(ds, fp_windows)

                # Change the flag for this dataset to 'processed'
                raster_dict[fn][1] = True

                # Store these footprint values in a dictionary keyed by
                # id and variable file name
                for (id_val, fp) in fp_values.iteritems():
                    try:
                        fp_value_dict[id_val][fn] = fp
                    except KeyError:
                        fp_value_dict[id_val] = {}
                        fp_value_dict[id_val][fn] = fp

                # Close this dataset - no longer needed
                raster_dict[fn][0] = None

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

        # Main loop to iterate over all years
        for year in years:
            print year

            # Get the subset of footprint offsets and windows for this year
            offsets = dict((x, fp_offsets[x]) for x in year_ids[year])
            windows = dict((x, fp_windows[x]) for x in year_ids[year])

            # Extract footprints for any variables that are not common to all
            # years, but specialized for this year
            for (var, fn) in ord_year_var_dict[year].iteritems():
                ds, processed = raster_dict[fn]
                if not processed:
                    print fn

                    # Extract footprint values for this dataset
                    fp_values = self.get_footprint_values(ds, windows)

                    # Set the processed flag to True
                    raster_dict[fn][1] = True

                    # Store these values
                    for (id_val, fp) in fp_values.iteritems():
                        try:
                            fp_value_dict[id_val][fn] = fp
                        except KeyError:
                            fp_value_dict[id_val] = {}
                            fp_value_dict[id_val][fn] = fp

                    # Close the dataset - no longer needed
                    raster_dict[fn][0] = None

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
                    fp_values.append(fp_value_dict[id][fn])

                # Set up an output instance to capture each pixel's neighbors
                # and distances
                obj = NNFootprint(id)

                # Run the imputation for each pixel in the footprint
                for o in offsets[id]:

                    # Get the ordination variable values for this offset
                    # Store in (1xv) array
                    v = np.array(self.get_values_from_offset(fp_values, o))
                    v = v[np.newaxis, :]

                    # Run the imputation
                    nn_ids, nn_dists = imp_model.get_neighbors(v, id=id_val)

                    # Append this pixel to the NNFootprint object
                    obj.append(NNPixel(nn_ids, nn_dists))

                # Store the neighbor information
                self.neighbor_data[id] = copy.deepcopy(obj)

    def calculate_neighbors_cross_validation(self):
        """
        Wrapper around get_predicted_neighbors_at_ids optimized for cross-
        validation (ie. using plots that went into model development).

        Returns
        -------
        None
        """

        # Alias for self.parameter_parser
        p = self.parameter_parser

        # ID field
        id_field = p.plot_id_field

        # Get the environmental matrix file and read the plot IDs
        # and image years into a dictionary
        env_file = p.environmental_matrix_file
        env_data = utilities.csv2rec(env_file)

        # Associate each plot with a model year; this is either the year the
        # model is associated with (for models that use imagery), or the
        # model_year (for models that don't use imagery)
        if p.model_type in p.imagery_model_types:
            id_x_year = dict((x[id_field], x.IMAGE_YEAR) for x in env_data)
        else:
            id_x_year = dict((x[id_field], p.model_year) for x in env_data)

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
        self.calculate_neighbors_at_ids(id_x_year, id_field=id_field)

    def _calculate_prediction_at_id(
            self, id_val, k, nsa_id_dict, id_x_index, independent):
        """
        Calculate model prediction for the given ID.  This method is called
        from the wrapper calculate_predictions_at_k.  See this method for
        more complete documentation of these parameters

        Parameters
        ----------
        id_val : int
            The ID for which to calculate model predictions

        k : int
            The number of neighbors over which to average predicted values

        nsa_id_dict : dict
            Dictionary that defines self-assignment for IDs in the model

        id_x_index : dict
            Crosswalk of ID to row index of the stand attribute data

        independent : bool
            Flag to determine whether to capture dependent or independent
            neighbors for each ID.

        Returns
        -------
        plot_prediction : array of PixelPrediction objects
            The predictions for all pixels in a plot footprint
        """

        # Get the neighbor data for this plot
        fp = self.neighbor_data[id_val]

        # If this is an independent run, get the 'no self-assign'
        # value for this plot ID
        plot_nsa = None
        if independent:
            plot_nsa = nsa_id_dict[id_val]

        # Alias for stand attribute data
        sad = self.stand_attr_data

        # Create an empty list which will store all PixelPrediction
        # instances
        plot_prediction = []

        # Iterate over pixels in the footprint
        for (pixel_number, pixel) in enumerate(fp.pixels):

            # For independent neighbors, create a 'independence mask' for
            # this pixel
            if independent:
                index_mask = np.array([
                    False if nsa_id_dict[x] == plot_nsa
                    else True for x in pixel.neighbors
                ])

            # Create a PixelPrediction instance
            pp = PixelPrediction(id_val, pixel_number, k)

            # Push the k neighbors and distances to the
            # PixelPrediction instance
            if independent:
                pp.neighbors = pixel.neighbors[index_mask][0:k]
                pp.distances = pixel.distances[index_mask][0:k]
            else:
                pp.neighbors = pixel.neighbors[0:k]
                pp.distances = pixel.distances[0:k]

            # Fix distance array for 0.0 values
            distances = np.where(
                pp.distances == 0.0, MIN_DIST, pp.distances)

            # Calculate the normalized weight array as the
            # inverse distance
            weights = 1.0 / distances
            weights /= weights.sum()

            # Get the indexes of the rows for the k nearest neighbors
            indexes = [id_x_index[x] for x in pp.neighbors]

            # Extract the data rows
            data_rows = sad[indexes]

            # Iterate over all attributes calculating the weighted
            # average of each attribute for this pixel
            for attr in self.attrs:
                weighted_mean = getattr(data_rows, attr) * weights
                pp.set_predicted_attr(attr, weighted_mean.sum())

            # Add this pixel prediction to the list
            plot_prediction.append(pp)

        # Return the plot_predictions dict
        return plot_prediction

    def calculate_predictions_at_k(
            self, k=1, id_field='FCID', independent=True, **kwargs):
        """
        Calculates model predictions for the given value of k at the given ID
        and yields the plot prediction back to the caller.

        Parameters
        ----------
        k : int
            The number of neighbors over which to average predicted values.
            Defaults to 1
        id_field : str
            Name of the ID field - should be either 'FCID' or 'PLTID'.
            Defaults to 'FCID'
        independent : bool
            Flag to determine whether to capture dependent or independent
            neighbors for each ID.  If independent=True, an nsa_id_dict should
            be specified as well if the user wants control over how
            independence is determined.  Defaults to True.

        Keywords
        --------
        nsa_id_dict : dict
            Dictionary that defines self-assignment for IDs in the model.
            The dictionary's keys are the IDs (as identified in the id_field)
            and the values are IDs that define self assignment.  For example,
            the typical way GNN is run is to not allow self assignment in
            prediction if plots share the same geographic location (LOC_ID).
            This dictionary would therefore be the mapping of FCID to LOC_ID.
            Defaults to None, in which case only strict self-assignment is
            not allowed.  If this keyword is not specified, it will behave
            as if nsa_id_dict=None.

        Returns
        -------
        plot_prediction : array of PixelPrediction objects
            The predictions for all pixels in a plot footprint
        """
        # If nsa_id_dict is None or the keyword is missing, create the
        # dictionary as a mapping of ID to itself for independent runs
        if independent:
            if 'nsa_id_dict' not in kwargs:
                kwargs['nsa_id_dict'] = None

            if kwargs['nsa_id_dict'] is None:
                ids = self.neighbor_data.keys()
                nsa_id_dict = dict((x, x) for x in ids)
            else:
                nsa_id_dict = kwargs['nsa_id_dict']
        else:
            nsa_id_dict = None

        # Alias for stand attribute data
        sad = self.stand_attr_data

        # Create crosswalk of ID to row index
        id_x_index = {}
        for (i, rec) in enumerate(sad):
            id_x_index[getattr(rec, id_field)] = i

        # For every ID in pr.neighbor_data, calculate its predicted values
        # and neighbors at each pixel
        for id_val in sorted(self.neighbor_data.keys()):

            # Retrieve the independent predicted values
            plot_prediction = self._calculate_prediction_at_id(
                id_val, k, nsa_id_dict, id_x_index, independent)

            # Yield this back to the caller
            yield plot_prediction

    def write_nn_index_file(self, id_field, nn_index_file):
        """
        Calculate the average index of self-assignment for each plot.  This
        is a useful screen for determining outliers.  The nn_index_file is
        written out as a result.

        Parameters
        ----------
        id_field: str
            Name of the ID field

        nn_index_file: str
            Name of the nn_index_file

        Returns
        -------
        None
        """

        # Open the nn_index file and print the header line
        nn_index_fh = open(nn_index_file, 'w')
        header_fields = (id_field, 'AVERAGE_POSITION')
        nn_index_fh.write(','.join(header_fields) + '\n')

        # For each ID, find how far a plot had to go for self assignment
        for id_val in sorted(self.neighbor_data.keys()):
            nn_footprint = self.neighbor_data[id_val]
            self_assign_indexes = []
            for nn_pixel in nn_footprint.pixels:

                # Find the occurrence of this ID in the neighbor list
                # Because we restrict the neighbors to only the first 100,
                # we may not find the self-assignment within those neighbors.
                # Set it to the max value in this case
                try:
                    index = np.where(nn_pixel.neighbors == id)[0][0] + 1
                except IndexError:
                    index = nn_pixel.neighbors.size
                self_assign_indexes.append(index)

            # Get the average index position across pixels
            average_position = np.mean(self_assign_indexes)
            nn_index_fh.write('%d,%.4f\n' % (id_val, average_position))

        # Clean up
        nn_index_fh.close()

    def write_zonal_pixel_record(self, plot_prediction, zonal_pixel_fh):
        """
        Write out a record for the zonal pixel file which details the k
        neighbors and distances for each pixel in the plot footprint

        Parameters
        ----------
        plot_prediction: array of PixelPrediction objects
            Prediction information for each pixel in the passed plot

        zonal_pixel_fh: file
            File handle to the zonal pixel file

        Returns
        -------
        None
        """

        # Write out the k neighbors and distances for each pixel
        for pp in plot_prediction:
            for i in xrange(pp.k):
                zonal_pixel_fh.write('%d,%d,%d,%d,%.8f\n' % (
                    pp.id,
                    pp.pixel_number + 1,
                    i + 1,
                    pp.neighbors[i],
                    pp.distances[i],
                ))

    def write_predicted_record(
            self, plot_prediction, predicted_fh, attrs=None):
        """
        Write out a record to the predicted file which calculates predicted
        stand attributes for the passed plot

        Parameters
        ----------
        plot_prediction: array of PixelPrediction objects
            Prediction information for each pixel in the passed plot
        predicted_fh: file
            File handle to the predicted file

        Keywords
        --------
        attrs: list of strs
            An optional list of attributes over which to create predictions

        Returns
        -------
        None
        """

        # Fill in attrs if none were passed
        if attrs is None:
            attrs = self.attrs

        # Print out the predicted means calculated across pixels
        means = []
        for attr in attrs:
            values = np.array(
                [x.get_predicted_attr(attr) for x in plot_prediction]
            )
            means.append(values.mean())

        output_data = ['%d' % plot_prediction[0].id]
        output_data.extend(['%.4f' % x for x in means])
        predicted_fh.write(','.join(output_data) + '\n')
