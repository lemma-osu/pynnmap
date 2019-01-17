import os

import numpy as np
from matplotlib import mlab
from osgeo import gdal, gdalconst

from pynnmap.diagnostics import diagnostic
from pynnmap.misc import histogram
from pynnmap.misc import utilities
from pynnmap.parser import parameter_parser as pp
from pynnmap.parser import xml_stand_metadata_parser as xsmp


class RegionalAccuracyDiagnostic(diagnostic.Diagnostic):

    def __init__(self, **kwargs):
        if 'parameters' in kwargs:
            p = kwargs['parameters']
            if isinstance(p, pp.ParameterParser):
                raster_name = ''.join([
                    'mr', str(p.model_region), '_nnmsk1'])
                self.predicted_raster = os.path.join(
                    p.model_directory, raster_name)
                self.area_estimate_file = p.area_estimate_file
                self.stand_attribute_file = p.stand_attribute_file
                self.stand_metadata_file = p.stand_metadata_file
                self.id_field = p.plot_id_field
                self.statistics_file = p.regional_accuracy_file
            else:
                err_msg = 'Passed object is not a ParameterParser object'
                raise ValueError(err_msg)
        else:
            err_msg = 'Only ParameterParser objects may be passed.'
            raise NotImplementedError(err_msg)

        # Ensure all input files are present
        files = [
            self.predicted_raster, self.area_estimate_file,
            self.stand_attribute_file, self.stand_metadata_file]
        try:
            self.check_missing_files(files)
        except diagnostic.MissingConstraintError as e:
            e.message += '\nSkipping RegionalAccuracyDiagnostic\n'
            raise e

    def get_observed_estimates(self):

        # Read the area estimate file into a recarray
        obs_data = utilities.csv2rec(self.area_estimate_file)

        # Get the nonforest hectares (coded as -10001)
        nf_row = obs_data[getattr(obs_data, self.id_field) == -10001][0]
        nf_hectares = nf_row.HECTARES

        # Get the nonsampled hectares (coded as -10002)
        ns_row = obs_data[getattr(obs_data, self.id_field) == -10002][0]
        ns_hectares = ns_row.HECTARES

        # Remove these rows from the recarray
        obs_data = obs_data[getattr(obs_data, self.id_field) > 0]

        # Return this information
        return obs_data, nf_hectares, ns_hectares

    def get_predicted_estimates(self):

        # Read in the predicted raster
        ds = gdal.Open(self.predicted_raster, gdalconst.GA_ReadOnly)
        rb = ds.GetRasterBand(1)

        # Get the cell area for converting from pixel counts to hectares
        gt = ds.GetGeoTransform()
        cell_area = gt[1] * gt[1]
        hectares_per_pixel = cell_area / 10000.0

        # Calculate statistics directly on the raster
        z_min, z_max = int(rb.GetMinimum()), int(rb.GetMaximum())
        z_range = z_max - z_min + 1
        hist = rb.GetHistogram(z_min - 0.5, z_max + 0.5, z_range, False, False)
        bins = list(range(z_min, z_max + 1))
        rat = [(x, y) for x, y in zip(bins, hist) if y != 0]

        # Get the IDs and counts (converted to hectares)
        id_recs = []
        nf_hectares = 0
        for (id_val, count) in rat:
            hectares = count * hectares_per_pixel 
            if id_val <= 0:
                nf_hectares += hectares
            else:
                id_recs.append((id_val, hectares))

        # Release the dataset
        ds = None

        # Convert this to a recarray
        names = (self.id_field, 'HECTARES')
        ids = np.rec.fromrecords(id_recs, names=names)

        # Read in the attribute file
        sad = utilities.csv2rec(self.stand_attribute_file)

        # Ensure that all IDs in the id_count_dict are in the attribute data
        ids_1 = getattr(ids, self.id_field)
        ids_2 = getattr(sad, self.id_field)
        if not np.all(np.in1d(ids_1, ids_2)):
            err_msg = 'Not all values in the raster are present in the '
            err_msg += 'attribute data'
            raise ValueError(err_msg)

        # Join the two recarrays together
        predicted_data = mlab.rec_join(self.id_field, ids, sad)
        return (predicted_data, nf_hectares)

    def insert_class(self, histogram, name, count):
        histogram.bin_counts = \
            np.insert(histogram.bin_counts, [0], count)
        histogram.bin_names.insert(0, name)
        return histogram

    def run_diagnostic(self):

        # Read in the observed data from the area estimate file
        (obs_area, obs_nf_hectares, obs_ns_hectares) = \
           self.get_observed_estimates()

        # Get the observed and predicted data arrays
        (prd_area, prd_nf_hectares) = self.get_predicted_estimates()
        prd_ns_hectares = 0.0

        # Get the weights of the two datasets
        obs_weights = obs_area.HECTARES
        prd_weights = prd_area.HECTARES

        # Open the output file and print out the header line
        stats_fh = open(self.statistics_file, 'w')
        header_fields = ['VARIABLE', 'DATASET', 'BIN_NAME', 'AREA']
        stats_fh.write(','.join(header_fields) + '\n')

        # Get a metadata parser
        mp = xsmp.XMLStandMetadataParser(self.stand_metadata_file)

        # Iterate over all fields and print out the area histogram statistics
        for v in obs_area.dtype.names:

            # Skip over the HECTARES field
            if v == 'HECTARES':
                continue

            # Get the metadata for this field
            try:
                fm = mp.get_attribute(v)
            except:
                err_msg = v + ' is missing metadata.'
                print(err_msg)
                continue

            # Skip over ID fields
            if fm.field_type == 'ID':
                continue

            # Get the actual data
            try:
                obs_vals = getattr(obs_area, v)
                prd_vals = getattr(prd_area, v)
            except AttributeError:
                continue

            obs_vw = histogram.VariableVW(obs_vals, obs_weights)
            prd_vw = histogram.VariableVW(prd_vals, prd_weights)

            # Figure out how to bin the data based on field type
            if fm.field_type == 'CONTINUOUS':
                bins = histogram.bin_continuous([obs_vw, prd_vw], bins=7)
            else:
                if fm.codes:
                    class_names = {}
                    for code in fm.codes:
                        class_names[code.code_value] = code.label
                else:
                    class_names = None
                bins = histogram.bin_categorical(
                    [obs_vw, prd_vw], class_names=class_names)

            bins[0].name = 'OBSERVED'
            bins[1].name = 'PREDICTED'

            # Handle special cases of nonsampled and nonforest area
            self.insert_class(bins[0], 'Unsampled', obs_ns_hectares)
            self.insert_class(bins[0], 'Nonforest', obs_nf_hectares)
            self.insert_class(bins[1], 'Unsampled', prd_ns_hectares)
            self.insert_class(bins[1], 'Nonforest', prd_nf_hectares)

            for bin in bins:
                for i in range(0, len(bin.bin_counts)):
                    out_data = [
                        '%s' % v,
                        '%s' % bin.name,
                        '"%s"' % bin.bin_names[i],
                        '%.3f' % bin.bin_counts[i],
                    ]
                    stats_fh.write(','.join(out_data) + '\n')
