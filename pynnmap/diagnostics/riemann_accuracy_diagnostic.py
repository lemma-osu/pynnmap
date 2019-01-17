import os

import numpy as np
from matplotlib import mlab

from pynnmap.core import prediction_run
from pynnmap.diagnostics import diagnostic
from pynnmap.misc import utilities
from pynnmap.parser import parameter_parser as pp
from pynnmap.parser import xml_stand_metadata_parser as xsmp


class ECDF:
    def __init__(self, observations):
        self.observations = observations

    def __call__(self, x):
        try:
            return (self.observations <= x).mean()
        except AttributeError:
            self.observations = np.array(self.observations)
            return (self.observations <= x).mean()


class RiemannComparison(object):

    def __init__(self, prefix, obs_file, prd_file, id_field, k):
        self.prefix = prefix
        self.obs_file = obs_file
        self.prd_file = prd_file
        self.id_field = id_field
        self.k = k


class RiemannVariable(object):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def gmfr_statistics(self):

        # Short circuit the condition where there is no observed or predicted
        # variance typically caused by no observed or predicted presences
        if self.x.var() == 0.0 or self.y.var() == 0.0:
            gmfr_stats = {}
            gmfr_stats['gmfr_a'] = 0.0
            gmfr_stats['gmfr_b'] = 0.0
            gmfr_stats['ac'] = 0.0
            gmfr_stats['ac_sys'] = 0.0
            gmfr_stats['ac_uns'] = 0.0
            return gmfr_stats

        x_mean = self.x.mean()
        y_mean = self.y.mean()

        b = np.sqrt(self.y.var() / self.x.var())
        a = y_mean - (b * x_mean)

        # Sum of squared differences
        ssd = ((self.x - self.y) * (self.x - self.y)).sum()

        # Sum of product differences for GMFR for systematic and unsystematic
        # differences
        c = -a / b
        d = 1.0 / b
        y_hat = a + b * self.x
        x_hat = c + d * self.y
        spd_u = (np.abs(self.x - x_hat) * np.abs(self.y - y_hat)).sum()
        spd_s = ssd - spd_u

        # Sum of potential differences
        term1 = np.abs(x_mean - y_mean) + np.abs(self.x - x_mean)
        term2 = np.abs(x_mean - y_mean) + np.abs(self.y - y_mean)
        spod = (term1 * term2).sum()

        # Agreement coefficients for total, systematic and
        # unsystematic differences
        ac = 1.0 - (ssd / spod)
        ac_sys = 1.0 - (spd_s / spod)
        ac_uns = 1.0 - (spd_u / spod)

        gmfr_stats = {}
        gmfr_stats['gmfr_a'] = a
        gmfr_stats['gmfr_b'] = b
        gmfr_stats['ac'] = ac
        gmfr_stats['ac_sys'] = ac_sys
        gmfr_stats['ac_uns'] = ac_uns

        return gmfr_stats

    def ks_statistics(self):

        x_sorted = np.sort(self.x)
        y_sorted = np.sort(self.y)

        global_max = np.max(np.hstack((x_sorted, y_sorted)))
        global_min = np.min(np.hstack((x_sorted, y_sorted)))

        bins = np.linspace(global_min, global_max, 1000)

        x_ecdf = ECDF(x_sorted)
        y_ecdf = ECDF(y_sorted)

        x_freq = np.array([x_ecdf(i) for i in bins])
        y_freq = np.array([y_ecdf(i) for i in bins])
        diff_freq = np.abs(x_freq - y_freq)
        ks_max = np.max(diff_freq)
        ks_mean = np.mean(diff_freq)

        ks_stats = {}
        ks_stats['ks_max'] = ks_max
        ks_stats['ks_mean'] = ks_mean

        return ks_stats


class RiemannAccuracyDiagnostic(diagnostic.Diagnostic):

    def __init__(self, **kwargs):
        if 'parameters' in kwargs:
            p = kwargs['parameters']
            if isinstance(p, pp.ParameterParser):

                # We need the parameter parser for many attributes when
                # running the diagnostic, so just keep a reference to it
                self.parameter_parser = p

                # We want to check on the existence of the hex attribute
                # file before running, so make this an instance property
                self.hex_attribute_file = p.hex_attribute_file
            else:
                err_msg = 'Passed object is not a ParameterParser object'
                raise ValueError(err_msg)
        else:
            err_msg = 'Only ParameterParser objects may be passed.'
            raise NotImplementedError(err_msg)

        # Ensure all input files are present
        files = [self.hex_attribute_file]
        try:
            self.check_missing_files(files)
        except diagnostic.MissingConstraintError as e:
            e.message += '\nSkipping RiemannAccuracyDiagnostic\n'
            raise e

    def write_hex_stats(
            self, data, id_field, stat_fields, min_plots_per_hex, out_file):

        # Summarize the observed output
        stats = mlab.rec_groupby(data, (id_field,), stat_fields)

        # Filter so that the minimum number of plots per hex is maintained
        stats = stats[stats.PLOT_COUNT >= min_plots_per_hex]

        # Write out the file
        utilities.rec2csv(stats, out_file)

    def run_diagnostic(self):

        # Shortcut to the parameter parser
        p = self.parameter_parser

        # ID field
        id_field = p.plot_id_field

        # Root directory for Riemann files
        root_dir = p.riemann_output_folder

        # Read in hex input file
        obs_data = utilities.csv2rec(self.hex_attribute_file)

        # Get the hexagon levels and ensure that the fields exist in the
        # hex_attribute file
        hex_resolutions = p.riemann_hex_resolutions
        hex_fields = [x[0] for x in hex_resolutions]
        for field in hex_fields:
            if field not in obs_data.dtype.names:
                err_msg = 'Field ' + field + ' does not exist in the '
                err_msg += 'hex_attribute file'
                raise ValueError(err_msg)

        # Create the directory structure based on the hex levels
        hex_levels = ['hex_' + str(x[1]) for x in hex_resolutions]
        all_levels = ['plot_pixel'] + hex_levels
        for level in all_levels:
            sub_dir = os.path.join(root_dir, level)
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)

        # Get the values of k
        k_values = p.riemann_k_values

        # Create a dictionary of plot ID to image year (or model_year for
        # non-imagery models) for these plots
        if p.model_type in p.imagery_model_types:
            id_x_year = dict((x[id_field], x.IMAGE_YEAR) for x in obs_data)
        else:
            id_x_year = dict((x[id_field], p.model_year) for x in obs_data)

        # Create a PredictionRun instance
        pr = prediction_run.PredictionRun(p)

        # Get the neighbors and distances for these IDs
        pr.calculate_neighbors_at_ids(id_x_year, id_field=id_field)

        # Create the lookup of id_field to LOC_ID for the hex plots
        nsa_id_dict = dict((x[id_field], x.LOC_ID) for x in obs_data)

        # Create a dictionary between id_field and no_self_assign_field
        # for the model plots
        env_file = p.environmental_matrix_file
        env_data = utilities.csv2rec(env_file)
        model_nsa_id_dict = dict(
            (getattr(x, id_field), x.LOC_ID) for x in env_data)

        # Stitch the two dictionaries together
        for id_val in sorted(model_nsa_id_dict.keys()):
            if id_val not in nsa_id_dict:
                nsa_id_dict[id_val] = model_nsa_id_dict[id_val]

        # Get the stand attribute metadata and retrieve only the
        # continuous accuracy attributes
        stand_metadata_file = p.stand_metadata_file
        mp = xsmp.XMLStandMetadataParser(stand_metadata_file)
        attrs = [
            x.field_name for x in mp.attributes
            if x.field_type == 'CONTINUOUS' and x.accuracy_attr == 1]

        # Subset the attributes for fields that are in the
        # hex_attribute file
        attrs = [x for x in attrs if x in obs_data.dtype.names]
        plot_pixel_obs = mlab.rec_keep_fields(obs_data, [id_field] + attrs)

        # Write out the plot_pixel observed file
        file_name = 'plot_pixel_observed.csv'
        output_file = os.path.join(root_dir, 'plot_pixel', file_name)
        utilities.rec2csv(plot_pixel_obs, output_file)

        # Iterate over values of k
        for k in k_values:

            # Construct the output file name
            file_name = '_'.join(('plot_pixel', 'predicted', 'k' + str(k)))
            file_name += '.csv'
            output_file = os.path.join(root_dir, 'plot_pixel', file_name)
            out_fh = open(output_file, 'w')

            # For the plot/pixel scale, retrieve the independent predicted
            # data for this value of k.  Even though attributes are being
            # returned from this function, we want to use the attribute list
            # that we've already found above.
            prediction_generator = pr.calculate_predictions_at_k(
                k=k, id_field=id_field, independent=True,
                nsa_id_dict=nsa_id_dict)

            # Write out the field names
            out_fh.write(id_field + ',' + ','.join(attrs) + '\n')

            # Write out the predictions for this k
            for plot_prediction in prediction_generator:

                # Write this record to the predicted attribute file
                pr.write_predicted_record(plot_prediction, out_fh, attrs=attrs)

            # Close this file
            out_fh.close()

        # Create the fields for which to extract statistics at the hexagon
        # levels
        mean_fields = [(id_field, len, 'PLOT_COUNT')]
        mean_fields.extend([(x, np.mean, x) for x in attrs])
        mean_fields = tuple(mean_fields)

        sd_fields = [(id_field, len, 'PLOT_COUNT')]
        sd_fields.extend([(x, np.std, x) for x in attrs])
        sd_fields = tuple(sd_fields)

        stat_sets = {
            'mean': mean_fields,
            'std': sd_fields,
        }

        # For each hexagon level, associate the plots with their hexagon ID
        # and find observed and predicted statistics for each hexagon
        for hex_resolution in hex_resolutions:

            (hex_id_field, hex_distance) = hex_resolution[0:2]
            min_plots_per_hex = hex_resolution[3]
            prefix = 'hex_' + str(hex_distance)

            # Create a crosswalk between the id_field and the hex_id_field
            id_x_hex = mlab.rec_keep_fields(obs_data, [id_field, hex_id_field])

            # Iterate over all sets of statistics and write a unique file
            # for each set
            for (stat_name, stat_fields) in stat_sets.items():

                # Get the output file name
                obs_out_file = \
                    '_'.join((prefix, 'observed', stat_name)) + '.csv'
                obs_out_file = os.path.join(root_dir, prefix, obs_out_file)

                # Write out the observed file
                self.write_hex_stats(
                    obs_data, hex_id_field, stat_fields, min_plots_per_hex,
                    obs_out_file)

            # Iterate over values of k for the predicted values
            for k in k_values:

                # Open the plot_pixel predicted file for this value of k
                # and join the hex_id_field to the recarray
                prd_file = 'plot_pixel_predicted_k' + str(k) + '.csv'
                prd_file = os.path.join(root_dir, 'plot_pixel', prd_file)
                prd_data = utilities.csv2rec(prd_file)
                prd_data = mlab.rec_join(id_field, prd_data, id_x_hex)

                # Iterate over all sets of statistics and write a unique file
                # for each set
                for (stat_name, stat_fields) in stat_sets.items():

                    # Get the output file name
                    prd_out_file = '_'.join((
                        prefix, 'predicted', 'k' + str(k), stat_name)) + '.csv'
                    prd_out_file = os.path.join(root_dir, prefix, prd_out_file)

                    # Write out the predicted file
                    self.write_hex_stats(
                        prd_data, hex_id_field, stat_fields, min_plots_per_hex,
                        prd_out_file)

        # Calculate the ECDF and AC statistics
        # For ECDF and AC, it is a paired comparison between the observed
        # and predicted data.  We do this at each value of k and for each
        # hex resolution level.

        # Open the stats file
        stats_file = p.hex_statistics_file
        stats_fh = open(stats_file, 'w')
        header_fields = ['LEVEL', 'K', 'VARIABLE', 'STATISTIC', 'VALUE']
        stats_fh.write(','.join(header_fields) + '\n')

        # Create a list of RiemannComparison instances which store the
        # information needed to do comparisons between observed and predicted
        # files for any level or value of k
        compare_list = []
        for hex_resolution in hex_resolutions:
            (hex_id_field, hex_distance) = hex_resolution[0:2]
            prefix = 'hex_' + str(hex_distance)
            obs_file = '_'.join((prefix, 'observed', 'mean')) + '.csv'
            obs_file = os.path.join(root_dir, prefix, obs_file)
            for k in k_values:
                prd_file = '_'.join((
                    prefix, 'predicted', 'k' + str(k), 'mean')) + '.csv'
                prd_file = os.path.join(root_dir, prefix, prd_file)
                r = RiemannComparison(
                    prefix, obs_file, prd_file, hex_id_field, k)
                compare_list.append(r)

        # Add the plot_pixel comparisons to this list
        prefix = 'plot_pixel'
        obs_file = 'plot_pixel_observed.csv'
        obs_file = os.path.join(root_dir, prefix, obs_file)
        for k in k_values:
            prd_file = 'plot_pixel_predicted_k' + str(k) + '.csv'
            prd_file = os.path.join(root_dir, prefix, prd_file)
            r = RiemannComparison(prefix, obs_file, prd_file, id_field, k)
            compare_list.append(r)

        # Do all the comparisons
        for c in compare_list:

            # Open the observed file
            obs_data = utilities.csv2rec(c.obs_file)

            # Open the predicted file
            prd_data = utilities.csv2rec(c.prd_file)

            # Ensure that the IDs between the observed and predicted
            # data line up
            ids1 = getattr(obs_data, c.id_field)
            ids2 = getattr(prd_data, c.id_field)
            if np.all(ids1 != ids2):
                err_msg = 'IDs do not match between observed and '
                err_msg += 'predicted data'
                raise ValueError(err_msg)

            for attr in attrs:
                arr1 = getattr(obs_data, attr)
                arr2 = getattr(prd_data, attr)
                rv = RiemannVariable(arr1, arr2)

                gmfr_stats = rv.gmfr_statistics()
                for stat in ('gmfr_a', 'gmfr_b', 'ac', 'ac_sys', 'ac_uns'):
                    stat_line = '%s,%d,%s,%s,%.4f\n' % (
                        c.prefix.upper(), c.k, attr, stat.upper(),
                        gmfr_stats[stat])
                    stats_fh.write(stat_line)

                ks_stats = rv.ks_statistics()
                for stat in ('ks_max', 'ks_mean'):
                    stat_line = '%s,%d,%s,%s,%.4f\n' % (
                        c.prefix.upper(), c.k, attr, stat.upper(),
                        ks_stats[stat])
                    stats_fh.write(stat_line)
