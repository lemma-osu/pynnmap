import numpy as np
import pandas as pd

from pynnmap.core.stand_attributes import StandAttributes
from pynnmap.core.independence_filter import IndependenceFilter
from pynnmap.core.attribute_predictor import AttributePredictor
from pynnmap.parser import xml_stand_metadata_parser as xsmp
from pynnmap.parser.xml_stand_metadata_parser import Flags


class PredictionOutput(object):

    def __init__(self, parameters):
        self.parameter_parser = parameters
        self.id_field = self.parameter_parser.plot_id_field

    def open_prediction_files(self, zonal_pixel_file, predicted_file, attrs):
        """
        Open the prediction files and write the header lines.  Return
        file handles to the caller

        Parameters
        ----------
        zonal_pixel_file : str
            Name of the zonal_pixel_file to open
        predicted_file : str
            Name of the predicted file to open
        attrs : list
            Attribute names

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
        predicted_fh = open(predicted_file, 'w')
        predicted_fh.write(self.id_field + ',' + ','.join(attrs) + '\n')

        return zonal_pixel_fh, predicted_fh

    @staticmethod
    def write_zonal_pixel_record(plot_prediction, zonal_pixel_fh):
        """
        Write out a record for the zonal pixel file which details the k
        neighbors and distances for each pixel in the plot footprint

        Parameters
        ----------
        plot_prediction: array of PixelPrediction objects
            Prediction information for each pixel in the passed plot
        zonal_pixel_fh: file
            File handle to the zonal pixel file
        """
        # Write out the k neighbors and distances for each pixel
        for pp in plot_prediction:
            for i in range(pp.k):
                zonal_pixel_fh.write('%d,%d,%d,%d,%.8f\n' % (
                    pp.id,
                    pp.pixel_number + 1,
                    i + 1,
                    pp.neighbors[i],
                    pp.distances[i],
                ))

    @staticmethod
    def write_predicted_record(plot_prediction, predicted_fh):
        """
        Write out a record to the predicted file which calculates predicted
        stand attributes for the passed plot

        Parameters
        ----------
        plot_prediction : array of PixelPrediction objects
            Prediction information for each pixel in the passed plot
        predicted_fh : file
            File handle to the predicted file
        """
        # Print out the predicted means calculated across pixels
        values = np.array([x.get_predicted_attrs() for x in plot_prediction])
        output_data = ['%d' % plot_prediction[0].id]
        output_data.extend(['%.4f' % x for x in values.mean(axis=0)])
        predicted_fh.write(','.join(output_data) + '\n')

    @staticmethod
    def write_nn_index_file(neighbor_data, id_field, nn_index_file):
        """
        Calculate the average index of self-assignment for each plot.  This
        is a useful screen for determining outliers.  The nn_index_file is
        written out as a result.

        Parameters
        ----------
        neighbor_data : dict
            Dictionary of IDs to neighbors and distances
        id_field: str
            Name of the ID field
        nn_index_file: str
            Name of the nn_index_file
        """
        # Open the nn_index file and print the header line
        nn_index_fh = open(nn_index_file, 'w')
        header_fields = (id_field, 'AVERAGE_POSITION')
        nn_index_fh.write(','.join(header_fields) + '\n')

        # For each ID, find how far a plot had to go for self assignment
        for id_val, fp in sorted(neighbor_data.items()):
            self_assign_indexes = []
            for nn_pixel in fp.pixels:
                # Find the occurrence of this ID in the neighbor list
                # Because we restrict the neighbors to only the first 100,
                # we may not find the self-assignment within those neighbors.
                # Set it to the max value in this case
                try:
                    index = np.where(nn_pixel.neighbors == id_val)[0][0] + 1
                except IndexError:
                    index = nn_pixel.neighbors.size
                self_assign_indexes.append(index)

            # Get the average index position across pixels
            average_position = float(np.mean(self_assign_indexes))
            nn_index_fh.write('%d,%.4f\n' % (id_val, average_position))

        # Clean up
        nn_index_fh.close()


class IndependentOutput(PredictionOutput):
    def __init__(self, parameters):
        super(IndependentOutput, self).__init__(parameters)

    def create_predictions(self, neighbor_data, no_self_assign_field='LOC_ID'):
        """
        Creates model predictions and zonal pixel files from independent
        predictions, ie. plots are not able to use themselves (or other
        'dependent' plots) as neighbors

        Parameters
        ----------
        neighbor_data : dict
            Dictionary of IDs to neighbors and distances
        no_self_assign_field : str
            ID field at which no self assignment is allowed.
            Defaults to LOC_ID
        """
        # Aliases
        p = self.parameter_parser

        # Create an independence filter based on the relationship of the
        # id_field and the no_self_assign_field.
        fn = p.plot_independence_crosswalk_file
        fields = [self.id_field, no_self_assign_field]
        df = pd.read_csv(fn, usecols=fields, index_col=self.id_field)
        fltr = IndependenceFilter.from_common_lookup(
            df.index, df[no_self_assign_field]
        )

        # Get the stand attributes and filter to continuous accuracy fields
        attr_fn = p.stand_attribute_file
        mp = xsmp.XMLStandMetadataParser(p.stand_metadata_file)
        attr_data = StandAttributes(attr_fn, mp, id_field=self.id_field)
        flags = Flags.CONTINUOUS | Flags.ACCURACY
        attrs = mp.filter(flags=flags)

        # Create a plot attribute predictor instance
        plot_attr_predictor = AttributePredictor(attr_data, fltr)

        # Open the prediction files
        zonal_pixel_file = p.independent_zonal_pixel_file
        predicted_file = p.independent_predicted_file
        zonal_pixel_fh, predicted_fh = \
            self.open_prediction_files(zonal_pixel_file, predicted_file, attrs)

        # Calculate the predictions for each plot
        for id_val, fp in sorted(neighbor_data.items()):
            prd = plot_attr_predictor.calculate_predictions_at_id(fp, p.k)
            self.write_zonal_pixel_record(prd, zonal_pixel_fh)
            self.write_predicted_record(prd, predicted_fh)

        # Close files
        zonal_pixel_fh.close()
        predicted_fh.close()


class DependentOutput(PredictionOutput):
    def __init__(self, parameters):
        super(DependentOutput, self).__init__(parameters)

    def create_predictions(self, neighbor_data):
        """
        Creates model predictions, zonal pixel, and nn_index files from
        dependent predictions, ie. plots are able to use themselves as
        neighbors

        Parameters
        ----------
        neighbor_data : dict
            Dictionary of IDs to neighbors and distances
        """
        # Aliases
        p = self.parameter_parser

        # Get the stand attributes
        attr_fn = p.stand_attribute_file
        mp = xsmp.XMLStandMetadataParser(p.stand_metadata_file)
        attr_data = StandAttributes(attr_fn, mp, id_field=self.id_field)

        # Open the prediction files
        zonal_pixel_file = p.dependent_zonal_pixel_file
        predicted_file = p.dependent_predicted_file
        flags = Flags.CONTINUOUS | Flags.ACCURACY
        attrs = mp.filter(flags=flags)
        zonal_pixel_fh, predicted_fh = \
            self.open_prediction_files(zonal_pixel_file, predicted_file, attrs)

        # # Write the dependent nn_index file
        nn_index_file = p.dependent_nn_index_file
        self.write_nn_index_file(neighbor_data, self.id_field, nn_index_file)

        # Create a plot attribute predictor instance
        plot_attr_predictor = AttributePredictor(attr_data)
        for id_val, fp in sorted(neighbor_data.items()):
            prd = plot_attr_predictor.calculate_predictions_at_id(fp, p.k)
            self.write_zonal_pixel_record(prd, zonal_pixel_fh)
            self.write_predicted_record(prd, predicted_fh)

        # Close files
        zonal_pixel_fh.close()
        predicted_fh.close()
