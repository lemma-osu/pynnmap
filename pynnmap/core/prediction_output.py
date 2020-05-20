import numpy as np
import pandas as pd

from pynnmap.core.stand_attributes import StandAttributes
from pynnmap.core.independence_filter import IndependenceFilter
from pynnmap.core.attribute_predictor import AttributePredictor
from pynnmap.misc.utilities import df_to_csv
from pynnmap.parser import xml_stand_metadata_parser as xsmp
from pynnmap.parser.xml_stand_metadata_parser import Flags


class PredictionOutput(object):
    def __init__(self, parameters):
        self.parameter_parser = parameters
        self.id_field = self.parameter_parser.plot_id_field

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
        nn_index_fh = open(nn_index_file, "w")
        header_fields = (id_field, "AVERAGE_POSITION")
        nn_index_fh.write(",".join(header_fields) + "\n")

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
            nn_index_fh.write("%d,%.4f\n" % (id_val, average_position))

        # Clean up
        nn_index_fh.close()

    def get_weights(self):
        w = self.parameter_parser.weights
        if w is not None:
            if len(w) != self.parameter_parser.k:
                raise ValueError("Length of weights does not equal k")
            w = np.array(w).reshape(1, len(w)).T
        return w


class IndependentOutput(PredictionOutput):
    def __init__(self, parameters):
        super(IndependentOutput, self).__init__(parameters)

    def create_predictions(self, neighbor_data, no_self_assign_field="LOC_ID"):
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

        # Get the stand attributes and filter to continuous accuracy fields
        attr_fn = p.stand_attribute_file
        mp = xsmp.XMLStandMetadataParser(p.stand_metadata_file)
        attr_data = StandAttributes(attr_fn, mp, id_field=self.id_field)

        # Create an independence filter based on the relationship of the
        # id_field and the no_self_assign_field.
        fn = p.plot_independence_crosswalk_file
        fields = [self.id_field, no_self_assign_field]
        df = pd.read_csv(fn, usecols=fields, index_col=self.id_field)
        fltr = IndependenceFilter.from_common_lookup(
            df.index, df[no_self_assign_field]
        )

        # Create a plot attribute predictor instance
        plot_attr_predictor = AttributePredictor(attr_data, fltr)

        # Calculate the predictions for each plot
        w = self.get_weights()
        predictions = plot_attr_predictor.calculate_predictions(
            neighbor_data, k=p.k, weights=w
        )

        # Write out zonal pixel file
        zp_df = plot_attr_predictor.get_zonal_pixel_df(predictions)
        df_to_csv(zp_df, p.independent_zonal_pixel_file, n_dec=8)

        # Write out predicted attribute file
        prd_df = plot_attr_predictor.get_predicted_attributes_df(
            predictions, self.id_field
        )
        df_to_csv(prd_df, p.independent_predicted_file, index=True)


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

        # # Write the dependent nn_index file
        nn_index_file = p.dependent_nn_index_file
        self.write_nn_index_file(neighbor_data, self.id_field, nn_index_file)

        # Create a plot attribute predictor instance
        plot_attr_predictor = AttributePredictor(attr_data)

        # Calculate the predictions for each plot
        w = self.get_weights()
        predictions = plot_attr_predictor.calculate_predictions(
            neighbor_data, k=p.k, weights=w
        )

        # Write out zonal pixel file
        zp_df = plot_attr_predictor.get_zonal_pixel_df(predictions)
        df_to_csv(zp_df, p.dependent_zonal_pixel_file, n_dec=8)

        # Write out predicted attribute file
        prd_df = plot_attr_predictor.get_predicted_attributes_df(
            predictions, self.id_field
        )
        df_to_csv(prd_df, p.dependent_predicted_file, index=True)
