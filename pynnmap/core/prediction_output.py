from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

from pynnmap.core import get_independence_filter
from pynnmap.core.attribute_predictor import (
    subset_neighbors,
    calculate_predicted_attributes,
)
from pynnmap.core.stand_attributes import StandAttributes
from pynnmap.misc.utilities import df_to_csv
from pynnmap.parser import xml_stand_metadata_parser as xsmp


class PredictionOutput(ABC):
    @property
    @abstractmethod
    def fltr(self):
        pass

    def __init__(self, parameters, neighbor_data):
        self.parameter_parser = parameters
        self.id_field = self.parameter_parser.plot_id_field
        self.plot_predictions = subset_neighbors(
            neighbor_data, self.parameter_parser.k, fltr=self.fltr
        )

    def write_zonal_records(self, zonal_pixel_fn):
        zps = [self.get_zonal_records(plot) for plot in self.plot_predictions]
        zp_df = pd.concat(zps)
        df_to_csv(zp_df, zonal_pixel_fn, n_dec=8)

    def get_zonal_records(self, plot):
        n_pixels = len(plot)
        k = len(plot[0].neighbors)
        n = [plot[i].neighbors[j] for i in range(n_pixels) for j in range(k)]
        d = [plot[i].distances[j] for i in range(n_pixels) for j in range(k)]
        return pd.DataFrame(
            {
                self.id_field: np.repeat(plot[0].id, k * n_pixels),
                "PIXEL_NUMBER": np.repeat(np.arange(n_pixels) + 1, k),
                "NEIGHBOR": np.tile(np.arange(k) + 1, n_pixels),
                "NEIGHBOR_ID": n,
                "DISTANCE": d,
            }
        )

    def write_attribute_predictions(self, predicted_fn):
        parser = self.parameter_parser

        # Get the stand attributes
        attr_fn = parser.stand_attribute_file
        mp = xsmp.XMLStandMetadataParser(parser.stand_metadata_file)
        attr_data = StandAttributes(attr_fn, mp, id_field=self.id_field)

        # Create the predictions for continuous, categorical and species
        # attributes
        prd_df = calculate_predicted_attributes(
            self.plot_predictions, attr_data, self.fltr, parser, self.id_field
        )
        df_to_csv(prd_df, predicted_fn, index=True)


class IndependentOutput(PredictionOutput):
    """
    Creates model predictions and zonal pixel files from independent
    predictions, ie. plots are not able to use themselves (or other
    'dependent' plots) as neighbors
    """

    @property
    def fltr(self):
        return get_independence_filter(self.parameter_parser)


class DependentOutput(PredictionOutput):
    """
    Creates model predictions, zonal pixel, and nn_index files from
    dependent predictions, ie. plots are able to use themselves as
    neighbors
    """

    fltr = None

    def write_nn_index_file(self, neighbor_data, nn_index_fn):
        """
        Calculate the average index of self-assignment for each plot.  This
        is a useful screen for determining outliers.  The nn_index_file is
        written out as a result.

        Parameters
        ----------
        neighbor_data : dict
            Dictionary of IDs to neighbors and distances
        nn_index_fn : str
            Path to the output NN index file
        """
        id_field = self.parameter_parser.plot_id_field

        with open(nn_index_fn, "w") as nn_index_fh:
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
