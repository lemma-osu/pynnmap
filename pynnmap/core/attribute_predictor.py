import numpy as np
import pandas as pd

from pynnmap.core.pixel_prediction import PixelPrediction
from pynnmap.parser.xml_stand_metadata_parser import Flags


# Minimum distance constant
MIN_DIST = 0.000000000001


class AttributePredictor(object):
    def __init__(self, stand_attributes, independence_filter=None):
        """
        TODO: Flesh this out
        Parameters
        ----------
        stand_attributes : StandAttributes
            Stand attributes for which to make predictions
        independence_filter : IndependenceFilter, optional
            Instance that defines non-independence for IDs in the model.
        """
        flags = Flags.CONTINUOUS | Flags.ACCURACY
        self.stand_attr_df = stand_attributes.get_attr_df(flags=flags)

        # For speed, create a lookup of id_field to row index and convert
        # attribute data to a numpy array
        indexes = np.arange(len(self.stand_attr_df))
        self.id_x_index = pd.Series(indexes, index=self.stand_attr_df.index)
        self.id_x_index = self.id_x_index.to_dict()
        self.attr_arr = self.stand_attr_df.values

        # Independence filter
        self.independence_filter = independence_filter

    def calculate_predictions(self, neighbor_data, k=1):
        # Iterate over all neighbors and calculate predictions
        d = {}
        col_names = self.stand_attr_df.columns
        for id_val, fp in sorted(neighbor_data.items()):
            prd = self.calculate_predictions_at_id(fp, k)
            id_val = prd[0].id
            values = np.mean([x.get_predicted_attrs() for x in prd], axis=0)
            d[id_val] = values
        return pd.DataFrame.from_dict(d, orient='index', columns=col_names)

    def calculate_predictions_at_id(self, fp, k):
        """
        Parameters
        ----------
        fp : NNFootprint
            The footprint for which to calculate model predictions
        k : int
            The number of neighbors over which to average predicted values

        Returns
        -------
        plot_prediction : array of PixelPrediction objects
            The predictions for all pixels in a plot footprint
        """
        # Create an empty list which will store all PixelPrediction
        # instances
        plot_prediction = []

        # Iterate over pixels in the footprint
        for pixel_number, pixel in enumerate(fp.pixels):
            # Create a PixelPrediction instance
            pp = PixelPrediction(fp.id, pixel_number, k)

            # Filter the neighbors using the independence mask
            if self.independence_filter:
                mask = self.independence_filter.mask(fp.id, pixel.neighbors)
                pp.neighbors = pixel.neighbors[mask][0:k]
                pp.distances = pixel.distances[mask][0:k]
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
            weights = weights.reshape(1, len(pp.neighbors)).T

            # Extract the data rows and attributes and multiply by weights
            indexes = [self.id_x_index[x] for x in pp.neighbors]
            data_rows = self.attr_arr[indexes]
            arr = (data_rows * weights).sum(axis=0)

            # Calculate the weighted average of all attributes for this pixel
            pp.set_predicted_attrs(arr)

            # Add this pixel prediction to the list
            plot_prediction.append(pp)

        # Return the plot_predictions dict
        return plot_prediction
