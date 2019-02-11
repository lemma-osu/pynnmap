import numpy as np

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
        self.sad = stand_attributes.get_attr_df(flags=flags)

        # Independence filter
        self.independence_filter = independence_filter

    def calculate_predictions_at_id(self, fp, k):
        """
        Calculate model prediction for the given ID.  This method is called
        from the wrapper calculate_predictions_at_k.  See this method for
        more complete documentation of these parameters

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

            # Extract the data rows and attributes
            data_rows = self.sad.loc[pp.neighbors, :]

            # Calculate the weighted average of all attributes for this pixel
            # This is captured as a pandas series after the sum reducer
            s = data_rows.mul(weights, axis=0).sum(axis=0)
            pp.set_predicted_attrs(s)

            # Add this pixel prediction to the list
            plot_prediction.append(pp)

        # Return the plot_predictions dict
        return plot_prediction
