from abc import ABC, abstractmethod
from functools import partial

import numpy as np
import pandas as pd

from pynnmap.core.pixel_prediction import PixelPrediction
from pynnmap.parser.xml_stand_metadata_parser import Flags


# Minimum distance constant
MIN_DIST = 0.000000000001


class AttributePredictor(ABC):
    @property
    @abstractmethod
    def flags(self):
        pass

    @property
    @abstractmethod
    def stat_func(self):
        pass

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
        self.stand_attr_df = stand_attributes.get_attr_df(flags=self.flags)

        # For speed, create a lookup of id_field to row index and convert
        # attribute data to a numpy array
        indexes = np.arange(len(self.stand_attr_df))
        self.id_x_index = pd.Series(indexes, index=self.stand_attr_df.index)
        self.id_x_index = self.id_x_index.to_dict()
        self.attr_arr = self.stand_attr_df.values

        # Independence filter
        self.independence_filter = independence_filter

    def calculate_predictions(self, neighbor_data, k=1, weights=None):
        # Iterate over all neighbors and calculate predictions
        predictions = []
        for id_val, fp in sorted(neighbor_data.items()):
            predictions.append(self.calculate_predictions_at_id(fp, k, weights))
        return predictions

    # TODO: This doesn't belong here - it is attribute independent
    def get_zonal_pixel_df(self, predictions):
        zps = []
        for prd in predictions:
            zps.append(self.prediction_to_zonal_records(prd))
        return pd.concat(zps)

    def get_predicted_attributes_df(self, predictions, id_field):
        d = {}
        col_names = self.stand_attr_df.columns
        for prd in predictions:
            values = self.stat_func([x.get_predicted_attrs() for x in prd])
            d[prd[0].id] = values
        prd_df = pd.DataFrame.from_dict(d, orient="index", columns=col_names)
        prd_df.sort_index(inplace=True)
        prd_df.index.rename(id_field, inplace=True)
        return prd_df

    @staticmethod
    def prediction_to_zonal_records(pp):
        n_pixels = len(pp)
        k = len(pp[0].neighbors)
        n = [pp[i].neighbors[j] for i in range(n_pixels) for j in range(k)]
        d = [pp[i].distances[j] for i in range(n_pixels) for j in range(k)]
        return pd.DataFrame(
            {
                "FCID": np.repeat(pp[0].id, k * n_pixels),
                "PIXEL_NUMBER": np.repeat(np.arange(n_pixels) + 1, k),
                "NEIGHBOR": np.tile(np.arange(k) + 1, n_pixels),
                "NEIGHBOR_ID": n,
                "DISTANCE": d,
            }
        )

    def calculate_predictions_at_id(self, fp, k, weights):
        """
        Parameters
        ----------
        fp : NNFootprint
            The footprint for which to calculate model predictions
        k : int
            The number of neighbors over which to average predicted values
        weights : np.array
            Weights for each neighbor, may be None

        Returns
        -------
        plot_prediction : array of PixelPrediction objects
            The predictions for all pixels in a plot footprint
        """
        # Create an empty list which will store all PixelPrediction
        # instances
        plot_prediction = []

        # Determine if weights need to be calculated based on NN distances
        calc_weights = True if weights is None else False

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
            distances = np.where(pp.distances == 0.0, MIN_DIST, pp.distances)

            # Calculate the normalized weight array as the
            # inverse distance
            if calc_weights:
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


def majority(a):
    v, c = np.unique(a, return_counts=True)
    ind = np.argmax(c)
    return v[ind]


class ContinuousAttributePredictor(AttributePredictor):
    flags = Flags.CONTINUOUS | Flags.ACCURACY
    stat_func = partial(np.mean, axis=0)


class CategoricalAttributePredictor(AttributePredictor):
    flags = Flags.CATEGORICAL | Flags.ACCURACY
    stat_func = partial(np.apply_along_axis, majority, 0)
