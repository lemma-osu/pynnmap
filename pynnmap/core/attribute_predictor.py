from abc import ABC, abstractmethod
from functools import partial, reduce

import numpy as np
import pandas as pd

from pynnmap.core import get_weights
from pynnmap.core.pixel_prediction import (
    PixelPrediction,
    PlotAttributePrediction,
)
from pynnmap.parser.xml_stand_metadata_parser import Flags


# Minimum distance constant
MIN_DIST = 0.000000000001


def subset_neighbors(neighbor_data, k=1, fltr=None):
    plot_predictions = []
    for id_val, fp in sorted(neighbor_data.items()):
        pixel_predictions = []
        for pixel_number, pixel in enumerate(fp.pixels):
            if fltr:
                mask = fltr.mask(fp.id, pixel.neighbors)
                neighbors = pixel.neighbors[mask][:k]
                distances = pixel.distances[mask][:k]
            else:
                neighbors = pixel.neighbors[:k]
                distances = pixel.distances[:k]
            pixel_predictions.append(
                PixelPrediction(id_val, pixel_number, k, neighbors, distances)
            )
        plot_predictions.append(pixel_predictions)
    return plot_predictions


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

    def calculate_predictions(self, plot_predictions, k=1, weights=None):
        return [
            self.calculate_predictions_at_id(plot, k, weights)
            for plot in plot_predictions
        ]

    def calculate_predictions_at_id(self, plot, k, weights):
        """
        Parameters
        ----------
        plot : list of PixelPrediction objects
            The pixel predictions for this plot
        k : int
            The number of neighbors over which to average predicted values
        weights : np.array
            Weights for each neighbor, may be None

        Returns
        -------
        plot_prediction : PlotAttributePrediction object
            The predictions for all pixels in a plot footprint
        """
        # Create an empty list which will store all attribute predictions
        # for each pixel
        pixel_predictions = []

        # Determine if weights need to be calculated based on NN distances
        calc_weights = weights is None

        # Iterate over pixels in the plot
        for pixel in plot:
            neighbors = pixel.neighbors[:k]
            if calc_weights:
                # Fix distance array for 0.0 values
                distances = np.where(
                    pixel.distances[:k] == 0.0, MIN_DIST, pixel.distances[:k]
                )
                weights = 1.0 / distances
                weights /= weights.sum()
                weights = weights.reshape(1, len(neighbors)).T
            else:
                weights = weights[:k]

            # Extract the data rows and attributes and multiply by weights
            # Only do this for the first k neighbors
            indexes = [self.id_x_index[x] for x in neighbors]
            data_rows = self.attr_arr[indexes]
            arr = (data_rows * weights).sum(axis=0)

            # Add this attribute prediction to the list
            pixel_predictions.append(arr)

        # Return a PlotAttributePrediction instance which holds the ID of
        # the plot and the 2D collection of predictions (pixels x attrs)
        return PlotAttributePrediction(plot[0].id, pixel_predictions)

    def get_predicted_attributes_df(self, predictions, id_field):
        d = {}
        col_names = self.stand_attr_df.columns
        for prd in predictions:
            values = self.stat_func(prd.attr_arr)
            d[prd.id] = values
        prd_df = pd.DataFrame.from_dict(d, orient="index", columns=col_names)
        prd_df.sort_index(inplace=True)
        prd_df.index.rename(id_field, inplace=True)
        return prd_df


def majority(a):
    v, c = np.unique(a, return_counts=True)
    ind = np.argmax(c)
    return v[ind]


class ContinuousAttributePredictor(AttributePredictor):
    flags = Flags.CONTINUOUS | Flags.NOT_SPECIES | Flags.ACCURACY
    stat_func = partial(np.mean, axis=0)


class CategoricalAttributePredictor(AttributePredictor):
    flags = Flags.CATEGORICAL | Flags.ACCURACY
    stat_func = partial(np.apply_along_axis, majority, 0)


class SpeciesAttributePredictor(AttributePredictor):
    flags = Flags.SPECIES | Flags.ACCURACY
    stat_func = partial(np.mean, axis=0)


def calculate_predicted_attributes(
    plot_predictions, attr_data, fltr, parser, id_field
):
    dfs = []
    for (kls, k, weights) in (
        (ContinuousAttributePredictor, parser.k, get_weights(parser)),
        (CategoricalAttributePredictor, 1, np.array([1.0])),
        (SpeciesAttributePredictor, 1, np.array([1.0])),
    ):
        predictor = kls(attr_data, fltr)
        # TODO: No need to create local predictions variable here, do it
        #  within AttributePredictor class
        predictions = predictor.calculate_predictions(
            plot_predictions, k=k, weights=weights
        )
        prd_df = predictor.get_predicted_attributes_df(predictions, id_field)
        dfs.append(prd_df)

    return reduce(lambda df1, df2: pd.merge(df1, df2, on=id_field), dfs)
