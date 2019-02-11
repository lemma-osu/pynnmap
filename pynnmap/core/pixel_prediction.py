class PixelPrediction(object):
    """
    Class to hold a given pixel's prediction including neighbor IDs, distances
    and predicted values for each continuous attribute.
    """
    def __init__(self, id_val, pixel_number, k):
        self.id = id_val
        self.pixel_number = pixel_number
        self.k = k
        self._predicted_df = None
        self._neighbors = None
        self._distances = None

    def __repr__(self):
        return '{kls}(\n id={id},\n neighbors={n},\n distances={d}\n)'.format(
            kls=self.__class__.__name__,
            id=self.id,
            n=self.neighbors,
            d=self.distances
        )

    @property
    def neighbors(self):
        return self._neighbors

    @neighbors.setter
    def neighbors(self, neighbors):
        self._neighbors = neighbors

    @property
    def distances(self):
        return self._distances

    @distances.setter
    def distances(self, distances):
        self._distances = distances

    def get_predicted_attrs(self):
        return self._predicted_df

    def set_predicted_attrs(self, df):
        self._predicted_df = df
