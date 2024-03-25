class PixelPrediction:
    """
    (Data)class to hold a pixel's prediction including neighbor IDs
    and distances up to k values
    """

    def __init__(self, id_val, pixel_number, k, neighbors, distances):
        self.id = id_val
        self.pixel_number = pixel_number
        self.k = k
        self.neighbors = neighbors
        self.distances = distances


class PlotAttributePrediction:
    """
    (Data)class to hold a plot's ID and predicted attribute values
    at the pixel scale.  The attr_arr is a 2D array of pixels (rows) by
    attributes (columns).
    """

    def __init__(self, id_val, attr_arr):
        self.id = id_val
        self.attr_arr = attr_arr
