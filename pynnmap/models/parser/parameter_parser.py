class MissingParameterException(Exception):
    pass


class ParameterParser(object):
    """
    Abstract base class for parsing model parameters from an
    input file.
    """

    def __init__(self):
        self.imagery_model_types = ['sppsz']
        self.no_imagery_model_types = ['sppba', 'trecov', 'wdycov', 'trepa']

    def __repr__(self):
        raise NotImplementedError

    def validate(self):
        raise NotImplementedError
