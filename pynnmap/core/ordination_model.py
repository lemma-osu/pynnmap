import json

import numpy as np


class OrdinationModel(object):

    def __init__(self):
        pass

    def __repr__(self):
        return_str = ''
        return_str += '\nNumber of axes: ' + repr(self.n_axes)
        return_str += '\nNumber of variables: ' + repr(self.n_variables)
        return_str += '\nNumber of species: ' + repr(self.n_species)
        return_str += '\nNumber of plots: ' + repr(self.n_plots) + '\n'
        return_str += '\nAxis weights:\n'
        return_str += repr(self.axis_weights) + '\n'
        return_str += '\nAxis intercepts:\n'
        return_str += repr(self.axis_intercepts) + '\n'
        return_str += '\nVariable names:\n'
        return_str += repr(self.var_names) + '\n'
        return_str += '\nVariable coefficients:\n'
        return_str += repr(self.var_coeff) + '\n'
        return_str += '\nSpecies names:\n'
        return_str += repr(self.species_names) + '\n'
        return_str += '\nSpecies scores:\n'
        return_str += repr(self.species_scores) + '\n'
        return_str += '\nPlot IDs:\n'
        return_str += repr(self.plot_ids) + '\n'
        return_str += '\nPlot scores:\n'
        return_str += repr(self.plot_scores) + '\n'
        return return_str

    def to_json(self):
        d = {}
        for i in self.__dict__:
            if isinstance(self.__dict__[i], int):
                d[i] = self.__dict__[i]
            elif isinstance(self.__dict__[i], dict):
                d[i] = self.__dict__[i]
            elif isinstance(self.__dict__[i], np.ndarray):
                d[i] = self.__dict__[i].tolist()
        return json.dumps(d)

    def plot_score(self, id):
        # Find the corresponding row in plot_ids
        row_num = self.plot_id_dict[id]
        return self.plot_scores[row_num]
