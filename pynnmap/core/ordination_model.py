from typing import Dict
import json

import numpy as np
from pydantic.dataclasses import dataclass


class Config:
    arbitrary_types_allowed = True


@dataclass(config=Config)
class OrdinationModel:
    axis_weights: np.ndarray
    var_names: np.ndarray
    var_coeff: np.ndarray
    var_means: np.ndarray
    species_names: np.ndarray
    species_scores: np.ndarray
    plot_ids: np.ndarray
    plot_scores: np.ndarray
    biplot_scores: np.ndarray

    def trim_axes(self, n_axes: int):
        return OrdinationModel(
            axis_weights=self.axis_weights[:n_axes],
            var_names=self.var_names,
            var_coeff=self.var_coeff[:, :n_axes],
            var_means=self.var_means,
            species_names=self.species_names,
            species_scores=self.species_scores[:, :n_axes],
            plot_ids=self.plot_ids,
            plot_scores=self.plot_scores[:, :n_axes],
            biplot_scores=self.biplot_scores[:, :n_axes],
        )

    @property
    def n_variables(self) -> int:
        return len(self.var_names)

    @property
    def n_axes(self) -> int:
        return self.axis_weights.size

    @property
    def n_species(self) -> int:
        return self.species_names.size

    @property
    def n_plots(self) -> int:
        return len(self.plot_ids)

    @property
    def axis_intercepts(self) -> int:
        return np.dot(self.var_means, self.var_coeff)

    @property
    def plot_id_dict(self) -> Dict[int, int]:
        return {x: idx for idx, x in enumerate(self.plot_ids)}

    @property
    def id_plot_dict(self) -> Dict[int, int]:
        return dict(enumerate(self.plot_ids))

    @property
    def var_name_dict(self) -> Dict[str, int]:
        return {x: idx for idx, x in enumerate(self.var_names)}
