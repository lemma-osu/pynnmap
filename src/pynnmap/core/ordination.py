from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sknnr.transformers import CCATransformer

from ..misc import numpy_ordination

BASEDIR = os.path.abspath(os.path.dirname(__file__))
VEGAN_SCRIPT = os.path.join(BASEDIR, "gnn_vegan.r")


@dataclass
class OrdinationParameters:
    spp_file: str
    env_file: str
    variables: list[str]
    id_field: str
    species_downweighting: bool
    species_transform: str
    ordination_file: str

    @classmethod
    def from_parser(cls, parser):
        return cls(
            spp_file=parser.species_matrix_file,
            env_file=parser.environmental_matrix_file,
            variables=parser.get_ordination_variable_names(),
            id_field=parser.plot_id_field,
            species_downweighting=parser.species_downweighting,
            species_transform=parser.species_transform,
            ordination_file=parser.get_ordination_file(),
        )


class Ordination:
    def __init__(self, parameters: OrdinationParameters):
        self.parameters = parameters

    def process_input(self) -> tuple[NDArray, NDArray, list[str], list[int]]:
        # Read in the species and environment matrices
        spp_df = pd.read_csv(self.parameters.spp_file)
        env_df = pd.read_csv(self.parameters.env_file)

        # Extract the plot IDs from both the species and environment matrices
        # and ensure that they are equal
        spp_plot_ids = spp_df[self.parameters.id_field]
        env_plot_ids = env_df[self.parameters.id_field]
        if not np.all(spp_plot_ids == env_plot_ids):
            err_msg = "Species and environment plot IDs do not match"
            raise ValueError(err_msg)

        # Drop the ID column from both dataframes
        spp_df.drop(labels=[self.parameters.id_field], axis=1, inplace=True)
        env_df.drop(labels=[self.parameters.id_field], axis=1, inplace=True)

        # For the environment matrix, only keep the variables specified
        env_df = env_df[self.parameters.variables]

        # Convert these matrices to pure floating point arrays
        spp = spp_df.values.astype(float)
        env = env_df.values.astype(float)

        # Apply transformation if desired
        if self.parameters.species_transform == "SQRT":
            spp = np.sqrt(spp)
        elif self.parameters.species_transform == "LOG":
            spp = np.log(spp)

        return spp, env, spp_df.columns, spp_plot_ids

    def run(self):
        raise NotImplementedError


class VeganOrdination(Ordination):
    def run(self):
        from rpy2 import robjects

        # Source the gnn_vegan R file
        robjects.r.source(VEGAN_SCRIPT)

        # Create an R vector to pass
        var_vector = robjects.StrVector(self.parameters.variables)

        # Create the vegan file
        robjects.r.write_vegan(
            self.method,
            self.parameters.spp_file,
            self.parameters.env_file,
            var_vector,
            self.parameters.id_field,
            self.parameters.species_transform,
            self.parameters.species_downweighting,
            self.parameters.ordination_file,
        )


class VeganCCAOrdination(VeganOrdination):
    method = "CCA"


class VeganRDAOrdination(VeganOrdination):
    method = "RDA"


class VeganDBRDAOrdination(VeganOrdination):
    method = "DBRDA"


@dataclass
class ConstrainedOrdinationResults:
    prefix: str
    rank: int
    species_names: list[str]
    env_names: list[str]
    site_ids: list[int]
    eigenvalues: NDArray
    env_means: NDArray
    coefficients: NDArray
    biplot_scores: NDArray
    species_centroids: NDArray
    species_tolerances: NDArray
    species_weights: NDArray
    species_n2: NDArray
    site_lc_scores: NDArray
    site_wa_scores: NDArray
    site_weights: NDArray
    site_n2: NDArray

    def write_output(self, ordination_file_name: str):
        with open(ordination_file_name, "w") as ordination_fh:
            self.write_eigenvalues(ordination_fh)
            self.write_variable_means(ordination_fh)
            self.write_coefficients(ordination_fh)
            self.write_biplot_scores(ordination_fh)
            self.write_species_centroids(ordination_fh)
            self.write_species_tolerances(ordination_fh)
            self.write_species_information(ordination_fh)
            self.write_site_lc_scores(ordination_fh)
            self.write_site_wa_scores(ordination_fh)
            self.write_site_information(ordination_fh)

    def header_str(self):
        return ",".join([f"{self.prefix}{i + 1}" for i in range(self.rank)])

    def write_cca_section(
        self,
        fh,
        *,
        heading: str,
        row_column_name: str,
        row_ids: list[str | int],
        data: NDArray,
        n_decimals: int,
    ):
        """Write a section of a CCA ordination file."""
        fh.write(f"### {heading} ###\n")
        fh.write(f"{row_column_name},{self.header_str()}\n")
        for i, row in enumerate(data):
            data_str = ",".join([f"{x:.{n_decimals}f}" for x in row])
            fh.write(f"{row_ids[i]},{data_str}\n")
        fh.write("\n")

    def write_eigenvalues(self, fh):
        fh.write("### Eigenvalues ###\n")
        for i, e in enumerate(self.eigenvalues):
            fh.write(f"{self.prefix}{i + 1},{e:.8f}\n")
        fh.write("\n")

    def write_variable_means(self, fh):
        fh.write("### Variable Means ###\n")
        for i, m in enumerate(self.env_means):
            fh.write(f"{self.env_names[i]},{m:.5f}\n")
        fh.write("\n")

    def write_coefficients(self, fh):
        self.write_cca_section(
            fh,
            heading="Coefficient Loadings",
            row_column_name="VARIABLE",
            row_ids=self.env_names,
            data=self.coefficients,
            n_decimals=11,
        )

    def write_biplot_scores(self, fh):
        self.write_cca_section(
            fh,
            heading="Biplot Scores",
            row_column_name="VARIABLE",
            row_ids=self.env_names,
            data=self.biplot_scores,
            n_decimals=9,
        )

    def write_species_centroids(self, fh):
        self.write_cca_section(
            fh,
            heading="Species Centroids",
            row_column_name="SPECIES",
            row_ids=self.species_names,
            data=self.species_centroids,
            n_decimals=9,
        )

    def write_species_tolerances(self, fh):
        self.write_cca_section(
            fh,
            heading="Species Tolerances",
            row_column_name="SPECIES",
            row_ids=self.species_names,
            data=self.species_tolerances,
            n_decimals=6,
        )

    def write_species_information(self, fh):
        fh.write("### Miscellaneous Species Information ###\n")
        fh.write("SPECIES,WEIGHT,N2\n")
        for i in range(len(self.species_weights)):
            column = self.species_names[i]
            weight = f"{self.species_weights[i]:.5f}"
            n2 = f"{self.species_n2[i]:.5f}"
            fh.write(f"{column},{weight},{n2}\n")
        fh.write("\n")

    def write_site_lc_scores(self, fh):
        self.write_cca_section(
            fh,
            heading="Site LC Scores",
            row_column_name="ID",
            row_ids=self.site_ids,
            data=self.site_lc_scores,
            n_decimals=11,
        )

    def write_site_wa_scores(self, fh):
        self.write_cca_section(
            fh,
            heading="Site WA Scores",
            row_column_name="ID",
            row_ids=self.site_ids,
            data=self.site_wa_scores,
            n_decimals=10,
        )

    def write_site_information(self, fh):
        fh.write("### Miscellaneous Site Information ###\n")
        fh.write("ID,WEIGHT,N2\n")
        for i in range(len(self.site_weights)):
            plot = self.site_ids[i]
            weight = f"{self.site_weights[i]:.6f}"
            n2 = f"{self.site_n2[i]:.6f}"
            fh.write(f"{plot},{weight},{n2}\n")


class NumpyOrdination(Ordination):
    def run(self):
        # Convert the species and environment matrices to numpy arrays
        spp, env, species_names, site_ids = self.process_input()

        # Create the ordination object
        ordination = self.ordination_cls(spp, env)

        # Write out the results
        species_weights, species_n2 = ordination.species_information()
        site_weights, site_n2 = ordination.site_information()
        ordination_results = ConstrainedOrdinationResults(
            prefix=self.ordination_prefix,
            rank=ordination.rank,
            eigenvalues=ordination.eigenvalues,
            env_means=ordination.env_means,
            species_names=species_names,
            env_names=self.parameters.variables,
            site_ids=site_ids,
            coefficients=ordination.coefficients(),
            biplot_scores=ordination.biplot_scores(),
            species_centroids=ordination.species_centroids(),
            species_tolerances=ordination.species_tolerances(),
            species_weights=species_weights,
            species_n2=species_n2,
            site_lc_scores=ordination.site_lc_scores(),
            site_wa_scores=ordination.site_wa_scores(),
            site_weights=site_weights,
            site_n2=site_n2,
        )
        ordination_results.write_output(self.parameters.ordination_file)


class NumpyCCAOrdination(NumpyOrdination):
    ordination_cls = numpy_ordination.NumpyCCA
    ordination_prefix = "CCA"


class NumpyRDAOrdination(NumpyOrdination):
    ordination_cls = numpy_ordination.NumpyRDA
    ordination_prefix = "RDA"


class SknnrCCAOrdination(Ordination):
    def run(self):
        # Convert the species and environment matrices to numpy arrays
        spp, env, species_names, site_ids = self.process_input()

        # Create the transformation object
        cca = CCATransformer().fit(env, spp)

        # TODO: This belongs in sknnr
        species_weights = np.sum(spp, axis=0)
        a = np.square(np.divide(spp, species_weights))
        species_n2 = 1.0 / a.sum(axis=0)

        # TODO: This belongs in sknnr
        site_weights = np.sum(spp, axis=1)
        a = np.square(np.divide(spp, np.expand_dims(site_weights, axis=1)))
        site_n2 = 1.0 / a.sum(axis=1)

        ordination_results = ConstrainedOrdinationResults(
            prefix="CCA",
            rank=cca.ordination_.rank,
            eigenvalues=cca.ordination_.eigenvalues,
            env_means=cca.ordination_.env_center,
            species_names=species_names,
            env_names=self.parameters.variables,
            site_ids=site_ids,
            coefficients=cca.ordination_.coefficients,
            biplot_scores=cca.ordination_.biplot_scores,
            species_centroids=cca.ordination_.species_scores,
            species_tolerances=cca.ordination_.species_tolerances,
            species_weights=species_weights,
            species_n2=species_n2,
            site_lc_scores=cca.ordination_.site_lc_scores,
            site_wa_scores=cca.ordination_.site_wa_scores,
            site_weights=site_weights,
            site_n2=site_n2,
        )
        ordination_results.write_output(self.parameters.ordination_file)
