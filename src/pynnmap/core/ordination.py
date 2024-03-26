import os
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from pynnmap.misc import numpy_ordination

BASEDIR = os.path.abspath(os.path.dirname(__file__))
VEGAN_SCRIPT = os.path.join(BASEDIR, "gnn_vegan.r")


@dataclass
class OrdinationParameters:
    spp_file: str
    env_file: str
    variables: List[str]
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


class Ordination(object):
    def __init__(self, parameters: OrdinationParameters):
        self.parameters = parameters

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


class NumpyOrdination(Ordination):
    def run(self):
        # Convert the species and environment matrices to numpy rec arrays
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

        # Create the ordination object
        ordination = self.ordination_cls(spp, env)
        prefix = self.ordination_prefix
        rank = ordination.rank

        def header_str(prefix, rank):
            return ",".join([f"{prefix}{i+1}" for i in range(rank)])

        # Two column (labels, values) - eigenvalues, means
        # Matrix (coefficient loadings, biplot, species centroids, tolerances)
        # Three column (labels)

        with open(self.parameters.ordination_file, "w") as numpy_fh:
            # Eigenvalues
            numpy_fh.write("### Eigenvalues ###\n")
            for i, e in enumerate(ordination.eigenvalues):
                numpy_fh.write(f"{prefix}{i+1},{e:.10f}\n")
            numpy_fh.write("\n")

            # Print out variable means
            numpy_fh.write("### Variable Means ###\n")
            for i, m in enumerate(ordination.env_means):
                numpy_fh.write(f"{self.parameters.variables[i]},{m:.10f}\n")
            numpy_fh.write("\n")

            # Print out environmental coefficients loadings
            numpy_fh.write("### Coefficient Loadings ###\n")
            numpy_fh.write(f"VARIABLE,{header_str(prefix, rank)}\n")
            for i, c in enumerate(ordination.coefficients()):
                coeff = ",".join([f"{x:.10f}" for x in c])
                numpy_fh.write(f"{self.parameters.variables[i]},{coeff}\n")
            numpy_fh.write("\n")

            # Print out biplot scores
            numpy_fh.write("### Biplot Scores ###\n")
            numpy_fh.write(f"VARIABLE,{header_str(prefix, rank)}\n")
            for i, b in enumerate(ordination.biplot_scores()):
                scores = ",".join([f"{x:.10f}" for x in b])
                numpy_fh.write(f"{self.parameters.variables[i]},{scores}\n")
            numpy_fh.write("\n")

            # Print out species centroids
            numpy_fh.write("### Species Centroids ###\n")
            numpy_fh.write(f"SPECIES,{header_str(prefix, rank)}\n")
            for i, c in enumerate(ordination.species_centroids()):
                scores = ",".join([f"{x:.10f}" for x in c])
                numpy_fh.write(f"{spp_df.columns[i]},{scores}\n")
            numpy_fh.write("\n")

            # Print out species tolerances
            numpy_fh.write("### Species Tolerances ###\n")
            numpy_fh.write(f"SPECIES,{header_str(prefix, rank)}\n")
            for i, t in enumerate(ordination.species_tolerances()):
                scores = ",".join([f"{x:.21f}" for x in t])
                numpy_fh.write(f"{spp_df.columns[i]},{scores}\n")
            numpy_fh.write("\n")

            # Print out miscellaneous species information
            numpy_fh.write("### Miscellaneous Species Information ###\n")
            numpy_fh.write("SPECIES,WEIGHT,N2\n")
            species_weights, species_n2 = ordination.species_information()
            for i in range(len(species_weights)):
                column = spp_df.columns[i]
                weight = f"{species_weights[i]:.10f}"
                n2 = f"{species_n2[i]:.10f}"
                numpy_fh.write(f"{column},{weight},{n2}\n")
            numpy_fh.write("\n")

            # Print out site LC scores
            numpy_fh.write("### Site LC Scores ###\n")
            numpy_fh.write(f"ID,{header_str(prefix, rank)}\n")
            for i, s in enumerate(ordination.site_lc_scores()):
                scores = ",".join([f"{x:.10f}" for x in s])
                numpy_fh.write(f"{spp_plot_ids[i]},{scores}\n")
            numpy_fh.write("\n")

            # Print out site WA scores
            numpy_fh.write("### Site WA Scores ###\n")
            numpy_fh.write(f"ID,{header_str(prefix, rank)}\n")
            for i, s in enumerate(ordination.site_wa_scores()):
                scores = ",".join([f"{x:.10f}" for x in s])
                numpy_fh.write(f"{spp_plot_ids[i]},{scores}\n")
            numpy_fh.write("\n")

            # Miscellaneous site information
            numpy_fh.write("### Miscellaneous Site Information ###\n")
            numpy_fh.write("ID,WEIGHT,N2\n")
            site_weights, site_n2 = ordination.site_information()
            for i in range(len(site_weights)):
                plot = spp_plot_ids[i]
                weight = f"{site_weights[i]:.10f}"
                n2 = f"{site_n2[i]:.10f}"
                numpy_fh.write(f"{plot},{weight},{n2}\n")


class NumpyCCAOrdination(NumpyOrdination):
    ordination_cls = numpy_ordination.NumpyCCA
    ordination_prefix = "CCA"


class NumpyRDAOrdination(NumpyOrdination):
    ordination_cls = numpy_ordination.NumpyRDA
    ordination_prefix = "RDA"
