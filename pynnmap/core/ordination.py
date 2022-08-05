import numpy as np
import pandas as pd

from pynnmap.misc import numpy_ordination
from pynnmap.parser import parameter_parser as pp

VEGAN_SCRIPT = 'L:/resources/code/models/pre_process/gnn_vegan.r'


class Ordination(object):
    def __init__(self):
        pass

    def run(self):
        raise NotImplementedError


class VeganOrdination(Ordination):
    def __init__(self, **kwargs):
        super(VeganOrdination, self).__init__()

        # If we've passed a ParameterParser object, use this to initialize
        # this instance.  Otherwise, all keywords need to be present in
        # order to initialize
        try:
            p = kwargs['parameters']
            if not isinstance(p, pp.ParameterParser):
                raise ValueError('Passed object is not a ParameterParser object')
            self.spp_file = p.species_matrix_file
            self.env_file = p.environmental_matrix_file
            self.variables = p.get_ordination_variable_names()
            self.id_field = p.plot_id_field
            self.species_downweighting = p.species_downweighting
            self.species_transform = p.species_transform
            self.ord_file = p.get_ordination_file()
        except KeyError:
            try:
                self.spp_file = kwargs['spp_file']
                self.env_file = kwargs['env_file']
                self.variables = kwargs['variables']
                self.id_field = kwargs['id_field'].upper()
                self.species_downweighting = kwargs['species_downweighting']
                self.species_transform = kwargs['species_transform']
                self.ord_file = kwargs['vegan_file']
            except KeyError:
                raise ValueError('Not all required parameters were passed')

    def run(self):
        from rpy2 import robjects

        # Source the gnn_vegan R file
        robjects.r.source(VEGAN_SCRIPT)

        # Create an R vector to pass
        var_vector = robjects.StrVector(self.variables)

        # Create the vegan file
        robjects.r.write_vegan(
            self.method, self.spp_file, self.env_file, var_vector,
            self.id_field, self.species_transform, self.species_downweighting,
            self.ord_file)


class VeganCCAOrdination(VeganOrdination):
    def __init__(self, **kwargs):
        super(VeganCCAOrdination, self).__init__(**kwargs)

        # Set the method for this run
        self.method = 'CCA'


class VeganRDAOrdination(VeganOrdination):
    def __init__(self, **kwargs):
        super(VeganRDAOrdination, self).__init__(**kwargs)

        # Set the method for this run
        self.method = 'RDA'


class VeganDBRDAOrdination(VeganOrdination):
    def __init__(self, **kwargs):
        super(VeganDBRDAOrdination, self).__init__(**kwargs)

        # Set the method for this run
        self.method = 'DBRDA'


class NumpyOrdination(Ordination):
    def __init__(self, **kwargs):
        super(NumpyOrdination, self).__init__()

        # If we've passed a ParameterParser object, use this to initialize
        # this instance.  Otherwise, all keywords need to be present in
        # order to initialize
        try:
            p = kwargs['parameters']
            if not isinstance(p, pp.ParameterParser):
                raise ValueError('Passed object is not a ParameterParser object')
            self.spp_file = p.species_matrix_file
            self.env_file = p.environmental_matrix_file
            self.variables = p.get_ordination_variable_names()
            self.id_field = p.plot_id_field
            self.species_downweighting = p.species_downweighting
            self.species_transform = p.species_transform
            self.ord_file = p.get_ordination_file()
        except KeyError:
            try:
                self.spp_file = kwargs['spp_file']
                self.env_file = kwargs['env_file']
                self.variables = kwargs['variables']
                self.id_field = kwargs['id_field'].upper()
                self.species_downweighting = kwargs['species_downweighting']
                self.species_transform = kwargs['species_transform']
                self.ord_file = kwargs['numpy_file']
            except KeyError:
                raise ValueError('Not all required parameters were passed')

    def run(self):
        raise NotImplementedError


class NumpyCCAOrdination(NumpyOrdination):
    def __init__(self, **kwargs):
        super(NumpyCCAOrdination, self).__init__(**kwargs)

    def run(self):

        # Convert the species and environment matrices to numpy rec arrays
        spp_df = pd.read_csv(self.spp_file)
        env_df = pd.read_csv(self.env_file)

        # Extract the plot IDs from both the species and environment matrices
        # and ensure that they are equal
        spp_plot_ids = spp_df[self.id_field]
        env_plot_ids = env_df[self.id_field]
        if not np.all(spp_plot_ids == env_plot_ids):
            err_msg = 'Species and environment plot IDs do not match'
            raise ValueError(err_msg)

        # Drop the ID column from both dataframes
        spp_df.drop(labels=[self.id_field], axis=1, inplace=True)
        env_df.drop(labels=[self.id_field], axis=1, inplace=True)

        # For the environment matrix, only keep the variables specified
        env_df = env_df[self.variables]

        # Convert these matrices to pure floating point arrays
        spp = spp_df.values.astype(float)
        env = env_df.values.astype(float)

        # Apply transformation if desired
        if self.species_transform == 'SQRT':
            spp = np.sqrt(spp)
        elif self.species_transform == 'LOG':
            spp = np.log(spp)

        # Create the CCA object
        cca = numpy_ordination.NumpyCCA(spp, env)

        with open(self.ord_file, 'w') as numpy_fh:
            # Eigenvalues
            numpy_fh.write('### Eigenvalues ###\n')
            for (i, e) in enumerate(cca.eigenvalues):
                numpy_fh.write(f'CCA{str(i + 1)},' + '%.10f' % e + '\n')
            numpy_fh.write('\n')

            # Print out variable means
            numpy_fh.write('### Variable Means ###\n')
            for (i, m) in enumerate(cca.env_means):
                numpy_fh.write('%s,%.10f\n' % (self.variables[i], m))
            numpy_fh.write('\n')

            # Print out environmental coefficients loadings
            numpy_fh.write('### Coefficient Loadings ###\n')
            header_str = ','.join(['CCA%d' % (i + 1) for i in range(cca.rank)])
            numpy_fh.write(f'VARIABLE,{header_str}' + '\n')
            for (i, c) in enumerate(cca.coefficients()):
                coeff = ','.join(['%.10f' % x for x in c])
                numpy_fh.write('%s,%s\n' % (self.variables[i], coeff))
            numpy_fh.write('\n')

            # Print out biplot scores
            numpy_fh.write('### Biplot Scores ###\n')
            header_str = ','.join(['CCA%d' % (i + 1) for i in range(cca.rank)])
            numpy_fh.write(f'VARIABLE,{header_str}' + '\n')
            for (i, b) in enumerate(cca.biplot_scores()):
                scores = ','.join(['%.10f' % x for x in b])
                numpy_fh.write('%s,%s\n' % (self.variables[i], scores))
            numpy_fh.write('\n')

            # Print out species centroids
            numpy_fh.write('### Species Centroids ###\n')
            header_str = ','.join(['CCA%d' % (i + 1) for i in range(cca.rank)])
            numpy_fh.write(f'SPECIES,{header_str}' + '\n')
            for (i, c) in enumerate(cca.species_centroids()):
                scores = ','.join(['%.10f' % x for x in c])
                numpy_fh.write('%s,%s\n' % (spp_df.columns[i], scores))
            numpy_fh.write('\n')

            # Print out species tolerances
            numpy_fh.write('### Species Tolerances ###\n')
            header_str = \
                    ','.join(['CCA%d' % (i + 1) for i in range(cca.rank)])
            numpy_fh.write(f'SPECIES,{header_str}' + '\n')
            for (i, t) in enumerate(cca.species_tolerances()):
                scores = ','.join(['%.21f' % x for x in t])
                numpy_fh.write('%s,%s\n' % (spp_df.columns[i], scores))
            numpy_fh.write('\n')

            # Print out miscellaneous species information
            numpy_fh.write('### Miscellaneous Species Information ###\n')
            numpy_fh.write('SPECIES,WEIGHT,N2\n')
            species_weights, species_n2 = cca.species_information()
            for i in range(len(species_weights)):
                numpy_fh.write('%s,%.10f,%.10f\n' % (
                    spp_df.columns[i], species_weights[i], species_n2[i]))
            numpy_fh.write('\n')

            # Print out site LC scores
            numpy_fh.write('### Site LC Scores ###\n')
            header_str = ','.join(['CCA%d' % (i + 1) for i in range(cca.rank)])
            numpy_fh.write(f'ID,{header_str}' + '\n')
            for (i, s) in enumerate(cca.site_lc_scores()):
                scores = ','.join(['%.10f' % x for x in s])
                numpy_fh.write('%d,%s\n' % (spp_plot_ids[i], scores))
            numpy_fh.write('\n')

            # Print out site WA scores
            numpy_fh.write('### Site WA Scores ###\n')
            header_str = ','.join(['CCA%d' % (i + 1) for i in range(cca.rank)])
            numpy_fh.write(f'ID,{header_str}' + '\n')
            for (i, s) in enumerate(cca.site_wa_scores()):
                scores = ','.join(['%.10f' % x for x in s])
                numpy_fh.write('%d,%s\n' % (spp_plot_ids[i], scores))
            numpy_fh.write('\n')

            # Miscellaneous site information
            numpy_fh.write('### Miscellaneous Site Information ###\n')
            numpy_fh.write('ID,WEIGHT,N2\n')
            site_weights, site_n2 = cca.site_information()
            for i in range(len(site_weights)):
                numpy_fh.write('%s,%.10f,%.10f\n' % (
                    spp_plot_ids[i], site_weights[i], site_n2[i]))


class NumpyRDAOrdination(NumpyOrdination):
    def __init__(self, **kwargs):
        super(NumpyRDAOrdination, self).__init__(**kwargs)

    def run(self):
        # Convert the species and environment matrices to numpy rec arrays
        spp_df = pd.read_csv(self.spp_file)
        env_df = pd.read_csv(self.env_file)

        # Extract the plot IDs from both the species and environment matrices
        # and ensure that they are equal
        spp_plot_ids = spp_df[self.id_field]
        env_plot_ids = env_df[self.id_field]
        if not np.all(spp_plot_ids == env_plot_ids):
            err_msg = 'Species and environment plot IDs do not match'
            raise ValueError(err_msg)

        # Drop the ID column from both dataframes
        spp_df.drop(labels=[self.id_field], axis=1, inplace=True)
        env_df.drop(labels=[self.id_field], axis=1, inplace=True)

        # For the environment matrix, only keep the variables specified
        env_df = env_df[self.variables]

        # Convert these matrices to pure floating point arrays
        spp = spp_df.values.astype(float)
        env = env_df.values.astype(float)

        # Apply transformation if desired
        if self.species_transform == 'SQRT':
            spp = np.sqrt(spp)
        elif self.species_transform == 'LOG':
            spp = np.log(spp)

        # Create the RDA object
        rda = numpy_ordination.NumpyRDA(spp, env)

        with open(self.ord_file, 'w') as numpy_fh:
            # Eigenvalues
            numpy_fh.write('### Eigenvalues ###\n')
            for i, e in enumerate(rda.eigenvalues):
                numpy_fh.write(f'RDA{str(i + 1)},' + '%.10f' % e + '\n')
            numpy_fh.write('\n')

            # Print out variable means
            numpy_fh.write('### Variable Means ###\n')
            for i, m in enumerate(rda.env_means):
                numpy_fh.write('%s,%.10f\n' % (self.variables[i], m))
            numpy_fh.write('\n')

            # Print out environmental coefficients loadings
            numpy_fh.write('### Coefficient Loadings ###\n')
            header_str = ','.join(['RDA%d' % (i + 1) for i in range(rda.rank)])
            numpy_fh.write(f'VARIABLE,{header_str}' + '\n')
            for i, c in enumerate(rda.coefficients()):
                coeff = ','.join(['%.10f' % x for x in c])
                numpy_fh.write('%s,%s\n' % (self.variables[i], coeff))
            numpy_fh.write('\n')

            # Print out biplot scores
            numpy_fh.write('### Biplot Scores ###\n')
            header_str = ','.join(['RDA%d' % (i + 1) for i in range(rda.rank)])
            numpy_fh.write(f'VARIABLE,{header_str}' + '\n')
            for i, b in enumerate(rda.biplot_scores()):
                scores = ','.join(['%.10f' % x for x in b])
                numpy_fh.write('%s,%s\n' % (self.variables[i], scores))
            numpy_fh.write('\n')

            # Print out species centroids
            numpy_fh.write('### Species Centroids ###\n')
            header_str = ','.join(['RDA%d' % (i + 1) for i in range(rda.rank)])
            numpy_fh.write(f'SPECIES,{header_str}' + '\n')
            for i, c in enumerate(rda.species_centroids()):
                scores = ','.join(['%.10f' % x for x in c])
                numpy_fh.write('%s,%s\n' % (spp_df.columns[i], scores))
            numpy_fh.write('\n')

            # Print out species tolerances
            numpy_fh.write('### Species Tolerances ###\n')
            header_str = \
                    ','.join(['RDA%d' % (i + 1) for i in range(rda.rank)])
            numpy_fh.write(f'SPECIES,{header_str}' + '\n')
            for i, t in enumerate(rda.species_tolerances()):
                scores = ','.join(['%.21f' % x for x in t])
                numpy_fh.write('%s,%s\n' % (spp_df.columns[i], scores))
            numpy_fh.write('\n')

            # Print out miscellaneous species information
            numpy_fh.write('### Miscellaneous Species Information ###\n')
            numpy_fh.write('SPECIES,WEIGHT,N2\n')
            species_weights, species_n2 = rda.species_information()
            for i in range(len(species_weights)):
                numpy_fh.write('%s,%.10f,%.10f\n' % (
                    spp_df.columns[i], species_weights[i], species_n2[i]))
            numpy_fh.write('\n')

            # Print out site LC scores
            numpy_fh.write('### Site LC Scores ###\n')
            header_str = ','.join(['RDA%d' % (i + 1) for i in range(rda.rank)])
            numpy_fh.write(f'ID,{header_str}' + '\n')
            for i, s in enumerate(rda.site_lc_scores()):
                scores = ','.join(['%.10f' % x for x in s])
                numpy_fh.write('%d,%s\n' % (spp_plot_ids[i], scores))
            numpy_fh.write('\n')

            # Print out site WA scores
            numpy_fh.write('### Site WA Scores ###\n')
            header_str = ','.join(['RDA%d' % (i + 1) for i in range(rda.rank)])
            numpy_fh.write(f'ID,{header_str}' + '\n')
            for i, s in enumerate(rda.site_wa_scores()):
                scores = ','.join(['%.10f' % x for x in s])
                numpy_fh.write('%d,%s\n' % (spp_plot_ids[i], scores))
            numpy_fh.write('\n')

            # Miscellaneous site information
            numpy_fh.write('### Miscellaneous Site Information ###\n')
            numpy_fh.write('ID,WEIGHT,N2\n')
            site_weights, site_n2 = rda.site_information()
            for i in range(len(site_weights)):
                numpy_fh.write('%s,%.10f,%.10f\n' % (
                    spp_plot_ids[i], site_weights[i], site_n2[i]))


class NumpyEUCOrdination(NumpyOrdination):
    def __init__(self):
        super(NumpyEUCOrdination, self).__init__()
        print('Created a NumpyEUCOrdination')

    def run(self):
        raise NotImplementedError


class NumpyCCORAOrdination(NumpyOrdination):
    def __init__(self):
        super(NumpyCCORAOrdination, self).__init__()
        print('Created a NumpyCCORAOrdination')

    def run(self):
        raise NotImplementedError


class CanocoOrdination(Ordination):
    def __init__(self):
        super(CanocoOrdination, self).__init__()

    def run(self):
        raise NotImplementedError


class CanocoCCAOrdination(CanocoOrdination):
    def __init__(self):
        super(CanocoCCAOrdination, self).__init__()
        print('Created a CanocoCCAOrdination')

    def run(self):
        raise NotImplementedError


class CanocoRDAOrdination(CanocoOrdination):
    def __init__(self):
        super(CanocoRDAOrdination, self).__init__()
        print('Created a CanocoRDAOrdination')

    def run(self):
        raise NotImplementedError
