import numpy as np

from pynnmap.diagnostics import diagnostic
from pynnmap.misc import utilities
from pynnmap.misc import classification_accuracy as ca
from pynnmap.parser import parameter_parser as pp


class VegetationClassDiagnostic(diagnostic.Diagnostic):

    qmd_breaks = np.array([0.0, 2.5, 25.0, 37.5, 50.0, 75.0])
    qmd_classes = np.array([1, 2, 3, 4, 5, 6])

    cancov_breaks = np.array([0.0, 10.0, 40.0, 70.0])
    cancov_classes = np.array([1, 2, 3, 4])

    def __init__(self, **kwargs):
        if 'parameters' in kwargs:
            p = kwargs['parameters']
            if isinstance(p, pp.ParameterParser):
                self.observed_file = p.stand_attribute_file
                self.predicted_file = p.independent_predicted_file
                self.vegclass_file = p.vegclass_file
                self.vegclass_kappa_file = p.vegclass_kappa_file
                self.vegclass_errmatrix_file = p.vegclass_errmatrix_file
                self.id_field = p.plot_id_field
            else:
                err_msg = 'Passed object is not a ParameterParser object'
                raise ValueError(err_msg)
        else:
            try:
                self.observed_file = kwargs['observed_file']
                self.predicted_file = kwargs['independent_predicted_file']
                self.vegclass_file = kwargs['vegclass_file']
                self.vegclass_kappa_file = kwargs['vegclass_kappa_file']
                self.vegclass_errmatrix_file = \
                    kwargs['vegclass_errmatrix_file']
                self.id_field = kwargs['id_field']
            except KeyError:
                err_msg = 'Not all required parameters were passed'
                raise ValueError(err_msg)

        # Ensure all input files are present
        files = [self.observed_file, self.predicted_file]
        try:
            self.check_missing_files(files)
        except diagnostic.MissingConstraintError as e:
            e.message += '\nSkipping VegetationClassDiagnostic\n'
            raise e

    def get_vegclass(self, rec, field_dict):
        """
        Return the VEGCLASS for this plot given the input data
        """

        qmd = getattr(rec, field_dict['qmd'])
        cancov = getattr(rec, field_dict['cancov'])
        bah = getattr(rec, field_dict['bah'])

        # Size class based on qmd
        index = np.digitize([qmd], self.qmd_breaks)
        sc = self.qmd_classes[index - 1][0]

        # Cover class based on cancov
        index = np.digitize([cancov], self.cancov_breaks)
        cc = self.cancov_classes[index - 1][0]

        # Step through logic to determine VEGCLASS
        if cc == 1:
            vc = 1
        elif cc == 2:
            vc = 2
        else:
            if bah >= 0.65:
                if sc <= 2:
                    vc = 3
                else:
                    vc = 4
            elif bah >= 0.2 and bah < 0.65:
                if sc <= 2:
                    vc = 5
                elif sc >= 3 and sc <= 4:
                    vc = 6
                else:
                    vc = 7
            else:
                if sc <= 2:
                    vc = 8
                elif sc >= 3 and sc <= 4:
                    vc = 9
                elif sc == 5:
                    vc = 10
                else:
                    vc = 11
        return vc

    def vegclass_aa(
            self, obs, prd, id_field='FCID', qmd_field='QMD_DOM',
            bah_field='BAH_PROP', cancov_field='CANCOV'):
        """
        Given observed and predicted recarrays from GNN output, this
        function creates a data file of observed and predicted vegetation
        class (VEGCLASS)
        """

        # Create a dict of needed field names
        field_dict = {
            'id': id_field,
            'qmd': qmd_field,
            'bah': bah_field,
            'cancov': cancov_field,
        }

        # Ensure that the fields identified exist in the recarrays
        for field in field_dict.values():
            if field not in obs.dtype.names:
                raise ValueError('Cannot find ' + field + ' in observed file')
            if field not in prd.dtype.names:
                raise ValueError('Cannot find ' + field + ' in predicted file')

        # Get the vegetation class for every observed and predicted record
        vc_dict = {}
        for rec in obs:
            id_val = getattr(rec, field_dict['id'])
            vc_dict[id_val] = {}
            obs_vc = self.get_vegclass(rec, field_dict)
            vc_dict[id_val]['obs_vc'] = obs_vc

        for rec in prd:
            id_val = getattr(rec, field_dict['id'])
            prd_vc = self.get_vegclass(rec, field_dict)
            vc_dict[id_val]['prd_vc'] = prd_vc

        return vc_dict

    def run_diagnostic(self):

        # Read the observed and predicted files into numpy recarrays
        obs = utilities.csv2rec(self.observed_file)
        prd = utilities.csv2rec(self.predicted_file)

        # Subset the observed data just to the IDs that are in the
        # predicted file
        obs_keep = np.in1d(
            getattr(obs, self.id_field), getattr(prd, self.id_field))
        obs = obs[obs_keep]

        # Calculate VEGCLASS for both the observed and predicted data
        vc_dict = self.vegclass_aa(obs, prd, id_field=self.id_field)

        # Print out the vegclass file
        vc_fh = open(self.vegclass_file, 'w')
        vc_fh.write(','.join((self.id_field, 'OBSERVED', 'PREDICTED')) + '\n')

        # Print out the observed and predicted vegetation classes
        for id_val in sorted(vc_dict.keys()):
            obs_vc = vc_dict[id_val]['obs_vc']
            prd_vc = vc_dict[id_val]['prd_vc']
            out_list = ['%d' % x for x in (id_val, obs_vc, prd_vc)]
            vc_fh.write(','.join(out_list) + '\n')
        vc_fh.close()

        # Create the vegetation class kappa and error matrix files
        vc_xml = 'L:/resources/code/xml/vegclass.xml'
        ca.classification_accuracy(
            self.vegclass_file, vc_xml, kappa_file=self.vegclass_kappa_file,
            err_matrix_file=self.vegclass_errmatrix_file)
