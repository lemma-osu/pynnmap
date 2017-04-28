import os

import numpy as np

from pynnmap.core import prediction_run
from pynnmap.diagnostics import diagnostic
from pynnmap.diagnostics import local_accuracy_diagnostic as lad
from pynnmap.diagnostics import vegetation_class_diagnostic as vcd
from pynnmap.misc import utilities
from pynnmap.parser import parameter_parser as pp


class ValidationPlotsAccuracyDiagnostic(diagnostic.Diagnostic):

    def __init__(self, **kwargs):
        if 'parameters' in kwargs:
            p = kwargs['parameters']
            if isinstance(p, pp.ParameterParser):
                self.observed_file = p.validation_attribute_file
                self.stand_metadata_file = p.stand_metadata_file
                self.parameter_parser = p
                self.id_field = p.plot_id_field

                # For the remainder of the files, get the values from the
                # parameter parser, but strip off the directory information
                # and prepend the validation directory
                pf = p.independent_predicted_file
                laf = p.local_accuracy_file
                vf = p.vegclass_file
                vkf = p.vegclass_kappa_file
                vef = p.vegclass_errmatrix_file

                vd = p.validation_output_folder
                self.predicted_file = \
                    os.path.join(vd, os.path.basename(pf))
                self.local_accuracy_file = \
                    os.path.join(vd, os.path.basename(laf))
                self.vegclass_file = \
                    os.path.join(vd, os.path.basename(vf))
                self.vegclass_kappa_file = \
                    os.path.join(vd, os.path.basename(vkf))
                self.vegclass_errmatrix_file = \
                    os.path.join(vd, os.path.basename(vef))
            else:
                err_msg = 'Passed object is not a ParameterParser object'
                raise ValueError(err_msg)
        else:
            err_msg = 'Only ParameterParser objects may be passed.'
            raise NotImplementedError(err_msg)

        # Ensure all input files are present
        files = [self.observed_file, self.stand_metadata_file]
        try:
            self.check_missing_files(files)
        except diagnostic.MissingConstraintError as e:
            e.message += '\nSkipping ValidationPlotsAccuracyDiagnostic\n'
            raise e

    def run_diagnostic(self):

        # Shortcut to the parameter parser
        p = self.parameter_parser

        # Read in the validation plots file
        validation_plots = utilities.csv2rec(self.observed_file)

        # Create a dictionary of plot ID to image year for these plots
        id_x_year = \
            dict((x[self.id_field], x.IMAGE_YEAR) for x in validation_plots)

        # Create a PredictionRun instance
        pr = prediction_run.PredictionRun(p)

        # Get the neighbors and distances for these IDs
        pr.calculate_neighbors_at_ids(id_x_year, id_field=self.id_field)

        # Retrieve the predicted data for these plots.  In essence, we can
        # retrieve the dependent neighbors because these plot IDs are
        # guaranteed not to be in the model
        prediction_generator = pr.calculate_predictions_at_k(
            k=p.k, id_field=self.id_field, independent=False)

        # Open the predicted file and write out the field names
        out_fh = open(self.predicted_file, 'w')
        out_fh.write(self.id_field + ',' + ','.join(pr.attrs) + '\n')

        # Write out the predictions
        for plot_prediction in prediction_generator:

            # Write this record to the predicted attribute file
            pr.write_predicted_record(plot_prediction, out_fh)

        # Close this file
        out_fh.close()

        # Run the LocalAccuracyDiagnostic on these files
        d = lad.LocalAccuracyDiagnostic(
            observed_file=self.observed_file,
            independent_predicted_file=self.predicted_file,
            stand_metadata_file=self.stand_metadata_file,
            local_accuracy_file=self.local_accuracy_file,
            id_field=self.id_field
        )
        d.run_diagnostic()

        # Run the VegetationClassDiagnostic on these files
        d = vcd.VegetationClassDiagnostic(
            observed_file=self.observed_file,
            independent_predicted_file=self.predicted_file,
            vegclass_file=self.vegclass_file,
            vegclass_kappa_file=self.vegclass_kappa_file,
            vegclass_errmatrix_file=self.vegclass_errmatrix_file,
            id_field=self.id_field,
        )
        d.run_diagnostic()
