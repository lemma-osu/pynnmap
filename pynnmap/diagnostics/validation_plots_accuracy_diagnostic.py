import os

import numpy as np
import pandas as pd

from pynnmap.core.attribute_predictor import AttributePredictor
from pynnmap.core.nn_finder import NNFinder
from pynnmap.core.stand_attributes import StandAttributes
from pynnmap.diagnostics import diagnostic
from pynnmap.diagnostics import local_accuracy_diagnostic as lad
from pynnmap.diagnostics import vegetation_class_diagnostic as vcd
from pynnmap.misc.utilities import df_to_csv
from pynnmap.parser import parameter_parser as pp
from pynnmap.parser import xml_stand_metadata_parser as xsmp


class ValidationPlotsAccuracyDiagnostic(diagnostic.Diagnostic):
    def __init__(self, **kwargs):
        if "parameters" in kwargs:
            p = kwargs["parameters"]
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
                self.predicted_file = os.path.join(vd, os.path.basename(pf))
                self.local_accuracy_file = os.path.join(
                    vd, os.path.basename(laf)
                )
                self.vegclass_file = os.path.join(vd, os.path.basename(vf))
                self.vegclass_kappa_file = os.path.join(
                    vd, os.path.basename(vkf)
                )
                self.vegclass_errmatrix_file = os.path.join(
                    vd, os.path.basename(vef)
                )
            else:
                err_msg = "Passed object is not a ParameterParser object"
                raise ValueError(err_msg)
        else:
            err_msg = "Only ParameterParser objects may be passed."
            raise NotImplementedError(err_msg)

        # Ensure all input files are present
        files = [self.observed_file, self.stand_metadata_file]
        try:
            self.check_missing_files(files)
        except diagnostic.MissingConstraintError as e:
            e.message += "\nSkipping ValidationPlotsAccuracyDiagnostic\n"
            raise e

    def run_diagnostic(self):
        # Shortcut to the parameter parser
        p = self.parameter_parser

        # Read in the validation plots file
        val_df = pd.read_csv(self.observed_file)

        # Create a dictionary of plot ID to image year for these plots
        s = pd.Series(val_df.IMAGE_YEAR.values, index=val_df[self.id_field])
        id_x_year = dict(s.to_dict())

        # Create a NNFinder object
        finder = NNFinder(p)
        neighbor_data = finder.calculate_neighbors_at_ids(id_x_year)

        # Retrieve the predicted data for these plots.  In essence, we can
        # retrieve the dependent neighbors because these plot IDs are
        # guaranteed not to be in the model
        attr_fn = p.stand_attribute_file
        mp = xsmp.XMLStandMetadataParser(p.stand_metadata_file)
        attr_data = StandAttributes(attr_fn, mp, id_field=self.id_field)
        attr_predictor = AttributePredictor(attr_data)

        # Set weights correctly
        # TODO: Duplicate code with PredictionOutput.get_weights()
        w = p.weights
        if w is not None:
            if len(w) != p.k:
                raise ValueError("Length of weights does not equal k")
            w = np.array(w).reshape(1, len(w)).T

        # Calculate the predictions
        predictions = attr_predictor.calculate_predictions(
            neighbor_data, k=p.k, weights=w
        )

        # Get the predicted attributes
        df = attr_predictor.get_predicted_attributes_df(
            predictions, self.id_field
        )

        df_to_csv(df, self.predicted_file, index=True)

        # Run the LocalAccuracyDiagnostic on these files
        d = lad.LocalAccuracyDiagnostic(
            observed_file=self.observed_file,
            independent_predicted_file=self.predicted_file,
            stand_metadata_file=self.stand_metadata_file,
            local_accuracy_file=self.local_accuracy_file,
            id_field=self.id_field,
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
