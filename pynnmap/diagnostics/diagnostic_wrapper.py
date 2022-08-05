import os

from pynnmap.diagnostics import local_accuracy_diagnostic as lad
from pynnmap.diagnostics import nn_index_outlier_diagnostic as niod
from pynnmap.diagnostics import regional_accuracy_diagnostic as rad
from pynnmap.diagnostics import riemann_accuracy_diagnostic as riemann
from pynnmap.diagnostics import species_accuracy_diagnostic as sad
from pynnmap.diagnostics import validation_plots_accuracy_diagnostic as vpad
from pynnmap.diagnostics import variable_deviation_outlier_diagnostic as vdod
from pynnmap.diagnostics import vegetation_class_diagnostic as vcd
from pynnmap.diagnostics import vegetation_class_outlier_diagnostic as vcod
from pynnmap.diagnostics import vegetation_class_variety_diagnostic as vcvd
from pynnmap.misc import utilities

# Dictionary of diagnostic name to diagnostic class
DIAGNOSTIC_TYPE = {
    'local_accuracy': lad.LocalAccuracyDiagnostic,
    'regional_accuracy': rad.RegionalAccuracyDiagnostic,
    'riemann_accuracy': riemann.RiemannAccuracyDiagnostic,
    'species_accuracy': sad.SpeciesAccuracyDiagnostic,
    'vegclass_accuracy': vcd.VegetationClassDiagnostic,
    'validation_accuracy': vpad.ValidationPlotsAccuracyDiagnostic,
    'nn_index_outlier': niod.NNIndexOutlierDiagnostic,
    'vegclass_outlier': vcod.VegetationClassOutlierDiagnostic,
    'vegclass_variety': vcvd.VegetationClassVarietyDiagnostic,
    'variable_deviation_outlier': vdod.VariableDeviationOutlierDiagnostic,
}


class DiagnosticWrapper(object):

    def __init__(self, parameter_parser):
        self.parameter_parser = parameter_parser

    def run_accuracy_diagnostics(self):
        p = self.parameter_parser

        # Ensure that the accuracy assessment folder has been created
        if p.accuracy_diagnostics and not os.path.exists(
            p.accuracy_assessment_folder
        ):
            os.makedirs(p.accuracy_assessment_folder)

        # Run each accuracy diagnostic
        for d in p.accuracy_diagnostics:
            try:
                kls = DIAGNOSTIC_TYPE[d]
                diagnostic = kls.from_parameter_parser(p)
                diagnostic.run_diagnostic()
            except KeyError as e:
                print(f'Key {d} is not a diagnostic')
            except utilities.MissingConstraintError as e:
                print(e.message)

    def run_outlier_diagnostics(self):
        p = self.parameter_parser

        # Ensure that the outlier folder has been created
        if p.outlier_diagnostics and not os.path.exists(
            p.outlier_assessment_folder
        ):
            os.makedirs(p.outlier_assessment_folder)

        for d in p.outlier_diagnostics:
            try:
                diagnostic = (DIAGNOSTIC_TYPE[d])(p)
                diagnostic.run_diagnostic()
            except MissingConstraintError as e:
                print(e.message)
