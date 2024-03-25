import os

from pynnmap.diagnostics.local_accuracy_diagnostic import (
    LocalAccuracyDiagnostic,
)
from pynnmap.diagnostics.error_matrix_diagnostic import ErrorMatrixDiagnostic
from pynnmap.diagnostics.nn_index_outlier_diagnostic import (
    NNIndexOutlierDiagnostic,
)
from pynnmap.diagnostics.regional_accuracy_diagnostic import (
    RegionalAccuracyDiagnostic,
)
from pynnmap.diagnostics.riemann_accuracy_diagnostic import (
    RiemannAccuracyDiagnostic,
)
from pynnmap.diagnostics.species_accuracy_diagnostic import (
    SpeciesAccuracyDiagnostic,
)
from pynnmap.diagnostics.validation_plots_accuracy_diagnostic import (
    ValidationPlotsAccuracyDiagnostic,
)
from pynnmap.diagnostics.variable_deviation_outlier_diagnostic import (
    VariableDeviationOutlierDiagnostic,
)
from pynnmap.diagnostics.vegetation_class_diagnostic import (
    VegetationClassDiagnostic,
)
from pynnmap.diagnostics.vegetation_class_outlier_diagnostic import (
    VegetationClassOutlierDiagnostic,
)
from pynnmap.diagnostics.vegetation_class_variety_diagnostic import (
    VegetationClassVarietyDiagnostic,
)
from pynnmap.misc import utilities

# Dictionary of diagnostic name to diagnostic class
DIAGNOSTIC_TYPE = {
    "local_accuracy": LocalAccuracyDiagnostic,
    "error_matrix_accuracy": ErrorMatrixDiagnostic,
    "regional_accuracy": RegionalAccuracyDiagnostic,
    "riemann_accuracy": RiemannAccuracyDiagnostic,
    "species_accuracy": SpeciesAccuracyDiagnostic,
    "vegclass_accuracy": VegetationClassDiagnostic,
    "validation_accuracy": ValidationPlotsAccuracyDiagnostic,
    "nn_index_outlier": NNIndexOutlierDiagnostic,
    "vegclass_outlier": VegetationClassOutlierDiagnostic,
    "vegclass_variety": VegetationClassVarietyDiagnostic,
    "variable_deviation_outlier": VariableDeviationOutlierDiagnostic,
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
            if d not in DIAGNOSTIC_TYPE:
                print(f"Key {d} is not a diagnostic")
                continue
            try:
                kls = DIAGNOSTIC_TYPE[d]
                diagnostic = kls.from_parameter_parser(p)
                diagnostic.run_diagnostic()
            except utilities.MissingConstraintError as e:
                print(e)

    def run_outlier_diagnostics(self):
        p = self.parameter_parser

        # Ensure that the outlier folder has been created
        if p.outlier_diagnostics and not os.path.exists(
            p.outlier_assessment_folder
        ):
            os.makedirs(p.outlier_assessment_folder)

        for d in p.outlier_diagnostics:
            if d not in DIAGNOSTIC_TYPE:
                print(f"Key {d} is not a diagnostic")
                continue
            try:
                kls = DIAGNOSTIC_TYPE[d]
                diagnostic = kls(p)
                diagnostic.run_diagnostic()
            except utilities.MissingConstraintError as e:
                print(e)
