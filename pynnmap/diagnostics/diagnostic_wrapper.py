import os

from models.diagnostics.diagnostic import MissingConstraintError
from models.diagnostics import local_accuracy_diagnostic as lad
from models.diagnostics import regional_accuracy_diagnostic as rad
from models.diagnostics import riemann_accuracy_diagnostic as riemann
from models.diagnostics import species_accuracy_diagnostic as sad
from models.diagnostics import vegetation_class_diagnostic as vcd
from models.diagnostics import validation_plots_accuracy_diagnostic as vpad

from models.diagnostics import nn_index_outlier_diagnostic as niod
from models.diagnostics import vegetation_class_outlier_diagnostic as vcod
from models.diagnostics import vegetation_class_variety_diagnostic as vcvd
from models.diagnostics import variable_deviation_outlier_diagnostic as vdod

from models.diagnostics.report import lemma_accuracy_report as lar
from models.diagnostics import outlier_formatter as out
from models.misc import utilities


class DiagnosticWrapper(object):

    # Dictionary of diagnostic name to diagnostic class
    diagnostic_type = {
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

    # Dictionary of outlier name to outlier formatter class
    outlier_formatter = {
        'vegclass_variety': out.VegclassVarietyFormatter,
        'vegclass_outlier': out.VegclassOutlierFormatter,
        'nn_index_outlier': out.NNIndexFormatter
    }

    def __init__(self, parameter_parser):
        self.parameter_parser = parameter_parser

    def run_accuracy_diagnostics(self):
        p = self.parameter_parser

        # Ensure that the accuracy assessment folder has been created
        if p.accuracy_diagnostics:
            if not os.path.exists(p.accuracy_assessment_folder):
                os.makedirs(p.accuracy_assessment_folder)

        # Run each accuracy diagnostic
        for d in p.accuracy_diagnostics:
            try:
                diagnostic = (self.diagnostic_type[d])(parameters=p)
                diagnostic.run_diagnostic()
            except MissingConstraintError as e:
                print e.message

        # Create the AA report if desired
        if p.accuracy_assessment_report:
            report = lar.LemmaAccuracyReport(p)
            report.create_accuracy_report()

    def run_outlier_diagnostics(self):
        p = self.parameter_parser

        # Ensure that the outlier folder has been created
        if p.outlier_diagnostics:
            if not os.path.exists(p.outlier_assessment_folder):
                os.makedirs(p.outlier_assessment_folder)

        for d in p.outlier_diagnostics:
            try:
                diagnostic = (self.diagnostic_type[d])(p)
                diagnostic.run_diagnostic()
            except MissingConstraintError as e:
                print e.message

    def load_outliers(self):
        p = self.parameter_parser

        #read in outlier files and push results to DB
        for d in p.outlier_diagnostics:
            outlier_diag = (self.diagnostic_type[d])(p)
            outlier_file = outlier_diag.get_outlier_filename()
            outlier_formatter = (self.outlier_formatter[d])(p)
            out_rec = utilities.csv2rec(outlier_file)
            if out_rec is not None:
                outlier_formatter.load_outliers(out_rec)
