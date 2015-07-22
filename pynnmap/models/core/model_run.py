import os
import sys
import subprocess
from models.diagnostics import diagnostic_wrapper as dw
from models.core import prediction_run as pr
from models.core import independent_run as ir
from models.core import dependent_run as dr
from models.parser import parameter_parser_factory as ppf


class ModelRun(object):

    def __init__(self, parameter_file):
        self.parser = ppf.get_parameter_parser(parameter_file)
        self.parameter_file = parameter_file

    def run(self):
        os.chdir(self.parser.model_directory)
        cmd = 'gnnrun ' + self.parameter_file
        subprocess.call(cmd)

    def post_process(self):
        from models.core import post_process_wrapper as ppw
        ppw.main(self.parser)

    def run_validation(self, run_accuracy_diagnostics=True,
        run_outlier_diagnostics=True):

        # Create a PredictionRun object
        prediction_run = pr.PredictionRun(self.parser)

        # Run the PredictionRun to create the neighbor/distance information
        prediction_run.calculate_neighbors_cross_validation()

        # Create an IndependentRun object
        independent_run = ir.IndependentRun(prediction_run)

        # Create the independent predicted data and zonal pixel file
        independent_run.create_predictions('LOC_ID')

        # Create a DependentRun object
        dependent_run = dr.DependentRun(prediction_run)

        # Create the dependent predicted data, zonal pixel file and
        # nn index file
        dependent_run.create_predictions()

        # If either type of diagnostic is requested, create the wrapper
        if run_accuracy_diagnostics or run_outlier_diagnostics:
            diagnostic_wrapper = dw.DiagnosticWrapper(self.parser)

            # Run the accuracy diagnostics if requested
            if run_accuracy_diagnostics:
                diagnostic_wrapper.run_accuracy_diagnostics()

            # Run the outlier diagnostics if present
            if run_outlier_diagnostics:
                diagnostic_wrapper.run_outlier_diagnostics()
                if self.parser.parameter_set == 'FULL':
                    diagnostic_wrapper.load_outliers()


def main():
    try:
        # model parameter file
        parameter_file = sys.argv[1]
        # flag for running GNN model (0=no, 1=yes)
        run_diag = int(sys.argv[2])
        # flag for running accuracy diagnostics (0=no, 1=yes)
        aa_diag = int(sys.argv[3])
        # flag for running outlier diagnostics (0=no, 1=yes)
        out_diag = int(sys.argv[4])

    except:
        print 'model_run.py usage:'
        print 'Parameter file: name and location of model input parameter file'
        print 'Full spatial model run flag: 0=no, 1=yes'
        print 'Accuracy diagnostics flag: 0=no, 1=yes'
        print 'Outlier diagnostics flag: 0=no, 1=yes'
    else:
        m = ModelRun(parameter_file)
        if run_diag == 1:
            m.run()
            m.post_process()
        if aa_diag == 1 or out_diag == 1:
            m.run_validation(aa_diag, out_diag)

if __name__ == '__main__':
    main()
