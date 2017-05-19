import click

from pynnmap.core import dependent_run as dr
from pynnmap.core import independent_run as ir
from pynnmap.core import prediction_run as pr
from pynnmap.diagnostics import diagnostic_wrapper as dw
from pynnmap.parser import parameter_parser_factory as ppf


@click.command(short_help='Find plot outliers based on user-defined tests')
@click.argument(
    'parameter-file',
    type=click.Path(exists=True),
    required=True)
def find_outliers(parameter_file):
    p = ppf.get_parameter_parser(parameter_file)

    # Create a PredictionRun object
    prediction_run = pr.PredictionRun(p)

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

    diagnostic_wrapper = dw.DiagnosticWrapper(p)
    diagnostic_wrapper.run_outlier_diagnostics()
