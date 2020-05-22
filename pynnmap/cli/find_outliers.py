import click

from pynnmap.core.prediction_output import DependentOutput, IndependentOutput
from pynnmap.core.nn_finder import NNFinder
from pynnmap.diagnostics import diagnostic_wrapper as dw
from pynnmap.parser import parameter_parser_factory as ppf


@click.command(short_help="Find plot outliers based on user-defined tests")
@click.argument("parameter-file", type=click.Path(exists=True), required=True)
def find_outliers(parameter_file):
    # Get the model parameters
    p = ppf.get_parameter_parser(parameter_file)

    # Create a NNFinder object
    finder = NNFinder(p)

    # Run cross-validation to create the neighbor/distance information
    neighbor_data = finder.calculate_neighbors_cross_validation()

    # Create an IndependentOutput object
    independent_output = IndependentOutput(p)

    # Create the independent predicted data and zonal pixel file
    independent_output.create_predictions(neighbor_data)

    # Create a DependentOutput object
    dependent_output = DependentOutput(p)

    # Create the dependent predicted data, zonal pixel file and nn index file
    dependent_output.create_predictions(neighbor_data)

    diagnostic_wrapper = dw.DiagnosticWrapper(p)
    diagnostic_wrapper.run_outlier_diagnostics()
