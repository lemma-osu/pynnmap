import click

from pynnmap.cli.cross_validate import run_cross_validate
from pynnmap.core.nn_finder import PixelNNFinder
from pynnmap.diagnostics import diagnostic_wrapper as dw
from pynnmap.parser import parameter_parser_factory as ppf


@click.command(short_help="Find plot outliers based on user-defined tests")
@click.argument("parameter-file", type=click.Path(exists=True), required=True)
def find_outliers(parameter_file):
    # Get the model parameters
    parser = ppf.get_parameter_parser(parameter_file)

    # Create a PixelNNFinder object
    finder = PixelNNFinder(parser)

    # Run cross-validation to create the neighbor/distance information
    run_cross_validate(parser, finder)

    # Run outlier analysis
    diagnostic_wrapper = dw.DiagnosticWrapper(parser)
    diagnostic_wrapper.run_outlier_diagnostics()
