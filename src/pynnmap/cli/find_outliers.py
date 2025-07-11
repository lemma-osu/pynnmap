import click

from ..core.nn_finder import PixelNNFinder, StoredPixelNNFinder
from ..diagnostics import diagnostic_wrapper as dw
from ..parser import parameter_parser_factory as ppf
from .cross_validate import run_cross_validate


@click.command(
    name="find-outliers", short_help="Find plot outliers based on user-defined tests"
)
@click.argument("parameter-file", type=click.Path(exists=True), required=True)
def main(parameter_file):
    # Get the model parameters
    parser = ppf.get_parameter_parser(parameter_file)

    # Determine the NNFinder type
    nn_finder = (
        PixelNNFinder(parser)
        if not parser.environmental_pixel_file
        else StoredPixelNNFinder(parser)
    )

    # Run cross-validation to create the neighbor/distance information
    run_cross_validate(parser, nn_finder)

    # Run outlier analysis
    diagnostic_wrapper = dw.DiagnosticWrapper(parser)
    diagnostic_wrapper.run_outlier_diagnostics()
