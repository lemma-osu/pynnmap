from __future__ import annotations

import click

from ..core.nn_finder import PixelNNFinder, PlotNNFinder, StoredPixelNNFinder
from ..core.prediction_output import DependentOutput, IndependentOutput
from ..diagnostics import diagnostic_wrapper as dw
from ..parser import parameter_parser_factory as ppf


def run_cross_validate(
    parser, nn_finder: PixelNNFinder | PlotNNFinder | StoredPixelNNFinder
) -> None:
    """
    Run cross-validation to create the neighbor/distance information.
    """
    # Calculate neighbors and distances
    neighbor_data = nn_finder.calculate_neighbors_cross_validation()

    # Calculate independent and dependent predictive accuracy
    independent_output = IndependentOutput(parser, neighbor_data)
    independent_output.write_zonal_records(parser.independent_zonal_pixel_file)
    independent_output.write_attribute_predictions(parser.independent_predicted_file)

    dependent_output = DependentOutput(parser, neighbor_data)
    dependent_output.write_zonal_records(parser.dependent_zonal_pixel_file)
    dependent_output.write_attribute_predictions(parser.dependent_predicted_file)
    dependent_output.write_nn_index_file(neighbor_data, parser.dependent_nn_index_file)


@click.command(name="cross-validate", short_help="Accuracy assessment for model plots")
@click.option(
    "--scale",
    type=click.Choice(["PIXEL", "PLOT"], case_sensitive=False),
    default="PIXEL",
    help="Calculate accuracy at pixel or plot scale",
)
@click.argument("parameter-file", type=click.Path(exists=True), required=True)
def main(scale, parameter_file):
    # Get the model parameters
    pixel_scale = scale == "PIXEL"
    parser = ppf.get_parameter_parser(parameter_file)

    # Determine the NNFinder type
    if pixel_scale:
        # Traditional route - extracting spatial information from plot locations
        if not parser.environmental_pixel_file:
            nn_finder = PixelNNFinder(parser)
        # Fast route - using pre-stored spatial information at pixel locations
        else:
            nn_finder = StoredPixelNNFinder(parser)
    else:
        nn_finder = PlotNNFinder(parser)

    # Run cross-validation to create the neighbor/distance information
    run_cross_validate(parser, nn_finder)

    # Calculate all accuracy diagnostics
    diagnostic_wrapper = dw.DiagnosticWrapper(parser)
    diagnostic_wrapper.run_accuracy_diagnostics()
