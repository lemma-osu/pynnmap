from __future__ import annotations

import click
import pandas as pd

from ..core.nn_finder import PixelNNFinder
from ..core.prediction_output import DependentOutput, IndependentOutput
from ..parser import parameter_parser_factory as ppf


def run_new_targets(
    parser,
    finder: PixelNNFinder,
    target_fcids: list[int],
    output_predicted_prefix: str,
    output_zonal_prefix: str,
) -> None:
    # Calculate neighbors for the new target plots
    neighbor_data = finder.calculate_neighbors_at_ids(target_fcids)

    # Calculate independent and dependent predictive accuracy
    independent_output = IndependentOutput(parser, neighbor_data)
    independent_zonal_pixel_file = f"{output_zonal_prefix}_pixel_independent.csv"
    independent_predicted_file = f"{output_predicted_prefix}_independent.csv"
    independent_output.write_zonal_records(independent_zonal_pixel_file)
    independent_output.write_attribute_predictions(independent_predicted_file)

    dependent_output = DependentOutput(parser, neighbor_data)
    dependent_zonal_pixel_file = f"{output_zonal_prefix}_pixel_dependent.csv"
    dependent_predicted_file = f"{output_predicted_prefix}_dependent.csv"
    dependent_output.write_zonal_records(dependent_zonal_pixel_file)
    dependent_output.write_attribute_predictions(dependent_predicted_file)


@click.command(
    name="new-targets", short_help="Run accuracy assessment on independent plots"
)
@click.argument("parameter-file", type=click.Path(exists=True), required=True)
@click.argument("target-plot-file", type=click.Path(exists=True), required=True)
@click.argument("output-predicted-prefix", type=click.STRING, required=True)
@click.argument("output-zonal-prefix", type=click.STRING, required=True)
def main(
    parameter_file,
    target_plot_file,
    output_predicted_prefix,
    output_zonal_prefix,
):
    # Get the model parameters
    parser = ppf.get_parameter_parser(parameter_file)

    # Create a NNFinder derived object - either pixel or plot
    finder = PixelNNFinder(parser)

    # Read in the target IDs
    id_field = parser.plot_id_field
    target_fcids = pd.read_csv(target_plot_file, usecols=[id_field])[id_field].tolist()

    run_new_targets(
        parser, finder, target_fcids, output_predicted_prefix, output_zonal_prefix
    )
