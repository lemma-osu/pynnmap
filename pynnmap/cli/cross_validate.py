import click

from pynnmap.core.prediction_output import DependentOutput, IndependentOutput
from pynnmap.core.nn_finder import NNFinder
from pynnmap.diagnostics import diagnostic_wrapper as dw
from pynnmap.parser import parameter_parser_factory as ppf


@click.command(short_help="Accuracy assessment for model plots")
@click.argument("parameter-file", type=click.Path(exists=True), required=True)
def cross_validate(parameter_file):
    # Get the model parameters
    parser = ppf.get_parameter_parser(parameter_file)

    # Create a NNFinder object
    finder = NNFinder(parser)

    # Run cross-validation to create the neighbor/distance information
    neighbor_data = finder.calculate_neighbors_cross_validation()

    # Calculate independent and dependent predictive accuracy
    output = IndependentOutput(parser, neighbor_data)
    output.write_zonal_records(parser.independent_zonal_pixel_file)
    output.write_attribute_predictions(parser.independent_predicted_file)

    output = DependentOutput(parser, neighbor_data)
    output.write_zonal_records(parser.dependent_zonal_pixel_file)
    output.write_attribute_predictions(parser.dependent_predicted_file)
    output.write_nn_index_file(neighbor_data, parser.dependent_nn_index_file)

    # Calculate all accuracy diagnostics
    diagnostic_wrapper = dw.DiagnosticWrapper(parser)
    diagnostic_wrapper.run_accuracy_diagnostics()
