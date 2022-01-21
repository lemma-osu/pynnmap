import click

from pynnmap.diagnostics import diagnostic_wrapper as dw
from pynnmap.parser import parameter_parser_factory as ppf


@click.command(
    name="run-diagnostics",
    short_help="Run diagnostics only"
)
@click.argument("parameter-file", type=click.Path(exists=True), required=True)
def main(parameter_file):
    # Get the model parameters
    p = ppf.get_parameter_parser(parameter_file)

    # Run the diagnostic wrapper - this assumes that the
    # prediction files have already been created
    diagnostic_wrapper = dw.DiagnosticWrapper(p)
    diagnostic_wrapper.run_accuracy_diagnostics()
