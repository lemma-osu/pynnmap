import click

from pynnmap.core import model_run


@click.command(short_help='Run accuracy assessment on independent plots')
@click.argument(
    'parameter-file',
    type=click.Path(exists=True),
    required=True)
def new_targets(parameter_file):
    m = model_run.ModelRun(parameter_file)
    print('Not Implemented')
