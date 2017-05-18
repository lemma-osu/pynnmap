import os
import subprocess

import click

from pynnmap.parser import parameter_parser_factory as ppf


@click.command(short_help='Apply model over spatial domain using kNN')
@click.argument(
    'parameter-file',
    type=click.Path(exists=True),
    required=True)
def impute(parameter_file):
    p = ppf.get_parameter_parser(parameter_file)
    os.chdir(p.model_directory)
    cmd = 'gnnrun ' + parameter_file
    subprocess.call(cmd)
