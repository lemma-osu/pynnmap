import os
import subprocess

import click

from ..parser import parameter_parser_factory as ppf


@click.command(name="impute", short_help="Apply model over spatial domain using kNN")
@click.argument("parameter-file", type=click.Path(exists=True), required=True)
def main(parameter_file):
    p = ppf.get_parameter_parser(parameter_file)
    os.chdir(p.model_directory)
    cmd = f"gnnrun {parameter_file}"
    subprocess.call(cmd)
