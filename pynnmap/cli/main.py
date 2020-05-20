from pkg_resources import iter_entry_points

import click
from click_plugins import with_plugins

import pynnmap


@with_plugins(list(iter_entry_points("pynnmap.cli_commands")))
@click.group()
@click.version_option(version=pynnmap.__version__, message="%(version)s")
def main_group():
    """
    Pynnmap command line interface
    """
