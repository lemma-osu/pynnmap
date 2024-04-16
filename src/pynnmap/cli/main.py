import click
from click_plugins import with_plugins
from importlib_metadata import entry_points

from ..__about__ import __version__


@with_plugins(list(entry_points(group="pynnmap")))
@click.group()
@click.version_option(version=__version__, message="%(version)s")
def main_group():
    """
    Pynnmap command line interface
    """
