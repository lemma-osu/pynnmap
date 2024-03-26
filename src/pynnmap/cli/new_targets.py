import click


@click.command(short_help="Run accuracy assessment on independent plots")
@click.argument("parameter-file", type=click.Path(exists=True), required=True)
def new_targets():
    print("Not Implemented")
