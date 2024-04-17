import click


@click.command(
    name="new-targets", short_help="Run accuracy assessment on independent plots"
)
@click.argument("parameter-file", type=click.Path(exists=True), required=True)
def main():
    print("Not Implemented")
