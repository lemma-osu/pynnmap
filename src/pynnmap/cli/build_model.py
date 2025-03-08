import click

from ..core import ordination
from ..parser import parameter_parser_factory as ppf

# Dictionary of ordination program and distance metric to type
# of ordination object to instantiate
ORD_DICT = {
    ("vegan", "CCA"): ordination.VeganCCAOrdination,
    ("vegan", "RDA"): ordination.VeganRDAOrdination,
    ("vegan", "DBRDA"): ordination.VeganDBRDAOrdination,
    ("numpy", "RDA"): ordination.NumpyRDAOrdination,
    ("sknnr", "CCA"): ordination.SknnrCCAOrdination,
}


@click.command(
    name="build-model", short_help="Build transformation model from spp/env data"
)
@click.argument("parameter-file", type=click.Path(exists=True), required=True)
def main(parameter_file):
    # Read in the parameters
    p = ppf.get_parameter_parser(parameter_file)

    # Create the ordination object
    ord_type = ORD_DICT[(p.ordination_program, p.distance_metric)]
    ord_params = ordination.OrdinationParameters.from_parser(p)
    ord_obj = ord_type(parameters=ord_params)

    # Run the ordination
    ord_obj.run()
