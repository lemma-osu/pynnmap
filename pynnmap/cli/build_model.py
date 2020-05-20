import click

from pynnmap.core import ordination
from pynnmap.parser import parameter_parser_factory as ppf


# Dictionary of ordination program and distance metric to type
# of ordination object to instantiate
ORD_DICT = {
    ("vegan", "CCA"): ordination.VeganCCAOrdination,
    ("vegan", "RDA"): ordination.VeganRDAOrdination,
    ("vegan", "DBRDA"): ordination.VeganDBRDAOrdination,
    ("canoco", "CCA"): ordination.CanocoCCAOrdination,
    ("canoco", "RDA"): ordination.CanocoRDAOrdination,
    ("numpy", "CCA"): ordination.NumpyCCAOrdination,
    ("numpy", "RDA"): ordination.NumpyRDAOrdination,
    ("numpy", "EUC"): ordination.NumpyEUCOrdination,
    ("numpy", "CCORA"): ordination.NumpyCCORAOrdination,
}


@click.command(short_help="Build transformation model from spp/env data")
@click.argument("parameter-file", type=click.Path(exists=True), required=True)
def build_model(parameter_file):
    # Read in the parameters
    p = ppf.get_parameter_parser(parameter_file)

    # Create the ordination object
    ord_type = ORD_DICT[(p.ordination_program, p.distance_metric)]
    ord_obj = ord_type(parameters=p)

    # Run the ordination
    ord_obj.run()
