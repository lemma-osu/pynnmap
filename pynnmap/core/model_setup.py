import os
import re
import shutil
import sys
from optparse import OptionParser

import numpy as np
from lxml import etree
from lxml import objectify
from matplotlib import mlab

from pynnmap.core import ordination
from pynnmap.database import plot_database
from pynnmap.database import web_database
from pynnmap.misc import utilities
from pynnmap.parser import parameter_parser_factory as ppf


class ModelSetup(object):

    # Dictionary of ordination program and distance metric to type
    # of ordination object to instantiate
    ord_type = {
        ('vegan', 'CCA'): ordination.VeganCCAOrdination,
        ('vegan', 'RDA'): ordination.VeganRDAOrdination,
        ('vegan', 'DBRDA'): ordination.VeganDBRDAOrdination,
        ('canoco', 'CCA'): ordination.CanocoCCAOrdination,
        ('canoco', 'RDA'): ordination.CanocoRDAOrdination,
        ('numpy', 'CCA'): ordination.NumpyCCAOrdination,
        ('numpy', 'RDA'): ordination.NumpyRDAOrdination,
        ('numpy', 'EUC'): ordination.NumpyEUCOrdination,
        ('numpy', 'CCORA'): ordination.NumpyCCORAOrdination,
    }

    def __init__(self, parameter_file, **kwargs):
        """
        Initializes a ModelSetup instance from a parameter file

        Parameters
        ----------
        parameter_file : str
            File which stores model parameters.  This is either an XML or INI
            file.  As of now, only the XML logic has been implemented.

        Returns
        -------
        None
        """
        self.parameter_parser = ppf.get_parameter_parser(parameter_file)

        # Ensure that the parameter parser has been fully fleshed out before
        # retrieving the modeling files.  If the parameter_parser references
        # a prototype file, a new instance of a parameter_parser is
        # returned that represents the fully fleshed out version
        self.parameter_parser = self.parameter_parser.get_parameters(**kwargs)
        p = self.parameter_parser

        # Write out the model file
        p.serialize()

        # Create a ModelDatabase instance
        if p.parameter_set == 'FULL':
            self.plot_db = plot_database.PlotDatabase(
                p.model_type, p.model_region, p.buffer, p.model_year,
                p.image_source, p.image_version, dsn=p.plot_dsn
            )

    def create_species_plot_count_file(self):
        p = self.parameter_parser

        # Store list of plot IDs into a string if this variable hasn't
        # yet been created
        if not hasattr(self, 'id_str'):
            self.id_str = self._get_id_string()

        spp_plot_table = self.plot_db.get_species_plot_counts(self.id_str)
        spp_plot_file = '%s/%s_spp_plot_counts.csv' % (
            p.model_directory, p.model_type)
        if p.model_type in p.imagery_model_types:
            utilities.rec2csv(spp_plot_table, spp_plot_file)
        else:
            # Create 2 ID strings for non-imagery models, one with inventory
            # and Ecoplots and one with inventory plots only
            try:
                ecoplot_index = p.plot_types.index('ecoplot')
            except ValueError:
                # If Ecoplots are not already in the list, create another ID
                # string with them included
                plot_types_w_eco = p.plot_types
                plot_types_w_eco.append('ecoplot')
                plot_types_w_eco_str = ','.join(plot_types_w_eco)
                id_str2 = self._get_id_string(plot_types_w_eco_str)
                id_eco = 2
            else:
                # If Ecoplot are already in the list, create another ID
                # string without them included
                plot_types_wo_eco = p.plot_types
                plot_types_wo_eco.remove('ecoplot')
                plot_types_wo_eco_str = ','.join(plot_types_wo_eco)
                id_str2 = self._get_id_string(plot_types_wo_eco_str)
                id_eco = 1

            spp_plot_table2 = self.plot_db.get_species_plot_counts(id_str2)

            # Join the plot counts w/ Ecoplots to the plot counts w/o Ecoplots
            if id_eco == 1:
                joined_spp_plot_table = mlab.rec_join(
                    'SPP_LAYER', spp_plot_table, spp_plot_table2, 'leftouter')
            else:
                joined_spp_plot_table = mlab.rec_join(
                    'SPP_LAYER', spp_plot_table2, spp_plot_table, 'leftouter')
            utilities.rec2csv(joined_spp_plot_table, spp_plot_file)

    def create_modeling_files(
            self, create_ordination_matrices=False, run_ordination=False,
            create_attribute_data=False, create_area_estimates=False,
            create_report_metadata=False, create_hex_attribute_file=False,
            create_validation_attribute_file=False,
            **kwargs):
        """
        Using the parameter file, create modeling files.  A number of
        keyword args can be used here to help guide modeling file input.  For
        the first three keyword parameters (model_directory, model_region,
        and model_year), no defaults are used and they are passed on for the
        ParameterParser instance to handle.

        Parameters
        ----------
        model_directory : str
            Directory where modeling files will be created

        model_region : int
            LEMMA model region

        model_year : int
            Modeling year

        create_spp_plot_counts : bool
            Flag whether to create a species plot count file
            Defaults to False

        create_ordination_matrices : bool
            Flag whether to create ordination matrices
            Defaults to False

        run_ordination : bool
            Flag whether to run ordination
            Defaults to False

        create_attribute_data : bool
            Flag whether to create attribute data
            Defaults to False

        create_area_estimates : bool
            Flag whether to create observed area estimate data
            Defaults to False

        create_report_metadata : bool
            Flag whether to create data used in accuracy assessment report
            Defaults to False

        create_hex_attribute_file : bool
            Flag whether to create the file containing plot IDs by Hexagon IDs
            and continuous summary variables for Annual plots
            Defaults to False

        create_validation_attribute_file : bool
            Flag whether to create the file containing structure and species
            attributes for the plots to be used in the validation diagnostics
            Defaults to False

        Returns
        -------
        None
        """

        # Create the optional outputs
        if create_ordination_matrices:
            self.create_ordination_matrices()
        if run_ordination:
            self.run_ordination()
        if create_attribute_data:
            self.create_attribute_data()
        if create_report_metadata:
            self.create_report_metadata()
        if create_area_estimates:
            self.create_area_estimates()
        if create_hex_attribute_file:
            self.create_hex_attribute_file()
        if create_validation_attribute_file:
            self.create_validation_attribute_file()

    def _get_id_string(self, plot_types=''):
        """
        Return a string of all IDs used in this model parameterization

        Parameters
        ----------
        None

        Returns
        -------
        id_str : str
            Comma-delimited string of IDs used in this model
        """

        p = self.parameter_parser

        # Look for the presence of id_list_file in the parameters; if present,
        # use this - otherwise use parameters to define the IDs
        if p.id_list_file:
            return self._read_id_list_file(p.id_list_file)

        # Format list parameters as comma-delimited strings
        if plot_types == '':
            plot_types = ','.join(p.plot_types)
        exclusion_codes = ','.join(p.exclusion_codes)
        plot_years = ','.join([str(x) for x in p.plot_years])
        image_years = ','.join([str(x) for x in p.image_years])

        # Get the plot ID string
        id_str = self.plot_db.get_plot_list(
            p.coincident_plots, p.lump_table, plot_years, image_years,
            plot_types, exclusion_codes)

        return id_str

    def _read_id_list_file(self, id_list_file):
        data = utilities.csv2rec(id_list_file)
        return ','.join([str(x[0]) for x in data])

    def create_ordination_matrices(self):
        """
        Create the species and environmental matrices needed for ordination
        modeling.  Write these files out to the location as specified in the
        parameter file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        p = self.parameter_parser

        # Format list parameters as comma-delimited strings
        plot_years = ','.join([str(x) for x in p.plot_years])
        image_years = ','.join([str(x) for x in p.image_years])
        ordination_variables = ','.join(p.get_ordination_variable_names())

        # Store list of plot IDs into a string if this variable hasn't
        # yet been created
        if not hasattr(self, 'id_str'):
            self.id_str = self._get_id_string()

        # Get the species matrix and write it out
        spp_table = self.plot_db.get_species_matrix(
            self.id_str, 'ORDINATION', p.lump_table)
        spp_file = p.species_matrix_file
        utilities.rec2csv(spp_table, spp_file)

        # Get the environmental matrix and write it out
        env_table = self.plot_db.get_environmental_matrix(
            self.id_str, plot_years, image_years, ordination_variables)
        env_file = p.environmental_matrix_file
        utilities.rec2csv(env_table, env_file)

    def run_ordination(self):
        """
        Run the ordination and output to files as specified in the
        parameter file

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        p = self.parameter_parser

        # Ensure the ordination matrices have already been created; otherwise
        # create them now
        spp_file = p.species_matrix_file
        env_file = p.environmental_matrix_file
        for f in (spp_file, env_file):
            if not os.path.exists(f):
                self.create_ordination_matrices()
                break

        # Create the ordination object
        ord_type = self.ord_type[(p.ordination_program, p.distance_metric)]
        ord = ord_type(parameters=p)

        # Run the ordination
        ord.run()

        # Thin plots to just one plot per location
        self.delete_duplicate_plots()

    def delete_duplicate_plots(self):
        """
        For locations where all plots have exactly the same CCA scores,
        delete all plots except for the plot measured closest to the model
        year.
        """

        p = self.parameter_parser

        # Store list of plot IDs into a string if this variable hasn't
        # yet been created
        if not hasattr(self, 'id_str'):
            self.id_str = self._get_id_string()

        # Get the list of plots to delete
        delete_list = self.plot_db.get_duplicate_plots_to_remove(self.id_str)

        if len(delete_list) > 0:
            # If there are plots to delete, run this code to delete them
            # Read the ordination file into a list
            ord_file = p.get_ordination_file()
            in_fh = open(ord_file, 'r')
            all_lines = in_fh.readlines()
            in_fh.close()

            # rename original file.  This asssumes that the file has a file
            # extension and we will prepend '_orig' before the file
            # extension
            pos = ord_file.rfind('.')
            orig_file = ord_file[0:pos] + '_orig' + ord_file[pos:]
            if os.path.exists(orig_file):
                os.remove(orig_file)
            os.rename(ord_file, orig_file)

            # Now open a new file to hold the modified output - this will
            # have the name of the original ordination file
            out_fh = open(ord_file, 'w')

            # regex for plot scores
            beg_re = re.compile('^###\s+Site\s+LC\s+Scores\s+###.*')
            end_re = re.compile('^\s*$')

            # flag for whether we're in the plot score section
            in_section = False
            header_line = False

            # Loop through all lines and write out everything except for
            # plots in the delete list
            for line in all_lines:
                if not in_section:
                    out_fh.write(line)
                    # If we've entered the plot score section change the
                    # flag to True
                    if beg_re.match(line):
                        in_section = True
                        header_line = True
                        continue

                if in_section:
                    if header_line:
                        out_fh.write(line)
                        header_line = False
                        continue

                    # See if we've encountered the end of the plot section
                    if end_re.match(line):
                        out_fh.write(line)
                        in_section = False
                    else:
                        line_array = line.split(',')
                        plot_id = int(line_array[0])

                        # Only write out the plot if this plot is NOT in
                        # the delete list
                        if plot_id not in delete_list:
                            out_fh.write(line)

            out_fh.close()

    def create_attribute_data(self):
        """
        Create the attribute data which has both stand level attributes as
        well as species data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        p = self.parameter_parser

        # Store list of plot IDs into a string if this variable hasn't
        # yet been created
        if not hasattr(self, 'id_str'):
            self.id_str = self._get_id_string()

        # Get the attribute data
        attribute_table = self.plot_db.get_attribute_data(self.id_str)
        attribute_file = p.stand_attribute_file
        utilities.rec2csv(attribute_table, attribute_file)
        field_names = attribute_table.dtype.names
        self.create_attribute_metadata(field_names)

    def create_attribute_metadata(self, field_names):
        """
        Create the attribute metadata based on the field_names parameter

        Parameters
        ----------
        field_names: list
            Field names for which to get metadata

        Returns
        -------
        None
        """

        p = self.parameter_parser

        # Get the metadata associated with the attribute data
        structure_fields, structure_codes = \
            self.plot_db.get_structure_metadata(p.model_project)
        species_fields = \
            self.plot_db.get_species_metadata()

        # Create the metadata XML
        xml_schema_file = \
            'http://lemma.forestry.oregonstate.edu/xml/stand_attributes.xsd'
        root_str = """
            <attributes
                xmlns:xsi="%s"
                xsi:noNamespaceSchemaLocation="%s"/>
        """
        root_str = root_str % (
            'http://www.w3.org/2001/XMLSchema-instance',
            xml_schema_file
        )
        root_elem = objectify.fromstring(root_str)

        for n in field_names:
            n = n.upper()
            other_fields = {}
            try:
                r = structure_fields[structure_fields.FIELD_NAME == n][0]
                other_fields['SPECIES_ATTR'] = 0
                other_fields['PROJECT_ATTR'] = r.PROJECT_ATTR
                other_fields['ACCURACY_ATTR'] = r.ACCURACY_ATTR
            except IndexError:
                try:
                    r = species_fields[species_fields.FIELD_NAME == n][0]
                    other_fields['SPECIES_ATTR'] = 1
                    other_fields['PROJECT_ATTR'] = 1
                    other_fields['ACCURACY_ATTR'] = 1
                except IndexError:
                    err_msg = n + ' has no metadata'
                    print err_msg
                    continue

            # Add the attribute element
            attribute_elem = etree.SubElement(root_elem, 'attribute')

            # Add all metadata common to both structure and species recarrays
            fields = (
                'FIELD_NAME', 'FIELD_TYPE', 'UNITS', 'DESCRIPTION',
                'SHORT_DESCRIPTION')
            for f in fields:
                child = etree.SubElement(attribute_elem, f.lower())
                attribute_elem[child.tag] = getattr(r, f)

            # Add special fields customized for structure and species
            fields = ('SPECIES_ATTR', 'PROJECT_ATTR', 'ACCURACY_ATTR')
            for f in fields:
                child = etree.SubElement(attribute_elem, f.lower())
                attribute_elem[child.tag] = other_fields[f]

            # Print out codes if they exist
            if r.CODED is True:
                codes_elem = etree.SubElement(attribute_elem, 'codes')
                try:
                    c_records = \
                        structure_codes[structure_codes.FIELD_NAME == n]
                except IndexError:
                    # try:
                    #     c_records = \
                    #         species_codes[species_codes.FIELD_NAME == n]
                    # except IndexError:
                    err_msg = 'Codes were not found for ' + n
                    print err_msg
                    continue

                for c_rec in c_records:
                    code_elem = etree.SubElement(codes_elem, 'code')
                    c_fields = ('CODE_VALUE', 'DESCRIPTION', 'LABEL')
                    for c in c_fields:
                        child = etree.SubElement(code_elem, c.lower())
                        code_elem[child.tag] = getattr(c_rec, c)

        tree = root_elem.getroottree()
        objectify.deannotate(tree)
        etree.cleanup_namespaces(tree)

        # Ensure that this tree validates against the schema file
        utilities.validate_xml(tree, xml_schema_file)

        # Write out this metadata file
        metadata_file = p.stand_metadata_file
        tree.write(metadata_file, pretty_print=True)

    def create_area_estimates(self):
        """
        Create the observed area estimates file which stores plot based
        estimates of stand variables

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        # Set an alias for the parameter parser
        p = self.parameter_parser

        # Get the 'eslf_only' flag from the exclusion codes
        # Removed since the DB proc does not have an option for
        # using only plot with ESLF codes anymore
#        if 'eslf_only' in p.exclusion_codes:
#            eslf_only = 0
#        else:
#            eslf_only = 1

        # Get the area expansion data
        area_estimate_table, nf_hectares, ns_hectares = \
            self.plot_db.get_area_estimates(p.regional_assessment_year)

        # Create nonforest and nonsampled records to be concatenated with the
        # existing area_estimate_table recarray.  The nonforest record
        # has an ID of -10001 and the nonsampled record has an ID of -10002
        id_field = p.plot_id_field
        new_recs = np.recarray(2, dtype=area_estimate_table.dtype)
        for f in new_recs.dtype.names:
            for rec in new_recs:
                setattr(rec, f, 0.0)
        setattr(new_recs[0], id_field, -10002)
        setattr(new_recs[0], 'HECTARES', ns_hectares)
        setattr(new_recs[1], id_field, -10001)
        setattr(new_recs[1], 'HECTARES', nf_hectares)
        area_estimate_table = np.hstack((new_recs, area_estimate_table))

        # Write out to a CSV file
        area_estimate_file = p.area_estimate_file
        aa_dir = os.path.dirname(area_estimate_file)
        if not os.path.exists(aa_dir):
            os.makedirs(aa_dir)
        utilities.rec2csv(area_estimate_table, area_estimate_file)

    def create_report_metadata(self):
        """
        Create the XML file containing metadata to be written into
        the accuracy assessment report

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        p = self.parameter_parser

        # Connect to the lemma web database
        web_db = web_database.WebDatabase(p.model_project,
                                          p.model_region, p.web_dsn)

        # Create the XML
        xml_schema_file = \
            'http://lemma.forestry.oregonstate.edu/xml/report_metadata.xsd'

        root_str = """
            <report_metadata
                xmlns:xsi="%s"
                xsi:noNamespaceSchemaLocation="%s"/>
        """
        root_str = root_str % (
            'http://www.w3.org/2001/XMLSchema-instance',
            xml_schema_file
        )

        root_elem = objectify.fromstring(root_str)

        # Get the model region overview
        mr_overview = web_db.get_model_region_info()

        field_names = mr_overview.dtype.names
        overview_elem = etree.SubElement(root_elem, 'overview')
        for f in field_names:
            child = etree.SubElement(overview_elem, f.lower())
            overview_elem[child.tag] = getattr(mr_overview[0], f)

        # Get contact info for people associated with this project
        people_info = web_db.get_people_info()

        field_names = people_info.dtype.names
        people_elem = etree.SubElement(root_elem, 'contact_information')
        for person in people_info:
            person_elem = etree.SubElement(people_elem, 'contact')
            for f in field_names:
                child = etree.SubElement(person_elem, f.lower())
                person_elem[child.tag] = getattr(person, f)

        # Store list of plot IDs into a string if this variable hasn't
        # yet been created
        if not hasattr(self, 'id_str'):
            self.id_str = self._get_id_string()

        # Subset the string of plot IDs to thin to one plot at a
        # location just for locations that have the exact same spectral
        # values for all plot measurements (i.e. places where the
        # imagery has been stabilized
        delete_list = self.plot_db.get_duplicate_plots_to_remove(self.id_str)
        if len(delete_list) > 0:
            id_list_subset = [int(x) for x in self.id_str.split(",")]

            for id in delete_list:
                try:
                    id_list_subset.remove(id)
                # if the ID is not in the list, go on to the next ID
                except ValueError:
                    continue

            # turn subsetted id_list into a string
            id_str_subset = ','.join(map(str, id_list_subset))
        else:
            id_str_subset = self.id_str

        # Get the plot data sources
        data_sources = self.plot_db.get_plot_data_source_summary(id_str_subset)
        field_names = data_sources.dtype.names
        data_sources_elem = etree.SubElement(root_elem, 'plot_data_sources')

        # Create subelements for each unique plot data source
        for ds in np.unique(data_sources.DATA_SOURCE):
            data_source_elem = \
                etree.SubElement(data_sources_elem, 'plot_data_source')
            child = etree.SubElement(data_source_elem, 'data_source')
            data_source_elem[child.tag] = ds
            child = etree.SubElement(data_source_elem, 'description')
            descriptions = \
                data_sources[np.where(data_sources.DATA_SOURCE == ds)]
            description = np.unique(descriptions)
            data_source_elem[child.tag] = description['DESCRIPTION'][0]
            years_elem = etree.SubElement(data_source_elem, 'assessment_years')
            recs = data_sources[np.where(data_sources.DATA_SOURCE == ds)]

            # Create subelements for each plot assessment years for
            # this data source
            for rec in recs:
                year_elem = etree.SubElement(years_elem, 'year')
                child = etree.SubElement(year_elem, 'assessment_year')
                year_elem[child.tag] = getattr(rec, 'ASSESSMENT_YEAR')
                child = etree.SubElement(year_elem, 'plot_count')
                year_elem[child.tag] = getattr(rec, 'PLOT_COUNT')

        # Get the species scientific and common names
        species_names = \
            self.plot_db.get_species_names(self.id_str, p.lump_table)

        field_names = species_names.dtype.names
        species_names_elem = etree.SubElement(root_elem, 'species_names')
        for species_name in species_names:
            species_name_elem = etree.SubElement(species_names_elem, 'species')
            for f in field_names:
                child = etree.SubElement(species_name_elem, f.lower())
                species_name_elem[child.tag] = getattr(species_name, f)

        # Get the ordination variable descriptions
        ordination_vars = ','.join(p.get_ordination_variable_names())
        ordination_descr = \
            self.plot_db.get_ordination_variable_descriptions(ordination_vars)
        field_names = ordination_descr.dtype.names
        ord_vars_elem = etree.SubElement(root_elem, 'ordination_variables')
        for ord_var in ordination_descr:
            ord_var_elem = \
                etree.SubElement(ord_vars_elem, 'ordination_variable')
            for f in field_names:
                child = etree.SubElement(ord_var_elem, f.lower())
                ord_var_elem[child.tag] = getattr(ord_var, f)

        tree = root_elem.getroottree()
        objectify.deannotate(tree)
        etree.cleanup_namespaces(tree)

        # Ensure that this tree validates against the schema file
        utilities.validate_xml(tree, xml_schema_file)

        # Write XML to file
        report_metadata_file = p.report_metadata_file
        aa_dir = os.path.dirname(report_metadata_file)
        if not os.path.exists(aa_dir):
            os.makedirs(aa_dir)
        tree.write(report_metadata_file, pretty_print=True)

    def create_hex_attribute_file(self):
        """
        Create the file containing hexagon IDs and continuous stand
        attributes for forested Annual plots to be used in Riemann
        accuracy diagnostics

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        p = self.parameter_parser

        # Format list parameters as comma-delimited strings
        plot_years = ','.join([str(x) for x in p.plot_years])
        image_years = ','.join([str(x) for x in p.image_years])

        # Get the crosswalk of plot IDs to Hex IDs and write it out
        hex_attributes = self.plot_db.get_hex_attributes(
            p.riemann_assessment_year, plot_years, image_years)

        hex_attribute_file = p.hex_attribute_file
        riemann_dir = os.path.dirname(hex_attribute_file)
        if not os.path.exists(riemann_dir):
            os.makedirs(riemann_dir)
        utilities.rec2csv(hex_attributes, hex_attribute_file)

    def create_validation_attribute_file(self):
        """
        Create the file containing structure and species
        attributes for the plots to be used in the validation
        accuracy diagnostics

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        p = self.parameter_parser

        # Get the attributes for the validation plots
        validation_attributes = self.plot_db.get_validation_attributes()

        # Write these data out to the validation_attribute_file
        validation_attribute_file = p.validation_attribute_file
        validation_dir = p.validation_output_folder
        if not os.path.exists(validation_dir):
            os.makedirs(validation_dir)
        utilities.rec2csv(validation_attributes, validation_attribute_file)


def main():
    # First argument is the parameter file and is required
    model_xml = sys.argv[1]

    parser = OptionParser()
    parser.add_option(
        '-r',
        dest='model_regions',
        help='List or range of GNN model regions for which to setup models')
    parser.add_option(
        '-d',
        dest='root_directory',
        help='Location of root directory for storing model results')
    parser.add_option(
        '-y',
        dest='years',
        help=(
            'Range of model years to run, or comma-delimited list of '
            'years to run'))
    parser.add_option(
        '-i',
        dest='image_version',
        help='Imagery version')
    parser.add_option(
        '-s',
        dest='model_specs',
        help='Model specifications used for directory naming; eg. tc_only_k1')
    parser.add_option(
        '-m',
        dest='run_mode',
        help=(
            'Run model: spatial (full spatial run) or point (plot '
            'locations only)'))

    options = (parser.parse_args())[0]

    # If root directory is specified on command line, use it
    # Otherwise, parse the model_xml file and construct the path from the
    # project parameter
    if options.root_directory:
        project_dir = options.root_directory
    else:
        p = ppf.get_parameter_parser(model_xml)
        project_dir = 'L:/orcawa/%s/models' % (p.model_project.lower())

    iv = options.image_version
    ms = options.model_specs

    # If optional arguments were supplied, loop through all years and
    # model regions specified
    if options.model_regions:
        # If regions are specified as a range, unpack into a list
        if options.model_regions.find('-') > 0:
            region_list = []
            regions_split = options.model_regions.split('-')
            start_region = int(regions_split[0])
            end_region = int(regions_split[1])
            for r in range(start_region, end_region + 1):
                region_list.append(r)
        else:
            # If regions are specified in a comma-delimited string,
            # add each region to a list
            region_list = options.model_regions.split(',')
            # Convert to ints
            region_list = map(int, region_list)

        # Loop through all model regions
        for region in region_list:
            if options.years:
                # If regions are specified as a range, unpack into a list
                if options.years.find('-') > 0:
                    year_list = []
                    years_split = options.years.split('-')
                    start_year = int(years_split[0])
                    end_year = int(years_split[1])
                    for y in range(start_year, end_year + 1):
                        year_list.append(y)

                else:
                    # If years are specified in a comma-delimited string,
                    # add each year to a list
                    year_list = options.years.split(',')
                    # Convert to ints
                    year_list = map(int, year_list)

                root_dir = 'mr%d/%s/%s' % (region, iv, ms)
                root_dir = '/'.join((project_dir, root_dir))

                # If run_mode option is not specified, default to spatial
                # mode and do not modify root directory
                if options.run_mode:
                    if options.run_mode == 'point':
                        # If run_mode is specified as 'point', add another
                        # subdirectory named point
                        root_dir = '/'.join([root_dir, 'point'])

                # Loop through all model
                for year in year_list:

                    # Create new folders, rename existing folder to _old
                    # and delete existing _old folders if necessary
                    model_dir = '/'.join([root_dir, str(year)])

                    if os.path.exists(model_dir):
                        # If the directory already exists see if there
                        # is already an _old directory
                        old_dir = model_dir + '_old'
                        if os.path.exists(old_dir):
                            # Delete the existing old directory
                            shutil.rmtree(old_dir)
                        # Rename existing model_dir to old_dir
                        os.rename(model_dir, old_dir)

                    # Create new empty model directory
                    os.makedirs(model_dir)

                    # Create modeling files
                    model = ModelSetup(
                        model_xml,
                        model_directory=model_dir,
                        model_region=region,
                        model_year=year
                    )

                    model.create_modeling_files(
                        model_directory=model_dir,
                        model_region=region,
                        model_year=year,
                        create_ordination_matrices=True,
                        run_ordination=True,
                        create_attribute_data=True,
                        create_area_estimates=True,
                        create_report_metadata=True,
                        create_hex_attribute_file=True,
                        create_validation_attribute_file=False
                    )

if __name__ == '__main__':
    main()
