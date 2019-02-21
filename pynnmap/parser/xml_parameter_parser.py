import os

import numpy as np
from lxml import etree
from lxml import objectify

from pynnmap.misc import utilities
from pynnmap.parser import xml_parser
from pynnmap.parser import parameter_parser


class XMLParameterParser(
        xml_parser.XMLParser, parameter_parser.ParameterParser):
    """
    Class for parsing full XML parameter files.  The model XML file must
    validate against the XML schema file, so we use this class to verify
    compliance.
    """

    def __init__(self, xml_file_name):
        """
        Initialize the NNXMLParser object by setting a reference to the
        XML parameter file.

        Parameters
        ----------
        xml_file_name : file
            name and location of XML model parameter file

        Returns
        -------
        None
        """

        # Call the base class constructors
        xml_parser.XMLParser.__init__(self, xml_file_name)
        parameter_parser.ParameterParser.__init__(self)

    def __repr__(self):
        """
        Return a string representation of the model parameter file
        Need to figure out how to do this...
        """
        return utilities.pretty_print(self.root)

    def write_tree(self, file_name):
        # Deannotate the tree
        objectify.deannotate(self.tree)
        etree.cleanup_namespaces(self.tree)

        # Ensure the newly created XML validates against the schema
        utilities.validate_xml(self.tree, self.xml_schema_file)

        # Write out the tree
        self.tree.write(file_name, pretty_print=True)

    def serialize(self):
        # Make the model directory if it doesn't already exist
        if not os.path.exists(self.model_directory):
            os.makedirs(self.model_directory)

        # Write out the file to this model directory
        out_file = os.path.join(self.model_directory, 'model.xml')
        self.write_tree(out_file)

    def _get_path(self, parameter, file_name):
        """
        Returns a fully qualified path of file locations given the model
        directory as the root.  If the parameter is under the
        'accuracy_assessment' or 'outlier_assessment' node, also append the
        accuracy_assessment_directory or outlier_assessment_directory as well.
        Should not be called directly, but is typically called from individual
        getters.

        Parameters
        ----------
        parameter : str
            The parameter which to look up and append to self.model_directory.
            The parameter should point to a file name

        Returns
        -------
        out : str
            The fully qualified path to the file location
        """

        md = self.model_directory
        aa_files = [
            'accuracy_assessment_report',
            'report_metadata_file',
            'local_accuracy_file',
            'species_accuracy_file',
            'vegclass_file',
            'vegclass_kappa_file',
            'vegclass_errmatrix_file',
            'area_estimate_file',
            'riemann_output_folder',
            'regional_accuracy_file',
            'validation_output_folder',
        ]
        outlier_files = [
            'nn_index_outlier_file',
            'vegclass_outlier_file',
            'vegclass_variety_file',
            'variable_deviation_file',
        ]
        riemann_files = [
            'hex_attribute_file',
            'hex_id_file',
            'hex_statistics_file',
        ]
        validation_files = [
            'validation_attribute_file',
        ]
        if parameter in aa_files:
            aa_dir = self.accuracy_assessment_folder
            out_path = os.path.join(aa_dir, file_name)
        elif parameter in outlier_files:
            outlier_dir = self.outlier_assessment_folder
            out_path = os.path.join(outlier_dir, file_name)
        elif parameter in riemann_files:
            riemann_dir = self.riemann_output_folder
            out_path = os.path.join(riemann_dir, file_name)
        elif parameter in validation_files:
            validation_dir = self.validation_output_folder
            out_path = os.path.join(validation_dir, file_name)
        else:
            out_path = os.path.join(md, file_name)

        return os.path.normpath(out_path)

    def get_spatial_filter(self):
        index = str(self.fl_elem.footprint_file).find('single')
        if index > -1:
            fltr = 'SINGLE'
        else:
            fltr = 'MULTI'
        return fltr

    # -------------------------------------------------------------------------
    # Parameter set
    # -------------------------------------------------------------------------

    @property
    def parameter_set(self):
        return str(self.root.parameter_set)

    @parameter_set.setter
    def parameter_set(self, value):
        self.root.parameter_set = value

    # -------------------------------------------------------------------------
    # File Locations
    # -------------------------------------------------------------------------

    @property
    def fl_elem(self):
        return self.root.file_locations

    @property
    def model_directory(self):
        return os.path.normpath(str(self.fl_elem.model_directory))

    @model_directory.setter
    def model_directory(self, value):
        self.fl_elem.model_directory = value

    @property
    def plot_dsn(self):
        return str(self.fl_elem.plot_dsn)

    @property
    def web_dsn(self):
        return str(self.fl_elem.web_dsn)

    @property
    def plot_id_field(self):
        return str(self.fl_elem.plot_id_field)

    @property
    def coordinate_file(self):
        return os.path.normpath(str(self.fl_elem.coordinate_file))

    @property
    def boundary_raster(self):
        return os.path.normpath(str(self.fl_elem.boundary_raster))

    @boundary_raster.setter
    def boundary_raster(self, value):
        self.fl_elem.boundary_raster = value

    @property
    def mask_raster(self):
        if self.fl_elem.find('mask_raster') is not None:
            return os.path.normpath(str(self.fl_elem.mask_raster))
        else:
            return ''

    @property
    def projection_file(self):
        return os.path.normpath(str(self.fl_elem.projection_file))

    @property
    def id_list_file(self):
        if self.fl_elem.find('id_list_file') is not None:
            file_name = str(self.fl_elem.id_list_file)
            return self._get_path('id_list_file', file_name)
        else:
            return ''

    @property
    def species_matrix_file(self):
        file_name = str(self.fl_elem.species_matrix_file)
        return self._get_path('species_matrix_file', file_name)

    @species_matrix_file.setter
    def species_matrix_file(self, value):
        self.fl_elem.species_matrix_file = value

    @property
    def environmental_matrix_file(self):
        file_name = str(self.fl_elem.environmental_matrix_file)
        return self._get_path('environmental_matrix_file', file_name)

    @environmental_matrix_file.setter
    def environmental_matrix_file(self, value):
        self.fl_elem.environmental_matrix_file = value

    @property
    def plot_independence_crosswalk_file(self):
        file_name = str(self.fl_elem.plot_independence_crosswalk_file)
        return self._get_path('plot_independence_crosswalk_file', file_name)

    @property
    def plot_year_crosswalk_file(self):
        file_name = str(self.fl_elem.plot_year_crosswalk_file)
        return self._get_path('plot_year_crosswalk_file', file_name)

    @property
    def stand_attribute_file(self):
        file_name = str(self.fl_elem.stand_attribute_file)
        return self._get_path('stand_attribute_file', file_name)

    @property
    def stand_metadata_file(self):
        file_name = str(self.fl_elem.stand_metadata_file)
        return self._get_path('stand_metadata_file', file_name)

    @property
    def footprint_file(self):
        return os.path.normpath(str(self.fl_elem.footprint_file))

    @property
    def independent_predicted_file(self):
        file_name = str(self.fl_elem.independent_predicted_file)
        return self._get_path('independent_predicted_file', file_name)

    @property
    def independent_zonal_pixel_file(self):
        file_name = str(self.fl_elem.independent_zonal_pixel_file)
        return self._get_path('independent_zonal_pixel_file', file_name)

    @property
    def dependent_predicted_file(self):
        file_name = str(self.fl_elem.dependent_predicted_file)
        return self._get_path('dependent_predicted_file', file_name)

    @property
    def dependent_zonal_pixel_file(self):
        file_name = str(self.fl_elem.dependent_zonal_pixel_file)
        return self._get_path('dependent_zonal_pixel_file', file_name)

    @property
    def dependent_nn_index_file(self):
        file_name = str(self.fl_elem.dependent_nn_index_file)
        return self._get_path('dependent_nn_index_file', file_name)

    # -------------------------------------------------------------------------
    # Model Parameters
    # -------------------------------------------------------------------------

    @property
    def mp_elem(self):
        return self.root.model_parameters

    @property
    def model_project(self):
        if self.mp_elem.find('model_project') is not None:
            return str(self.mp_elem.model_project)
        else:
            return ''

    @property
    def model_region(self):
        return int(self.mp_elem.model_region)

    @model_region.setter
    def model_region(self, value):
        self.mp_elem.model_region = value

    @property
    def model_year(self):
        return int(self.mp_elem.model_year)

    @model_year.setter
    def model_year(self, value):
        self.mp_elem.model_year = value

    @property
    def model_type(self):
        model_type_elem = self.mp_elem.model_type
        return str((model_type_elem.getchildren())[0].tag)

    @property
    def model_type_elem(self):
        return (self.mp_elem.model_type.getchildren())[0]

    @property
    def image_source(self):
        if self.model_type in self.imagery_model_types:
            mt_elem = self.model_type_elem
            return str(mt_elem.image_source)
        else:
            return ''

    @property
    def image_version(self):
        if self.model_type in self.imagery_model_types:
            mt_elem = self.model_type_elem
            return float(mt_elem.image_version)
        else:
            return 0.0

    @property
    def plot_image_crosswalk(self):
        if self.model_type in self.imagery_model_types:
            pi_crosswalk_elem = self.model_type_elem.plot_image_crosswalk
            if pi_crosswalk_elem.xpath('keyword'):
                return str(pi_crosswalk_elem.keyword)
            else:
                return [
                    (int(p.plot_year), int(p.image_year))
                    for p in pi_crosswalk_elem.getchildren()]
        else:
            return []

    @plot_image_crosswalk.setter
    def plot_image_crosswalk(self, records):
        if self.model_type in self.imagery_model_types:
            # Create a new XML tree of these pairs
            new_pi_crosswalk_elem = objectify.Element('plot_image_crosswalk')
            for rec in records.itertuples():
                child = (
                    etree.SubElement(new_pi_crosswalk_elem, 'plot_image_pair')
                )
                try:
                    child.plot_year = rec.PLOT_YEAR
                    child.image_year = rec.IMAGE_YEAR
                except ValueError:
                    err_msg = (
                        'Record does not have PLOT_YEAR or IMAGE_YEAR '
                        'attributes'
                    )
                    raise ValueError(err_msg)

            # Replace the old XML tree with the newly created one
            pi_crosswalk_elem = self.model_type_elem.plot_image_crosswalk
            parent = pi_crosswalk_elem.getparent()
            parent.replace(pi_crosswalk_elem, new_pi_crosswalk_elem)

        else:
            raise NotImplementedError

    @property
    def image_years(self):
        if self.model_type in self.imagery_model_types:
            pic = self.plot_image_crosswalk
            if isinstance(pic, str):
                return []
            else:
                return [x[1] for x in pic]
        else:
            return []

    @property
    def plot_years(self):
        if self.model_type not in self.imagery_model_types:
            py_elem = self.model_type_elem.plot_years
            return [int(x) for x in py_elem.getchildren()]
        else:
            pic = self.plot_image_crosswalk
            if isinstance(pic, str):
                return []
            else:
                return [x[0] for x in pic]

    @plot_years.setter
    def plot_years(self, year_list):
        if self.model_type not in self.imagery_model_types:

            # Create a new XML tree of these pairs
            new_py_elem = objectify.Element('plot_years')
            for year in year_list:
                etree.SubElement(new_py_elem, 'plot_year')
                new_py_elem.plot_year[-1] = year

            # Replace the old XML tree with the newly created one
            py_elem = self.model_type_elem.plot_years
            parent = py_elem.getparent()
            parent.replace(py_elem, new_py_elem)
        else:
            raise NotImplementedError

    @property
    def coincident_plots(self):
        return int(self.mp_elem.coincident_plots)

    @property
    def lump_table(self):
        return int(self.mp_elem.lump_table)

    @property
    def buffer(self):
        return int(self.mp_elem.buffer)

    @property
    def plot_types(self):
        pt_elem = self.mp_elem.plot_types
        return [str(x.tag).lower() for x in pt_elem.getchildren() if x == 1]

    @property
    def exclusion_codes(self):
        ec_elem = self.mp_elem.exclusion_codes
        return [str(x.tag).lower() for x in ec_elem.getchildren() if x == 0]

    # -------------------------------------------------------------------------
    # Ordination Parameters
    # -------------------------------------------------------------------------

    @property
    def op_elem(self):
        return self.root.ordination_parameters

    @property
    def ordination_program(self):
        return (self.op_elem.getchildren())[0].tag

    @property
    def ordination_program_element(self):
        return (self.op_elem.getchildren())[0]

    @property
    def distance_metric(self):
        program_elem = self.ordination_program_element
        return str(program_elem.distance_metric)

    @property
    def species_transform(self):
        program_elem = self.ordination_program_element
        try:
            return str(program_elem.species_transform)
        except ValueError:
            return None

    @property
    def species_downweighting(self):
        program_elem = self.ordination_program_element
        try:
            return int(program_elem.species_downweighting)
        except ValueError:
            return None

    def get_ordination_file(self):
        file_xwalk = {
            'vegan': 'vegan_file',
            'numpy': 'numpy_file',
        }
        ord_program = self.ordination_program
        program_elem = self.ordination_program_element
        try:
            file_element_tag = file_xwalk[ord_program]
            file_name = str(program_elem.find(file_element_tag))
            return self._get_path(file_element_tag, file_name)
        except KeyError:
            msg = 'No ordination file for ' + ord_program
            raise KeyError(msg)

    @property
    def variable_filter(self):
        return str(self.op_elem.variable_filter)

    def get_ordination_variables(self, model_year=None):
        ov_elem = self.op_elem.ordination_variables
        if (ov_elem.getchildren())[0].tag == 'keyword':
            return str((ov_elem.getchildren())[0])
        else:
            if model_year is None:
                model_year = self.model_year
            v_list = []
            for v in ov_elem.getchildren():
                if v.get('variable_type') == 'STATIC':
                    v_list.append((str(v.variable_name), str(v.variable_path)))
                elif v.get('variable_type') == 'TEMPORAL':
                    if int(v.get('model_year')) == model_year:
                        v_list.append(
                            (str(v.variable_name), str(v.variable_path)))
            return v_list

    def set_ordination_variables(self, records):
        # Create a new XML tree of these pairs
        new_ov_elem = objectify.Element('ordination_variables')
        for rec in records.itertuples():
            child = etree.SubElement(new_ov_elem, 'ordination_variable')
            try:
                child.variable_name = rec.VARIABLE_NAME
                child.variable_path = rec.VARIABLE_PATH
                if rec.MODEL_YEAR == 0:
                    child.set('variable_type', 'STATIC')
                else:
                    child.set('variable_type', 'TEMPORAL')
                    child.set('model_year', str(rec.MODEL_YEAR))
            except ValueError:
                err_msg = (
                    'Record does not have VARIABLE_NAME, VARIABLE_NAME or '
                    'MODEL_YEAR attributes'
                )
                raise ValueError(err_msg)

        # Replace the old XML tree with the newly created one
        ov_elem = self.op_elem.ordination_variables
        parent = ov_elem.getparent()
        parent.replace(ov_elem, new_ov_elem)

    def get_ordination_variable_names(self, model_year=None):
        ord_vars = self.get_ordination_variables(model_year=model_year)
        if isinstance(ord_vars, list):
            return [str(x[0]) for x in ord_vars]
        else:
            return None

    # -------------------------------------------------------------------------
    # Imputation Parameters
    # -------------------------------------------------------------------------

    @property
    def ip_elem(self):
        return self.root.imputation_parameters

    @property
    def number_axes(self):
        return int(self.ip_elem.number_axes)

    @property
    def use_axis_weighting(self):
        return int(self.ip_elem.use_axis_weighting)

    @property
    def k(self):
        return int(self.ip_elem.k)

    @property
    def max_neighbors(self):
        return int(self.ip_elem.max_neighbors)

    # -------------------------------------------------------------------------
    # Domain Parameters
    # -------------------------------------------------------------------------

    @property
    def dp_elem(self):
        return self.root.domain_parameters

    @property
    def domain(self):
        return str((self.dp_elem.getchildren())[0].tag)

    @property
    def domain_element(self):
        return (self.dp_elem.getchildren())[0]

    @property
    def point_x(self):
        if self.domain == 'point':
            return float(self.domain_element.x)
        else:
            return None

    @property
    def point_y(self):
        if self.domain == 'point':
            return float(self.domain_element.y)
        else:
            return None

    @property
    def list_points(self):
        points = []
        if self.domain == 'list':
            child_elem = (self.domain_element.getchildren())[0]
            if child_elem.tag == 'points':
                points = \
                    [(point.x, point.y) for point in child_elem.getchildren()]
            else:
                recs = utilities.csv2rec(str(child_elem))
                points = [(point.X, point.Y) for point in recs]
        return points

    @property
    def window_cell_size(self):
        if self.domain == 'window':
            return float(self.domain_element.cell_size)
        else:
            return None

    @property
    def envelope(self):
        if self.domain == 'window':
            d_elem = self.domain_element
            return [float(x) for x in d_elem.envelope.getchildren()]

    @envelope.setter
    def envelope(self, env):
        if self.domain == 'window':
            d_elem = self.domain_element
            d_elem.envelope.x_min = env[0]
            d_elem.envelope.y_min = env[1]
            d_elem.envelope.x_max = env[2]
            d_elem.envelope.y_max = env[3]

    @property
    def axes_file(self):
        if self.domain == 'window':
            return str(self.domain_element.output.axes_file)
        else:
            return None

    @property
    def neighbor_file(self):
        if self.domain == 'window':
            return str(self.domain_element.output.neighbor_file)
        else:
            return None

    @property
    def distance_file(self):
        if self.domain == 'window':
            return str(self.domain_element.output.distance_file)
        else:
            return None

    @property
    def output_format(self):
        if self.domain == 'window':
            return str(self.domain_element.output.output_format)
        else:
            return None

    @property
    def write_axes(self):
        return int(self.dp_elem.write_axes)

    @property
    def write_neighbors(self):
        return int(self.dp_elem.write_neighbors)

    @property
    def write_distances(self):
        return int(self.dp_elem.write_distances)

    # -------------------------------------------------------------------------
    # Accuracy Assessment
    # -------------------------------------------------------------------------

    @property
    def aa_elem(self):
        return self.root.accuracy_assessment

    @property
    def accuracy_assessment_folder(self):
        folder_name = str(self.aa_elem.accuracy_assessment_folder)
        return self._get_path('accuracy_assessment_folder', folder_name)

    @property
    def accuracy_assessment_report(self):
        if self.aa_elem.find('accuracy_assessment_report') is not None:
            file_name = str(self.aa_elem.accuracy_assessment_report)
            return self._get_path('accuracy_assessment_report', file_name)
        else:
            return ''

    @accuracy_assessment_report.setter
    def accuracy_assessment_report(self, value):
        if self.aa_elem.find('accuracy_assessment_report') is not None:
            self.aa_elem.accuracy_assessment_report = value

    @property
    def report_metadata_file(self):
        if self.aa_elem.find('report_metadata_file') is not None:
            file_name = str(self.aa_elem.report_metadata_file)
            return self._get_path('report_metadata_file', file_name)
        else:
            return ''

    @property
    def accuracy_diagnostics_element(self):
        if self.aa_elem.find('diagnostics') is not None:
            return self.aa_elem.diagnostics
        else:
            return None

    @property
    def accuracy_diagnostics(self):
        ade = self.accuracy_diagnostics_element
        if ade is not None:
            return [x.tag for x in ade.iterchildren()]
        else:
            return []

    @property
    def local_accuracy_file(self):
        if self.accuracy_diagnostics:
            ade = self.accuracy_diagnostics_element
            file_name = str(ade.local_accuracy.output_file)
            return self._get_path('local_accuracy_file', file_name)
        else:
            return None

    @property
    def species_accuracy_file(self):
        if self.accuracy_diagnostics:
            ade = self.accuracy_diagnostics_element
            file_name = str(ade.species_accuracy.output_file)
            return self._get_path('species_accuracy_file', file_name)
        else:
            return None

    @property
    def regional_assessment_year(self):
        if self.accuracy_diagnostics:
            ade = self.accuracy_diagnostics_element
            return int(ade.regional_accuracy.assessment_year)
        else:
            return None

    @property
    def area_estimate_file(self):
        if self.accuracy_diagnostics:
            ade = self.accuracy_diagnostics_element
            file_name = str(ade.regional_accuracy.area_estimate_file)
            return self._get_path('area_estimate_file', file_name)
        else:
            return None

    @property
    def regional_accuracy_file(self):
        if self.accuracy_diagnostics:
            ade = self.accuracy_diagnostics_element
            file_name = str(ade.regional_accuracy.output_file)
            return self._get_path('regional_accuracy_file', file_name)
        else:
            return None

    @property
    def vegclass_file(self):
        if self.accuracy_diagnostics:
            ade = self.accuracy_diagnostics_element
            file_name = str(ade.vegclass_accuracy.vegclass_file)
            return self._get_path('vegclass_file', file_name)
        else:
            return None

    @property
    def vegclass_kappa_file(self):
        if self.accuracy_diagnostics:
            ade = self.accuracy_diagnostics_element
            file_name = str(ade.vegclass_accuracy.vegclass_kappa_file)
            return self._get_path('vegclass_kappa_file', file_name)
        else:
            return None

    @property
    def vegclass_errmatrix_file(self):
        if self.accuracy_diagnostics:
            ade = self.accuracy_diagnostics_element
            file_name = str(ade.vegclass_accuracy.vegclass_errmatrix_file)
            return self._get_path('vegclass_errmatrix_file', file_name)
        else:
            return None

    @property
    def riemann_element(self):
        ade = self.accuracy_diagnostics_element
        if ade is not None:
            if ade.find('riemann_accuracy') is not None:
                return ade.riemann_accuracy
            else:
                return None

    @property
    def riemann_assessment_year(self):
        r_elem = self.riemann_element
        if r_elem is not None:
            return int(r_elem.assessment_year)
        else:
            return None

    @property
    def riemann_output_folder(self):
        r_elem = self.riemann_element
        if r_elem is not None:
            folder_name = str(r_elem.output_folder)
            return self._get_path('riemann_output_folder', folder_name)
        else:
            return ''

    @property
    def hex_attribute_file(self):
        r_elem = self.riemann_element
        if r_elem is not None:
            file_name = str(r_elem.hex_attribute_file)
            return self._get_path('hex_attribute_file', file_name)
        else:
            return ''

    @property
    def hex_id_file(self):
        r_elem = self.riemann_element
        if r_elem is not None:
            file_name = str(r_elem.hex_id_file)
            return self._get_path('hex_id_file', file_name)
        else:
            return ''

    @property
    def hex_statistics_file(self):
        r_elem = self.riemann_element
        if r_elem is not None:
            file_name = str(r_elem.hex_statistics_file)
            return self._get_path('hex_statistics_file', file_name)
        else:
            return ''

    @property
    def riemann_hex_resolutions(self):
        r_elem = self.riemann_element
        if r_elem is not None:
            hex_resolutions = []
            for elem in r_elem.hex_resolutions.iterchildren():
                field_name = str(elem.field_name)
                intercell_spacing = int(elem.intercell_spacing)
                area = float(elem.area)
                minimum_plots_per_hex = int(elem.minimum_plots_per_hex)
                hex_resolutions.append((
                    field_name, intercell_spacing, area, minimum_plots_per_hex)
                )
            return hex_resolutions
        else:
            return []

    @property
    def riemann_k_values(self):
        r_elem = self.riemann_element
        if r_elem is not None:
            k_values = [int(x) for x in r_elem.k_values.iterchildren()]
            return k_values
        else:
            return []

    @property
    def validation_element(self):
        ade = self.accuracy_diagnostics_element
        if ade is not None:
            if ade.find('validation_accuracy') is not None:
                return ade.validation_accuracy
            else:
                return None

    @property
    def validation_output_folder(self):
        v_elem = self.validation_element
        if v_elem is not None:
            folder_name = str(v_elem.output_folder)
            return self._get_path('validation_output_folder', folder_name)
        else:
            return ''

    @property
    def validation_attribute_file(self):
        v_elem = self.validation_element
        if v_elem is not None:
            file_name = str(v_elem.validation_attribute_file)
            return self._get_path('validation_attribute_file', file_name)
        else:
            return ''

    @property
    def include_in_report(self):
        out_list = []
        if self.accuracy_diagnostics:
            ade = self.accuracy_diagnostics_element
            for x in ade.iterchildren():
                try:
                    if x.include_in_report == 1:
                        out_list.append(x.tag)
                except AttributeError:
                    pass
        return out_list

    # -------------------------------------------------------------------------
    # Outlier Assessment
    # -------------------------------------------------------------------------

    @property
    def oa_elem(self):
        if self.root.find('outlier_assessment') is not None:
            return self.root.outlier_assessment
        else:
            return None

    @property
    def outlier_assessment_folder(self):
        folder_name = str(self.oa_elem.outlier_assessment_folder)
        return self._get_path('outlier_assessment_folder', folder_name)

    @property
    def outlier_diagnostics_element(self):
        oa_elem = self.oa_elem
        if oa_elem is not None:
            if self.oa_elem.find('diagnostics') is not None:
                return self.oa_elem.diagnostics
            else:
                return None
        else:
            return None

    @property
    def outlier_diagnostics(self):
        ode = self.outlier_diagnostics_element
        if ode is not None:
            return [x.tag for x in ode.iterchildren()]
        else:
            return []

    @property
    def nn_index_outlier_file(self):
        if self.outlier_diagnostics:
            ode = self.outlier_diagnostics_element
            file_name = str(ode.nn_index_outlier.nn_index_outlier_file)
            return self._get_path('nn_index_outlier_file', file_name)
        else:
            return None

    @property
    def index_threshold(self):
        if self.outlier_diagnostics:
            ode = self.outlier_diagnostics_element
            return float(ode.nn_index_outlier.index_threshold)
        else:
            return None

    @property
    def vegclass_outlier_file(self):
        if self.outlier_diagnostics:
            ode = self.outlier_diagnostics_element
            file_name = str(ode.vegclass_outlier.vegclass_outlier_file)
            return self._get_path('vegclass_outlier_file', file_name)
        else:
            return None

    @property
    def vegclass_variety_file(self):
        if self.outlier_diagnostics:
            ode = self.outlier_diagnostics_element
            file_name = str(ode.vegclass_variety.vegclass_variety_file)
            return self._get_path('vegclass_variety_file', file_name)
        else:
            return None

    @property
    def deviation_variables(self):
        deviation_variables = []
        if self.outlier_diagnostics:
            ode = self.outlier_diagnostics_element
            if ode.find('variable_deviation_outlier') is not None:
                vdo = ode.variable_deviation_outlier
                for elem in vdo.iterchildren(tag='variable'):
                    variable_name = str(elem.variable_name)
                    min_deviation = float(elem.min_deviation)
                    deviation_variables.append((variable_name, min_deviation))
        return deviation_variables

    @property
    def variable_deviation_file(self):
        if self.outlier_diagnostics:
            ode = self.outlier_diagnostics_element
            if ode.find('variable_deviation_outlier') is not None:
                file_name = str(ode.variable_deviation_outlier.output_file)
                return self._get_path('variable_deviation_file', file_name)
            else:
                return ''
        else:
            return ''
