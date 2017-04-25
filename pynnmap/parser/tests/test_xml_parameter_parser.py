from os.path import normpath
import unittest
from pynnmap.parser import xml_parameter_parser as xpp


class SppszPrototypeParserTest(unittest.TestCase):

    def setUp(self):
        sppsz_proto_name = 'L:/resources/code/xml/sppsz_parameters.xml'
        self.sppsz_prototype = xpp.XMLParameterParser(sppsz_proto_name)

    def test_normal_init(self):
        obj = self.sppsz_prototype.create_model_xml(
            'D:/test', 224, 1996)
        obj.write_tree('./data/sppsz_full.xml')

    def test_get_properties(self):
        obj = self.sppsz_prototype
        props = (
            # Parameter set
            ('parameter_set', 'PROTOTYPE'),

            # File locations
            ('model_directory',
                normpath('L:/model_dir')),
            ('plot_dsn', 'rocky2lemma'),
            ('web_dsn', 'web_lemma'),
            ('coordinate_file',
                normpath('C:/_coords/fcid_coords.csv')),
            ('boundary_raster',
                normpath('L:/orcawa/cmonster/bounds/mr224')),
            ('mask_raster',
                normpath('L:/orcawa/spatial/masks/gap_nfmask30')),
            ('projection_file',
                normpath((
                    'L:/orcawa/spatial/_other/'
                    'projection_files/esri_national_albers.prj'))),
            ('species_matrix_file',
                normpath('L:/model_dir/sppsz.csv')),
            ('environmental_matrix_file',
                normpath('L:/model_dir/sppsz_spatudb.csv')),
            ('stand_attribute_file',
                normpath('L:/model_dir/stand_attr.csv')),
            ('stand_metadata_file',
                normpath('L:/model_dir/stand_attr.xml')),
            ('footprint_file',
                normpath((
                    'L:/resources/code/bin/footprint/'
                    'fp_multi_pixel.txt'))),
            ('independent_predicted_file',
                normpath('L:/model_dir/predicted_independent.csv')),
            ('independent_zonal_pixel_file',
                normpath('L:/model_dir/zonal_pixel_independent.csv')),
            ('dependent_predicted_file',
                normpath('L:/model_dir/predicted_dependent.csv')),
            ('dependent_zonal_pixel_file',
                normpath('L:/model_dir/zonal_pixel_dependent.csv')),
            ('dependent_nn_index_file',
                normpath('L:/model_dir/nn_index_dependent.csv')),

            # Model parameters
            ('model_project', 'CMONSTER'),
            ('model_region', 224),
            ('model_year', 2005),
            ('model_type', 'sppsz'),
            ('image_source', 'LARSE'),
            ('image_version', 1.0),
            ('plot_image_crosswalk', 'DEFAULT'),
            ('image_years', []),
            ('plot_years', []),
            ('coincident_plots', 1),
            ('lump_table', 1),
            ('buffer', 1),
            ('plot_types', ['periodic', 'annual', 'fia_special']),
            ('exclusion_codes', [
                'clouds', 'coords', 'snow', 'inexact_coords',
                'questionable_coords', 'duplicate_coords', 'mismatch',
                'disturb', 'strucedge', 'validation', 'pvt_glc_mismatch',
                'unusual_plcom_spp', 'for_minority', 'eslf_only']),

            # Ordination parameters
            ('ordination_program', 'vegan'),
            ('distance_metric', 'CCA'),
            ('species_transform', 'SQRT'),
            ('species_downweighting', 0),
            ('variable_filter', 'RAW'),

            # Imputation parameters
            ('number_axes', 8),
            ('use_axis_weighting', 1),
            ('k', 1),
            ('max_neighbors', 100),

            # Domain parameters
            ('domain', 'window'),
            ('window_cell_size', 30.0),
            ('envelope', [-2040000.0, 2522000.0, -2037000.0, 2525000.0]),
            ('axes_file', 'axis'),
            ('neighbor_file', 'nnplt'),
            ('distance_file', 'nndst'),
            ('write_axes', 4),
            ('write_neighbors', 1),
            ('write_distances', 1),

            # Accuracy assessment
            ('accuracy_assessment_folder',
                normpath('L:/model_dir/aa')),
            ('accuracy_assessment_report',
                normpath('L:/model_dir/aa/accuracy_report.pdf')),
            ('report_metadata_file',
                normpath('L:/model_dir/aa/report_metadata.xml')),
            ('accuracy_diagnostics', [
                'local_accuracy', 'species_accuracy', 'vegclass_accuracy',
                'regional_accuracy', 'riemann_accuracy',
                'validation_accuracy']),
            ('local_accuracy_file',
                normpath('L:/model_dir/aa/local_accuracy.csv')),
            ('species_accuracy_file',
                normpath('L:/model_dir/aa/species_accuracy.csv')),
            ('vegclass_file',
                normpath('L:/model_dir/aa/vegclass.csv')),
            ('vegclass_kappa_file',
                normpath('L:/model_dir/aa/vegclass_kappa.csv')),
            ('vegclass_errmatrix_file',
                normpath('L:/model_dir/aa/vegclass_errmat.csv')),
            ('area_estimate_file',
                normpath('L:/model_dir/aa/area_estimates.csv')),
            ('regional_accuracy_file',
                normpath('L:/model_dir/aa/regional_accuracy.csv')),
            ('riemann_output_folder',
                normpath('L:/model_dir/aa/riemann_accuracy')),
            ('hex_attribute_file',
                normpath((
                    'L:/model_dir/aa/riemann_accuracy/'
                    'hex_attributes.csv'))),
            ('hex_statistics_file',
                normpath((
                    'L:/model_dir/aa/riemann_accuracy/'
                    'riemann_accuracy.csv'))),
            ('riemann_hex_resolutions', [
                ('HEX_10_ID', 10.0, 8660.0, 2),
                ('HEX_30_ID', 30.0, 78100.0, 4),
                ('HEX_50_ID', 50.0, 216500.0, 8),
            ]),
            ('riemann_k_values', [1, 2, 5, 10, 20]),
            ('validation_output_folder',
                normpath('L:/model_dir/aa/validation_accuracy')),
            ('validation_attribute_file',
                normpath('L:/model_dir/aa/validation_accuracy/observed.csv')),
            ('include_in_report', [
                'local_accuracy', 'species_accuracy', 'vegclass_accuracy',
                'regional_accuracy', 'riemann_accuracy']),

            # Outlier assessment
            ('outlier_assessment_folder',
                normpath('L:/model_dir/outliers')),
            ('outlier_diagnostics',
                ['nn_index_outlier', 'vegclass_outlier', 'vegclass_variety']),
            ('nn_index_outlier_file',
                normpath('L:/model_dir/outliers/nn_index_outliers.csv')),
            ('index_threshold', 10.0),
            ('vegclass_outlier_file',
                normpath('L:/model_dir/outliers/vegclass_outliers.csv')),
            ('vegclass_variety_file',
                normpath('L:/model_dir/outliers/vegclass_variety.csv')),
        )
        for (v1, v2) in props:
            self.assertEqual(getattr(obj, v1), v2)

    def test_set_properties(self):

        obj = self.sppsz_prototype
        props = (

            # Parameter set
            ('parameter_set', 'FULL', 'FULL'),

            # File locations
            ('model_directory', 'test_dir', 'test_dir'),
            ('boundary_raster', 'test_raster', 'test_raster'),
            ('species_matrix_file', 'test_sppsz.csv',
                normpath('test_dir/test_sppsz.csv')),
            ('environmental_matrix_file', 'test_sppsz_spatudb.csv',
                normpath('test_dir/test_sppsz_spatudb.csv')),

            # Model parameters
            ('model_region', 1, 1),
            ('model_year', 2000, 2000),
            ('plot_image_crosswalk', [(1990, 1990), (1991, 1991)],
                [(1990, 1990), (1991, 1991)]),

            # Domain parameters
            ('envelope', [0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]),

            # Accuracy assessment
            ('accuracy_assessment_report', 'test_report.pdf',
                normpath('test_dir/aa/test_report.pdf')),
        )
        for (v1, v2, v3) in props:
            try:
                assert(hasattr(obj, v1))
            except AssertionError:
                err_msg = v1 + ' has no setter defined'
                raise AttributeError(err_msg)

            try:
                setattr(obj, v1, v2)
            except AttributeError:
                err_msg = 'Cannot set attribute for ' + v1
                raise AttributeError(err_msg)

            self.assertEqual(getattr(obj, v1), v3)

    def test_get_ordination_file(self):
        obj = self.sppsz_prototype
        self.assertEqual(
            obj.get_ordination_file(), normpath('L:/model_dir/vegan_cca.txt'))

    def test_ordination_variables(self):
        obj = self.sppsz_prototype

        # Setter takes different parameters than what the getter returns
        # Neither are properties
        ord_variables = [
            ('spam', 'path_to_spam', 2000),
            ('eggs', 'path_to_eggs', 0),
        ]
        obj.set_ordination_variables(ord_variables)

        ord_variables = [
            ('spam', 'path_to_spam'),
            ('eggs', 'path_to_eggs'),
        ]
        self.assertEqual(
            obj.get_ordination_variables(2000), ord_variables)
        self.assertEqual(
            obj.get_ordination_variables(1999), ord_variables[1:])

    def test_optional_fields(self):
        obj = self.sppsz_prototype

        # Remove all optional parameters and test values
        elems = [
            (obj.root.file_locations, 'mask_raster'),
            (obj.root.model_parameters, 'model_project'),
            (obj.root.accuracy_assessment, 'accuracy_assessment_report'),
            (obj.root.accuracy_assessment, 'report_metadata_file'),
            (obj.root.accuracy_assessment, 'diagnostics'),
            (obj.root.outlier_assessment, 'diagnostics'),
        ]

        for (path, tag) in elems:
            a = path.find(tag)
            path.remove(a)

        # Ensure the object still validates
        xpp.utilities.validate_xml(obj.tree, obj.xml_schema_file)

        # Ensure the properties of the missing elements return the correct
        # values
        props = (
            ('mask_raster', ''),
            ('model_project', ''),
            ('accuracy_assessment_report', ''),
            ('report_metadata_file', ''),
            ('accuracy_diagnostics', []),
            ('outlier_diagnostics', []),
        )
        for (v1, v2) in props:
            self.assertEqual(getattr(obj, v1), v2)


class TrecovPrototypeParserTest(unittest.TestCase):

    def setUp(self):
        trecov_proto_name = 'L:/resources/code/xml/trecov_parameters.xml'
        self.trecov_prototype = xpp.XMLParameterParser(trecov_proto_name)

    def test_normal_init(self):
        obj = self.trecov_prototype.create_model_xml(
            'D:/test', 118, 2012)
        obj.write_tree('./data/trecov_full.xml')

    def test_get_properties(self):
        obj = self.trecov_prototype
        props = (
            # Parameter set
            ('parameter_set', 'PROTOTYPE'),

            # File locations
            ('model_directory',
                normpath('L:/model_dir')),
            ('plot_dsn', 'rocky2lemma'),
            ('web_dsn', 'web_lemma'),
            ('coordinate_file',
                normpath('C:/_coords/fcid_coords.csv')),
            ('boundary_raster',
                normpath('L:/orcawa/nwfp/bounds/mr224')),
            ('mask_raster',
                normpath('L:/orcawa/spatial/masks/gap_nfmask30')),
            ('projection_file',
                normpath((
                    'L:/orcawa/spatial/_other/'
                    'projection_files/esri_national_albers.prj'))),
            ('species_matrix_file',
                normpath('L:/model_dir/trecov.csv')),
            ('environmental_matrix_file',
                normpath('L:/model_dir/trecov_spatudb.csv')),
            ('stand_attribute_file',
                normpath('L:/model_dir/stand_attr.csv')),
            ('stand_metadata_file',
                normpath('L:/model_dir/stand_attr.xml')),
            ('footprint_file',
                normpath((
                    'L:/resources/code/bin/footprint/'
                    'fp_single_pixel.txt'))),
            ('independent_predicted_file',
                normpath('L:/model_dir/predicted_independent.csv')),
            ('independent_zonal_pixel_file',
                normpath('L:/model_dir/zonal_pixel_independent.csv')),
            ('dependent_predicted_file',
                normpath('L:/model_dir/predicted_dependent.csv')),
            ('dependent_zonal_pixel_file',
                normpath('L:/model_dir/zonal_pixel_dependent.csv')),
            ('dependent_nn_index_file',
                normpath('L:/model_dir/nn_index_dependent.csv')),

            # Model parameters
            ('model_project', 'DISS'),
            ('model_region', 118),
            ('model_year', 2012),
            ('model_type', 'trecov'),
            ('image_source', ''),
            ('image_version', 0.0),
            ('plot_image_crosswalk', []),
            ('image_years', []),
            ('plot_years', []),
            ('coincident_plots', 0),
            ('lump_table', 1),
            ('buffer', 0),
            ('plot_types', [
                'periodic', 'annual', 'ecoplot', 'firemon', 'fia_special']),
            ('exclusion_codes', [
                'coords', 'inexact_coords', 'questionable_coords',
                'duplicate_coords', 'pvt_glc_mismatch', 'unusual_plcom_spp',
                'for_minority', 'eslf_only']),

            # Ordination parameters
            ('ordination_program', 'numpy'),
            ('distance_metric', 'CCA'),
            ('species_transform', 'SQRT'),
            ('species_downweighting', 0),
            ('variable_filter', 'RAW'),

            # Imputation parameters
            ('number_axes', 8),
            ('use_axis_weighting', 1),
            ('k', 1),
            ('max_neighbors', 100),

            # Domain parameters
            ('domain', 'window'),
            ('window_cell_size', 30.0),
            ('envelope', [-2040000.0, 2522000.0, -2037000.0, 2525000.0]),
            ('axes_file', 'axis'),
            ('neighbor_file', 'nnplt'),
            ('distance_file', 'nndst'),
            ('write_axes', 4),
            ('write_neighbors', 1),
            ('write_distances', 1),

            # Accuracy assessment
            ('accuracy_assessment_folder',
                normpath('L:/model_dir/aa')),
            ('accuracy_assessment_report',
                normpath('L:/model_dir/aa/accuracy_report.pdf')),
            ('report_metadata_file',
                normpath('L:/model_dir/aa/report_metadata.xml')),
            ('accuracy_diagnostics', [
                'local_accuracy', 'species_accuracy',
                'regional_accuracy', 'riemann_accuracy']),
            ('local_accuracy_file',
                normpath('L:/model_dir/aa/local_accuracy.csv')),
            ('species_accuracy_file',
                normpath('L:/model_dir/aa/species_accuracy.csv')),
            ('area_estimate_file',
                normpath('L:/model_dir/aa/area_estimates.csv')),
            ('regional_accuracy_file',
                normpath('L:/model_dir/aa/regional_accuracy.csv')),
            ('riemann_output_folder',
                normpath('L:/model_dir/aa/riemann_accuracy')),
            ('hex_attribute_file',
                normpath(
                    'L:/model_dir/aa/riemann_accuracy/hex_attributes.csv')),
            ('hex_statistics_file',
                normpath(
                    'L:/model_dir/aa/riemann_accuracy/riemann_accuracy.csv')),
            ('riemann_hex_resolutions', [
                ('HEX_10_ID', 10.0, 8660.0, 2),
                ('HEX_30_ID', 30.0, 78100.0, 4),
                ('HEX_50_ID', 50.0, 216500.0, 8),
            ]),
            ('riemann_k_values', [1, 2, 5, 10, 20]),
            ('include_in_report', [
                'local_accuracy', 'species_accuracy', 'regional_accuracy',
                'riemann_accuracy']),
        )
        for (v1, v2) in props:
            self.assertEqual(getattr(obj, v1), v2)

    def test_set_properties(self):

        obj = self.trecov_prototype

        # Props are a 3-tuple of tag, new value to set, expected value to get
        props = (

            # Parameter set
            ('parameter_set', 'FULL', 'FULL'),

            # File locations
            ('model_directory', 'test_dir', 'test_dir'),
            ('boundary_raster', 'test_raster', 'test_raster'),
            ('species_matrix_file', 'test_trecov.csv',
                normpath('test_dir/test_trecov.csv')),
            ('environmental_matrix_file', 'test_trecov_spatudb.csv',
                normpath('test_dir/test_trecov_spatudb.csv')),

            # Model parameters
            ('model_region', 1, 1),
            ('model_year', 2000, 2000),
            ('plot_years', [1990, 1991], [1990, 1991]),

            # Domain parameters
            ('envelope', [0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]),

            # Accuracy assessment
            ('accuracy_assessment_report', 'test_report.pdf',
                normpath('test_dir/aa/test_report.pdf')),
        )
        for (v1, v2, v3) in props:
            try:
                assert(hasattr(obj, v1))
            except AssertionError:
                err_msg = v1 + ' has no setter defined'
                raise AttributeError(err_msg)

            try:
                setattr(obj, v1, v2)
            except AttributeError:
                err_msg = 'Cannot set attribute for ' + v1
                raise AttributeError(err_msg)

            self.assertEqual(getattr(obj, v1), v3)

    def test_get_ordination_file(self):
        obj = self.trecov_prototype
        self.assertEqual(
            obj.get_ordination_file(), normpath('L:/model_dir/numpy_cca.txt'))

    def test_ordination_variables(self):
        obj = self.trecov_prototype

        # Setter takes different parameters than what the getter returns
        # Neither are properties
        ord_variables = [
            ('spam', 'path_to_spam', 2000),
            ('eggs', 'path_to_eggs', 0),
        ]
        obj.set_ordination_variables(ord_variables)

        ord_variables = [
            ('spam', 'path_to_spam'),
            ('eggs', 'path_to_eggs'),
        ]
        self.assertEqual(
            obj.get_ordination_variables(2000), ord_variables)
        self.assertEqual(
            obj.get_ordination_variables(1999), ord_variables[1:])

    def test_optional_fields(self):
        obj = self.trecov_prototype

        # Remove all optional parameters and test values
        elems = [
            (obj.root.file_locations, 'mask_raster'),
            (obj.root.model_parameters, 'model_project'),
            (obj.root.accuracy_assessment, 'accuracy_assessment_report'),
            (obj.root.accuracy_assessment, 'report_metadata_file'),
            (obj.root.accuracy_assessment, 'diagnostics'),
        ]

        for (path, tag) in elems:
            a = path.find(tag)
            path.remove(a)

        # Ensure the object still validates
        xpp.utilities.validate_xml(obj.tree, obj.xml_schema_file)

        # Ensure the properties of the missing elements return the correct
        # values
        props = (
            ('mask_raster', ''),
            ('model_project', ''),
            ('accuracy_assessment_report', ''),
            ('report_metadata_file', ''),
            ('accuracy_diagnostics', []),
        )
        for (v1, v2) in props:
            self.assertEqual(getattr(obj, v1), v2)


class SppszFullParserTest(unittest.TestCase):

    def setUp(self):
        sppsz_full_name = './data/sppsz_full.xml'
        self.sppsz_full = xpp.XMLParameterParser(sppsz_full_name)

    def test_get_properties(self):
        obj = self.sppsz_full
        props = (
            # Parameter set
            ('parameter_set', 'FULL'),

            # File locations
            ('model_directory',
                normpath('D:/test')),
            ('plot_dsn', 'rocky2lemma'),
            ('web_dsn', 'web_lemma'),
            ('coordinate_file',
                normpath('C:/_coords/fcid_coords.csv')),
            ('boundary_raster',
                normpath('L:/orcawa/cmonster/bounds/mr224')),
            ('mask_raster',
                normpath('L:/orcawa/spatial/masks/gap_nfmask30')),
            ('projection_file',
                normpath((
                    'L:/orcawa/spatial/_other/projection_files/'
                    'esri_national_albers.prj'))),
            ('species_matrix_file',
                normpath('D:/test/sppsz.csv')),
            ('environmental_matrix_file',
                normpath('D:/test/sppsz_spatudb.csv')),
            ('stand_attribute_file',
                normpath('D:/test/stand_attr.csv')),
            ('stand_metadata_file',
                normpath('D:/test/stand_attr.xml')),
            ('footprint_file',
                normpath(
                    'L:/resources/code/bin/footprint/fp_multi_pixel.txt')),
            ('independent_predicted_file',
                normpath('D:/test/predicted_independent.csv')),
            ('independent_zonal_pixel_file',
                normpath('D:/test/zonal_pixel_independent.csv')),
            ('dependent_predicted_file',
                normpath('D:/test/predicted_dependent.csv')),
            ('dependent_zonal_pixel_file',
                normpath('D:/test/zonal_pixel_dependent.csv')),
            ('dependent_nn_index_file',
                normpath('D:/test/nn_index_dependent.csv')),

            # Model parameters
            ('model_project', 'CMONSTER'),
            ('model_region', 224),
            ('model_year', 1996),
            ('model_type', 'sppsz'),
            ('image_source', 'LARSE'),
            ('image_version', 1.0),
            ('plot_image_crosswalk', [
                (1980, 1984), (1991, 1991), (1992, 1992), (1993, 1993),
                (1994, 1994), (1995, 1995), (1996, 1996), (1997, 1997),
                (1998, 1998), (1999, 1999), (2000, 2000), (2001, 2001),
                (2002, 2002), (2003, 2003), (2004, 2004), (2005, 2005),
                (2006, 2006), (2007, 2007), (2008, 2008), (2009, 2009),
                (2010, 2010),
            ]),
            ('image_years', [
                1984, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                2010,
            ]),
            ('plot_years', [
                1980, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999,
                2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009,
                2010,
            ]),
            ('coincident_plots', 1),
            ('lump_table', 1),
            ('buffer', 1),
            ('plot_types', ['periodic', 'annual', 'fia_special']),
            ('exclusion_codes', [
                'clouds', 'coords', 'snow', 'inexact_coords',
                'questionable_coords', 'duplicate_coords', 'mismatch',
                'disturb', 'strucedge', 'validation', 'pvt_glc_mismatch',
                'unusual_plcom_spp', 'for_minority', 'eslf_only']),

            # Ordination parameters
            ('ordination_program', 'vegan'),
            ('distance_metric', 'CCA'),
            ('species_transform', 'SQRT'),
            ('species_downweighting', 0),
            ('variable_filter', 'RAW'),

            # Imputation parameters
            ('number_axes', 8),
            ('use_axis_weighting', 1),
            ('k', 1),
            ('max_neighbors', 100),

            # Domain parameters
            ('domain', 'window'),
            ('window_cell_size', 30.0),
            ('envelope', [-2184015.0, 2261985.0, -1938015.0, 2822985.0]),
            ('axes_file', 'axis'),
            ('neighbor_file', 'nnplt'),
            ('distance_file', 'nndst'),
            ('write_axes', 4),
            ('write_neighbors', 1),
            ('write_distances', 1),

            # Accuracy assessment
            ('accuracy_assessment_folder',
                normpath('D:/test/aa')),
            ('accuracy_assessment_report',
                normpath('D:/test/aa/mr224_sppsz_1996_aa.pdf')),
            ('report_metadata_file',
                normpath('D:/test/aa/report_metadata.xml')),
            ('accuracy_diagnostics', [
                'local_accuracy', 'species_accuracy', 'vegclass_accuracy',
                'regional_accuracy', 'riemann_accuracy',
                'validation_accuracy']),
            ('local_accuracy_file',
                normpath('D:/test/aa/local_accuracy.csv')),
            ('species_accuracy_file',
                normpath('D:/test/aa/species_accuracy.csv')),
            ('vegclass_file',
                normpath('D:/test/aa/vegclass.csv')),
            ('vegclass_kappa_file',
                normpath('D:/test/aa/vegclass_kappa.csv')),
            ('vegclass_errmatrix_file',
                normpath('D:/test/aa/vegclass_errmat.csv')),
            ('area_estimate_file',
                normpath('D:/test/aa/area_estimates.csv')),
            ('regional_accuracy_file',
                normpath('D:/test/aa/regional_accuracy.csv')),
            ('riemann_output_folder',
                normpath('D:/test/aa/riemann_accuracy')),
            ('hex_attribute_file',
                normpath('D:/test/aa/riemann_accuracy/hex_attributes.csv')),
            ('hex_statistics_file',
                normpath('D:/test/aa/riemann_accuracy/riemann_accuracy.csv')),
            ('riemann_hex_resolutions', [
                ('HEX_10_ID', 10.0, 8660.0, 2),
                ('HEX_30_ID', 30.0, 78100.0, 4),
                ('HEX_50_ID', 50.0, 216500.0, 8),
            ]),
            ('riemann_k_values', [1, 2, 5, 10, 20]),
            ('validation_output_folder',
                normpath('D:/test/aa/validation_accuracy')),
            ('validation_attribute_file',
                normpath('D:/test/aa/validation_accuracy/observed.csv')),
            ('include_in_report', [
                'local_accuracy', 'species_accuracy', 'vegclass_accuracy',
                'regional_accuracy', 'riemann_accuracy']),

            # Outlier assessment
            ('outlier_assessment_folder',
                normpath('D:/test/outliers')),
            ('outlier_diagnostics',
                ['nn_index_outlier', 'vegclass_outlier', 'vegclass_variety']),
            ('nn_index_outlier_file',
                normpath('D:/test/outliers/nn_index_outliers.csv')),
            ('index_threshold', 10.0),
            ('vegclass_outlier_file',
                normpath('D:/test/outliers/vegclass_outliers.csv')),
            ('vegclass_variety_file',
                normpath('D:/test/outliers/vegclass_variety.csv')),
        )
        for (v1, v2) in props:
            self.assertEqual(getattr(obj, v1), v2)

    def test_get_ordination_file(self):
        obj = self.sppsz_full
        self.assertEqual(
            obj.get_ordination_file(), normpath('D:/test/vegan_cca.txt'))

    def test_ordination_variables(self):
        obj = self.sppsz_full
        ord_vars = [
            ('LAT', 'L:/orcawa/spatial/misc/lat30'),
            ('LON', 'L:/orcawa/spatial/misc/lon30'),
            ('ANNPRE', 'L:/orcawa/spatial/prism/annpre30'),
            ('ANNTMP', 'L:/orcawa/spatial/prism/anntmp30'),
            ('AUGMAXT', 'L:/orcawa/spatial/prism/augmaxt30'),
            ('COASTPROX', 'L:/orcawa/spatial/prism/coastprox30'),
            ('CONTPRE', 'L:/orcawa/spatial/prism/contpre30'),
            ('DECMINT', 'L:/orcawa/spatial/prism/decmint30'),
            ('SMRTP', 'L:/orcawa/spatial/prism/smrtp30'),
            ('ASPTR', 'L:/orcawa/spatial/topography/asptr30'),
            ('DEM', 'L:/orcawa/spatial/topography/dem30'),
            ('PRR', 'L:/orcawa/spatial/topography/prr30'),
            ('SLPPCT', 'L:/orcawa/spatial/topography/slppct30'),
            ('TPI450', 'L:/orcawa/spatial/topography/tpi45030'),
            ('TC1', 'M:/spatial/tm/larse/v_1_0/tc/tc19630'),
            ('TC2', 'M:/spatial/tm/larse/v_1_0/tc/tc29630'),
            ('TC3', 'M:/spatial/tm/larse/v_1_0/tc/tc39630'),
        ]
        self.assertEqual(
            obj.get_ordination_variables(model_year=1996), ord_vars)
        self.assertEqual(
            obj.get_ordination_variables(), ord_vars)


class TrecovFullParserTest(unittest.TestCase):

    def setUp(self):
        trecov_full_name = './data/trecov_full.xml'
        self.trecov_full = xpp.XMLParameterParser(trecov_full_name)

    def test_get_properties(self):
        obj = self.trecov_full
        props = (
            # Parameter set
            ('parameter_set', 'FULL'),

            # File locations
            ('model_directory',
                normpath('D:/test')),
            ('plot_dsn', 'rocky2lemma'),
            ('web_dsn', 'web_lemma'),
            ('coordinate_file',
                normpath('C:/_coords/fcid_coords.csv')),
            ('boundary_raster',
                normpath('L:/orcawa/diss/bounds/mr118')),
            ('mask_raster',
                normpath('L:/orcawa/spatial/masks/gap_nfmask30')),
            ('projection_file',
                normpath((
                    'L:/orcawa/spatial/_other/projection_files/'
                    'esri_national_albers.prj'))),
            ('species_matrix_file',
                normpath('D:/test/trecov.csv')),
            ('environmental_matrix_file',
                normpath('D:/test/trecov_spatudb.csv')),
            ('stand_attribute_file',
                normpath('D:/test/stand_attr.csv')),
            ('stand_metadata_file',
                normpath('D:/test/stand_attr.xml')),
            ('footprint_file',
                normpath(
                    'L:/resources/code/bin/footprint/fp_single_pixel.txt')),
            ('independent_predicted_file',
                normpath('D:/test/predicted_independent.csv')),
            ('independent_zonal_pixel_file',
                normpath('D:/test/zonal_pixel_independent.csv')),
            ('dependent_predicted_file',
                normpath('D:/test/predicted_dependent.csv')),
            ('dependent_zonal_pixel_file',
                normpath('D:/test/zonal_pixel_dependent.csv')),
            ('dependent_nn_index_file',
                normpath('D:/test/nn_index_dependent.csv')),

            # Model parameters
            ('model_project', 'DISS'),
            ('model_region', 118),
            ('model_year', 2012),
            ('model_type', 'trecov'),
            ('image_source', ''),
            ('image_version', 0.0),
            ('plot_image_crosswalk', []),
            ('image_years', []),
            ('plot_years', [
                1961, 1962, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972,
                1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982,
                1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992,
                1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002,
                2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010,
            ]),
            ('coincident_plots', 0),
            ('lump_table', 1),
            ('buffer', 0),
            ('plot_types', [
                'periodic', 'annual', 'ecoplot', 'firemon', 'fia_special']),
            ('exclusion_codes', [
                'coords', 'inexact_coords', 'questionable_coords',
                'duplicate_coords', 'pvt_glc_mismatch', 'unusual_plcom_spp',
                'for_minority', 'eslf_only']),

            # Ordination parameters
            ('ordination_program', 'numpy'),
            ('distance_metric', 'CCA'),
            ('species_transform', 'SQRT'),
            ('species_downweighting', 0),
            ('variable_filter', 'RAW'),

            # Imputation parameters
            ('number_axes', 8),
            ('use_axis_weighting', 1),
            ('k', 1),

            # Domain parameters
            ('domain', 'window'),
            ('window_cell_size', 30.0),
            ('envelope', [-2307015.0, 2348985.0, -1932015.0, 2912985.0]),
            ('axes_file', 'axis'),
            ('neighbor_file', 'nnplt'),
            ('distance_file', 'nndst'),
            ('write_axes', 4),
            ('write_neighbors', 1),
            ('write_distances', 1),

            # Accuracy assessment
            ('accuracy_assessment_folder',
                normpath('D:/test/aa')),
            ('accuracy_assessment_report',
                normpath('D:/test/aa/mr118_trecov_2012_aa.pdf')),
            ('report_metadata_file',
                normpath('D:/test/aa/report_metadata.xml')),
            ('accuracy_diagnostics', [
                'local_accuracy', 'species_accuracy', 'regional_accuracy',
                'riemann_accuracy']),
            ('local_accuracy_file',
                normpath('D:/test/aa/local_accuracy.csv')),
            ('species_accuracy_file',
                normpath('D:/test/aa/species_accuracy.csv')),
            ('area_estimate_file',
                normpath('D:/test/aa/area_estimates.csv')),
            ('regional_accuracy_file',
                normpath('D:/test/aa/regional_accuracy.csv')),
            ('riemann_output_folder',
                normpath('D:/test/aa/riemann_accuracy')),
            ('hex_attribute_file',
                normpath(
                    'D:/test/aa/riemann_accuracy/hex_attributes.csv')),
            ('hex_statistics_file',
                normpath(
                    'D:/test/aa/riemann_accuracy/riemann_accuracy.csv')),
            ('riemann_hex_resolutions', [
                ('HEX_10_ID', 10.0, 8660.0, 2),
                ('HEX_30_ID', 30.0, 78100.0, 4),
                ('HEX_50_ID', 50.0, 216500.0, 8),
            ]),
            ('riemann_k_values', [1, 2, 5, 10, 20]),
            ('include_in_report', [
                'local_accuracy', 'species_accuracy', 'regional_accuracy',
                'riemann_accuracy']),
        )
        for (v1, v2) in props:
            self.assertEqual(getattr(obj, v1), v2)

    def test_get_ordination_file(self):
        obj = self.trecov_full
        self.assertEqual(
            obj.get_ordination_file(), normpath('D:/test/numpy_cca.txt'))

    def test_ordination_variables(self):
        obj = self.trecov_full
        ord_vars = [
            ('ASHDEPTH', 'L:/orcawa/spatial/misc/ashdepth30'),
            ('LAT', 'L:/orcawa/spatial/misc/lat30'),
            ('LON', 'L:/orcawa/spatial/misc/lon30'),
            ('PYROFLOW', 'L:/orcawa/spatial/misc/pyroflow30'),
            ('ANNPRE', 'L:/orcawa/spatial/prism/annpre30'),
            ('ANNTMP', 'L:/orcawa/spatial/prism/anntmp30'),
            ('AUGMAXT', 'L:/orcawa/spatial/prism/augmaxt30'),
            ('COASTPROX', 'L:/orcawa/spatial/prism/coastprox30'),
            ('CONTPRE', 'L:/orcawa/spatial/prism/contpre30'),
            ('CVPRE', 'L:/orcawa/spatial/prism/cvpre30'),
            ('DECMINT', 'L:/orcawa/spatial/prism/decmint30'),
            ('DIFTMP', 'L:/orcawa/spatial/prism/diftmp30'),
            ('SMRPRE', 'L:/orcawa/spatial/prism/smrpre30'),
            ('SMRTMP', 'L:/orcawa/spatial/prism/smrtmp30'),
            ('SMRTP', 'L:/orcawa/spatial/prism/smrtp30'),
            ('ALLUVIAL', 'L:/orcawa/spatial/soil/alluvial30'),
            ('SAND', 'L:/orcawa/spatial/soil/sand30'),
            ('SILICIC', 'L:/orcawa/spatial/soil/silicic30'),
            ('ULTRAMAFIC', 'L:/orcawa/spatial/soil/ultramafic30'),
            ('ASPTR', 'L:/orcawa/spatial/topography/asptr30'),
            ('DEM', 'L:/orcawa/spatial/topography/dem30'),
            ('MLI', 'L:/orcawa/spatial/topography/mli30'),
            ('PRR', 'L:/orcawa/spatial/topography/prr30'),
            ('SLPPCT', 'L:/orcawa/spatial/topography/slppct30'),
            ('TPI150', 'L:/orcawa/spatial/topography/tpi15030'),
            ('TPI300', 'L:/orcawa/spatial/topography/tpi30030'),
            ('TPI450', 'L:/orcawa/spatial/topography/tpi45030'),
        ]
        self.assertEqual(
            obj.get_ordination_variables(model_year=1996), ord_vars)
        self.assertEqual(
            obj.get_ordination_variables(), ord_vars)

if __name__ == '__main__':
    unittest.main()
