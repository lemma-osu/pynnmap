import numpy
from pynnmap.database import database as db
from pynnmap.misc import utilities


class PlotDatabase(db.Database):

    def __init__(self, model_type, model_region, buffer,
                 model_year, image_source,
                 image_version, dsn='rocky2lemma'):
        """
        Initializes a ModelDatabase instance and sets up the metadata for
        the tables that it defines

        Parameters
        ----------
        model_type : str
            Model type.  Should be one of ('sppsz', 'trecov', 'wdycov', etc.)
        model_region : str
            LEMMA modeling region
        buffer : bool
            indicates whether model covers MR buffer
        model_year : int
            modeling year
        image_source : str
            source of imagery (i.e. LARSE, RSAC)
        image_version : float
            version of imagery
        dsn : str
            Name of DSN for model database

        Returns
        -------
        None
        """

        self.model_type = model_type
        self.model_region = model_region
        self.buffer = buffer
        self.model_year = model_year
        self.image_source = image_source
        self.image_version = image_version
        self.dsn = dsn

    def get_area_estimates(self, comparison_year):
        """
        Get estimates of acres separated into forest, nonforest
        and nonsampled areas

        Parameters
        ----------
        comparison_year : int
            Year for which plot assessment years will be compared.  If more
            than one plot exists at a location, the plot that is temporally
            closer to this parameter will be chosen

        Returns
        -------
        observed_data: numpy.recarray
            structure and species data and # forest hectares
        nf_hectares: float
            # nonforest hectares
        ns_hectares: float
            # nonsampled hectares
        """

        sql_base = """
           EXEC lemma.GET_AREA_EXPANSIONS
            @model_region = %d,
            @model_type = '%s',
            @glc_group = '%s',
            @model_year = %d
        """

        # Forested area
        sql = sql_base % (self.model_region, self.model_type, 'F',
            comparison_year)
        (records, descr) = self.get_data(sql)
        observed_data = utilities.pyodbc2rec(records, descr)

        # Nonforest area
        sql = sql_base % (self.model_region, self.model_type, 'NF',
            comparison_year)
        (records, descr) = self.get_data(sql)
        nf_hectares = records[0][0]

        # Non-sampled area
        sql = sql_base % (self.model_region, self.model_type, 'NS',
            comparison_year)
        (records, descr) = self.get_data(sql)
        ns_hectares = records[0][0]

        return observed_data, nf_hectares, ns_hectares

    def get_attribute_data(self, id_str):
        """
        Get an attribute table which includes both continuous stand
        variables and species by abundance measure (basal area, percent
        cover, etc.) for use in accuracy assessment for NN models.

        Parameters
        ----------
        id_str: str
            Comma-delimited list of IDs to include

        Returns
        -------
        attibute_data: numpy.recarray
            Structure and species attribute data summarized at
                the requested summary level
        """

        sql = """
            EXEC lemma.GET_MODEL_ATTRIBUTES
            @fcids = '%s',
            @model_region = %d,
            @model_type = '%s'
        """

        sql = sql % (
            id_str, self.model_region, self.model_type)
        (records, descr) = self.get_data(sql)
        stand_attribute_data = utilities.pyodbc2rec(records, descr)
        return stand_attribute_data

    def get_environmental_matrix(self, id_str, plot_years, image_years,
                                 spatial_vars):
        """
        Get a matrix of plots by environmental variables for use as
        input to NN models

        Parameters
        ----------
        id_str: str
            comma-delimited list of IDs to include
        plot_years: str
            comma-delimited list of plot assessment years
        image_years: str
            comma-delimited list of image years
        spatial_vars: str
            comma-delimited list of spatial variables to return

        Returns
        -------
        env_matrix: numpy.recarray
            **ID: unique plot identifier 
            ordination variables (names correspond to spatial_vars list)
        """

        sql = """
            EXEC lemma.GET_ENVIRONMENTAL_MATRIX
              @fcids = '%s',
              @model_region = %d,
              @model_type = '%s',
              @plot_years = '%s',
              @image_years = '%s',
              @image_source = '%s',
              @image_version = %f,
              @spatial_vars = '%s'
        """
        sql = sql % (id_str, self.model_region,
                     self.model_type, plot_years, image_years,
                     self.image_source, self.image_version, spatial_vars)
        records, descr = self.get_data(sql)
        env_matrix = utilities.pyodbc2rec(records, descr)
        return env_matrix

    def get_field_names(self, table_name):
        """
        Get field names for this table

        Parameters
        ----------
        table_name: str
            name of table for which to get field names

        Returns
        -------
        field_names: numpy.array
        """

        sql = """
            EXEC lemma.GET_FIELD_NAMES
                @table_name = '%s'
        """

        sql = sql % (table_name)
        records, descr = self.get_data(sql)
        field_names = [str(x[0]) for x in records]
        return numpy.array(field_names)

    def get_hex_attributes(self, comparison_year, plot_years, image_years):
        """
        Get hexagon IDs and continuous stand attributes for forested
        Annual plots to be used in Riemann accuracy diagnostics

        Parameters
        ----------
        comparison_year: int
            Year for which plot assessment years will be compared.  If more
            than one plot exists at a location, the plot that is temporally
            closer to this parameter will be chosen
        plot_years: str
            comma-delimited list of plot assessment years
        image_years: str
            comma-delimited list of image years

        Returns
        -------
        hex_attributes: numpy.recarray

        """
        sql = """
            EXEC lemma.GET_HEX_ATTRIBUTES
                @model_region = %d,
                @model_type = '%s',
                @model_year = %d,
                @plot_years = '%s',
                @image_years = '%s',
                @image_source = '%s',
                @image_version = %f
        """
        sql = sql % (self.model_region, self.model_type,
                     comparison_year, 
                     plot_years, image_years, self.image_source,
                     self.image_version)
        records, descr = self.get_data(sql)
        hex_attributes = utilities.pyodbc2rec(records, descr)
        return hex_attributes

    def get_image_years(self):
        """
        Return all image years for this model region

        Parameters
        ----------
        None

        Returns
        -------
        image_years: numpy.array
        """

        sql = """
            EXEC lemma.GET_IMAGE_YEARS
                @model_region = %d,
                @buffer = %d,
                @image_source = '%s',
                @image_version = %f
        """
        sql = sql % (self.model_region, self.buffer,
                     self.image_source, self.image_version)
        records, descr = self.get_data(sql)
        image_years = [int(x[0]) for x in records]
        return numpy.array(image_years)

    def get_metadata(self, table_names):
        """
        Get metadata for the fields and codes in the tables passed in

        Parameters
        ----------
        table_names: str
            comma-delimited list of table names to get metadata for

        Returns
        -------
        metadata_fields: numpy.recarray
            field names and descriptions
        metadata_codes: numpy.recarray
            code names and descriptions
        """

        metadata_fields = self.get_metadata_fields(table_names)
        metadata_codes = self.get_metadata_codes(table_names)
        return metadata_fields, metadata_codes

    def get_metadata_field_dictionary(self, table_names):
        metadata = {}
        metadata_table = self.get_metadata_fields(table_names)

        # Load the metadata into a dictionary
        for row in metadata_table:
            # First get the name of the field the metadata is describing
            # This will be the key to the metadata dictionary
            field_name = row.FIELD_NAME
            field_dict = {}
            for key in row.dtype.names:
                if key == 'CODED':
                    if row[key] == 0:
                        field_dict['CODES'] = ''
                    else:
                        # Build the code dictionary and insert
                        code_dict = \
                            self.get_metadata_code_dictionary(field_name)
                        field_dict['CODES'] = code_dict
                else:
                    field_dict[key] = row[key]

            # Delete the FIELD_NAME entry from the inner dictionary
            # del(field_dict['FIELD_NAME'])
            metadata[field_name] = field_dict
        return(metadata)

    def get_metadata_code_dictionary(self, field_name):
        """
        Build a dictionary of codes for a given field

        Parameters
        ----------
        table_name : str
            The database object name
        field_name : str
            The field name within the table_name

        Returns
        -------
        code_dict : dict
            A dictionary of all codes associated with this field
        """

        sql = "EXEC lemma.GET_METADATA_CODES '"
        sql += field_name + "'"
        code_dict = {}
        (records, desc) = self.get_data(sql)
        code_table = utilities.pyodbc2rec(records, desc)
        for row in code_table:
            code_value = row.CODE_VALUE
            inner_dict = {}
            for key in row.dtype.names:
                # if key <> 'CODE_VALUE':
                inner_dict[key] = row[key]
            code_dict[code_value] = inner_dict
        return code_dict

    def get_metadata_fields(self, table_names):
        """
        Get metadata for just the fields in the tables passed in

        Parameters
        ----------
        table_names: str
            comma-delimited list of table names to get metadata for

        Returns
        -------
        metadata_fields: numpy.recarray
            field names and descriptions
        """
        sql = """
            EXEC lemma.GET_METADATA_TABLE_FIELDS
                @table_names = '%s'
        """
        sql = sql % (table_names)
        (records, desc) = self.get_data(sql)
        metadata_fields = utilities.pyodbc2rec(records, desc)
        return metadata_fields

    def get_metadata_codes(self, table_names):
        """
        Get metadata for just the codes in the tables passed in

        Parameters
        ----------
        table_names: str
            comma-delimited list of table names to get metadata for

        Returns
        -------
        metadata_codes: numpy.recarray
            code names and descriptions
        """
        sql = """
            EXEC lemma.GET_METADATA_TABLE_CODES
                @table_names = '%s'
        """
        sql = sql % (table_names)
        try:
            (records, desc) = self.get_data(sql)
        except IndexError:
            # If the query returned no records, return None
            # this means there were no coded fields in the table(s)
            metadata_codes = None
        else:
            metadata_codes = utilities.pyodbc2rec(records, desc)

        return metadata_codes

    def get_model_region_window(self):
        """
        Get the bounding coordinates for this model region

        Parameters
        ----------
        None

        Returns
        -------
        mr_bounds : numpy.recarray
            X_MIN, Y_MIN, X_MAX, Y_MAX, BOUNDARY_RASTER (file location)
        """

        sql = """
            EXEC lemma.GET_MODEL_REGION_WINDOW
                @model_region = %d
        """
        sql = sql % (self.model_region)
        (records, descr) = self.get_data(sql)
        bounds = utilities.pyodbc2rec(records, descr)
        return bounds

    def get_ordination_variable_descriptions(self, ordination_vars):
        """
        """
        sql = """
            EXEC lemma.GET_ORDINATION_VARIABLE_DESCRIPTIONS
                @ordination_vars = '%s'
            """
        sql = sql % (ordination_vars)
        (records, descr) = self.get_data(sql)
        ordination_vars = utilities.pyodbc2rec(records, descr)
        return ordination_vars

    def get_ordination_variable_list(self, var_types, variable_filter):
        """
        Query the database to get a list of ordination variables and their
        file locations for this model region

        Parameters
        ----------
        var_types: str
            keyword to specify which variables to return
            Values:
                ALL = all variables that have complete coverage for this MR
                DEFAULT = subset of variables chosen for this MR
        variable_filter: str
            method of filtering ordination variables: RAW or FOCAL
        Returns
        -------
        ordination_table : numpy.recarray
            VARIABLE_NAME, VARIABLE_PATH
        """

        # Create the parameter signature and get the possible ordination
        # variables
        sql = """
            EXEC lemma.GET_ORDINATION_VARIABLE_LIST
              @model_region = %d,
              @model_type = '%s',
              @model_year = %d,
              @buffer = %d,
              @var_types = '%s',
              @image_source = '%s',
              @image_version = %f,
              @variable_filter = '%s'
        """
        sql = sql % (self.model_region, self.model_type, self.model_year,
                     self.buffer, var_types, self.image_source,
                     self.image_version, variable_filter)
        (records, descr) = self.get_data(sql)
        ordination_table = utilities.pyodbc2rec(records, descr)
        return ordination_table

    def get_plot_data_source_summary(self, ids):
        """
        Get plot counts grouped by data sources and assessment years

        Parameters
        ----------
        ids : str
            comma-delimited list of FCIDs for plots to include

        Returns
        -------
        plot_data_sources : numpy.recarray
            ID, DATA_SOURCE, ASSESSMENT_YEAR
        """

        sql = """
            EXEC lemma.GET_PLOT_DATA_SOURCE_SUMMARY
            @fcids = '%s'
        """
        sql = sql % (ids)
        (records, descr) = self.get_data(sql)
        plot_data_sources = utilities.pyodbc2rec(records, descr)
        return plot_data_sources

    def get_plot_image_pairs(self, keyword):
        """
        Query the database to get all plot assessment years and available
        imagery years and crosswalk the returned information using the
        logic associated with each keyword

        Parameters
        ----------
        keyword : str
            Keyword to determine the logic of how to match plot assessment
            years to imagery years. The only keyword implemented now is
            'DEFAULT', which matches plots to the closest year of imagery
            available.

        Returns
        -------
        plot_image_years : numpy.recarray
            PLOT_YEAR, IMAGE_YEAR
        """

        sql = """
            EXEC lemma.GET_PLOT_IMAGE_PAIRS
              @model_region = %d,
              @buffer = %d,
              @image_source = '%s',
              @image_version = %f,
              @keyword = '%s'
        """
        sql = sql % (self.model_region, self.buffer, self.image_source,
                     self.image_version, keyword)
        (records, descr) = self.get_data(sql)
        plot_image_years = utilities.pyodbc2rec(records, descr)
        return plot_image_years

    def get_plot_list(self, coincident_plots, lump_table, plot_years,
            image_years, plot_types, exclusion_codes):
        """
        Get a list of plot IDs that match the criteria of the parameters

        Parameters
        ----------
        coincident_plots: bool
            indicates whether to include multiple assessments at each location
        plot_years: str
            comma-delimited list of plot assessment years
        image_years: str
            comma-delimited list of image years
        plot_types: str
            comma-delimited list of plot types
            Values: ANNUAL, PERIODIC, ECOPLOT, JACKJO, FIA_SPECIAL
        exclusion_codes: str
            comma-delimited list of attributes for excluding plots
            Values: COORDS, INEXACT_COORDS, QUESTIONABLE_COORDS,
                DUPLICATE_COORDS, MISMATCH, DISTURB, STRUCEDGE,
                PVT_GLC_MISMATCH, UNUSUAL_PLCOM_SPP, MULTGLC,
                MULTFORCC, MULTPAG, MULTSERIES, FOR_MINORITY, ESLF_ONLY

        Returns
        -------
        id_str: str
            Comma-delimited string of integer FCIDs
        """

        # lemma.GET_PLOT_LIST is the regular proc to call
        # Use lemma.GET_PLOT_LIST_OLD_FOREST to subset to plots that
        # are within the LT stable mask and have a brightness value
        # between 1000-2000 (1000 <= TC1 <= 2000)
        sql = """
            EXEC lemma.GET_PLOT_LIST
              @model_region = %d,
              @model_year = %d,
              @model_type = '%s',
              @buffer = %d,
              @coincident_plots = %d,
              @lump_table = %d,
              @plot_types = '%s',
              @exclusion_codes = '%s',
              @plot_years = '%s',
              @image_years = '%s',
              @image_source = '%s',
              @image_version = %f
        """
        sql = sql % (self.model_region, self.model_year,
            self.model_type, self.buffer, coincident_plots, lump_table,
            plot_types, exclusion_codes, plot_years, image_years,
            self.image_source, self.image_version)
        (id_table, descr) = self.get_data(sql)
        id_str = ','.join([str(x.FCID) for x in id_table])
        return id_str

    def get_plot_years(self):
        """
        Return all plot assessment years for this model region

        Returns
        -------
        plot_years: numpy.array
        """

        sql = """
            EXEC lemma.GET_PLOT_YEARS
              @model_region = %d
        """
        sql = sql % (self.model_region)
        (records, descr) = self.get_data(sql)
        plot_years = [int(x[0]) for x in records]
        return numpy.array(plot_years)

    def get_species_names(self, id_str, lump_table):
        """
        Get common names and scientific names for species codes

        Parameters
        ----------
        id_str: str
            comma-delimited list of IDs to include

        Returns
        -------
        species_names : numpy.recarray
            SPP_SYMBOL, SCIENTIFIC_NAME, COMMON_NAME
        """

        sql = """
        EXEC lemma.GET_SPECIES_NAMES
            @fcids = '%s',
            @model_type = '%s',
            @lump_table = %d
        """

        sql = sql % (id_str, self.model_type, lump_table)
        (records, descr) = self.get_data(sql)
        species_info = utilities.pyodbc2rec(records, descr)
        return species_info

    def get_species_matrix(self, id_str, purpose, lump_table):
        """
        Get a species data in the format of a crosstab table of
        species by abundance measure (basal area, percent cover, etc.)
        for use as input to NN models or for attaching to model output
        or for using in model accuracy assessment

        Parameters
        ----------
        id_str: str
            comma-delimited list of IDs to include
        purpose: str
            ORDINATION: for use as input to model ordination
            ATTRIBUTE: for joining attributes to model output
            ACCURACY: for use in accuracy assessment
        lump_table: bool
            indicates whether to lump species using this model region's
            lump table in the model database

        Returns
        -------
        species_matrix: numpy.recarray
            **ID: unique plot identifier (FCID)
            species abundance (names differ by which species occur in
                model region)
        """

        sql = """
            EXEC lemma.GET_SPECIES_MATRIX
              @fcids = '%s',
              @model_region = %d,
              @model_type = '%s',
              @purpose = '%s',
              @lump_table = %d
        """
        sql = sql % (id_str, self.model_region,
                     self.model_type, purpose, lump_table)
        (records, descr) = self.get_data(sql)
        species_matrix = utilities.pyodbc2rec(records, descr)
        return species_matrix

    def get_species_metadata(self):
        """
        Get metadata for all of the species attributes

        Returns
        -------
            fields: numpy.recarray
                field names and definitions
            codes: numpy.recarray
                code names and definitions
        """
        sql = """
            EXEC lemma.GET_SPECIES_METADATA
              @model_region = %d,
              @model_type = '%s'
        """
        sql = sql % (self.model_region, self.model_type)
        records, descr = self.get_data(sql)
        species_metadata = utilities.pyodbc2rec(records, descr)
        return species_metadata

    def get_species_plot_counts(self, id_str):
        sql = """
            EXEC lemma.GET_SPECIES_PLOT_COUNTS
              @fcids = '%s',
              @model_region = %d,
              @model_type = '%s'
        """

        sql = sql % (id_str, 
                     self.model_region, self.model_type)

        records, descr = self.get_data(sql)
        species_plot_counts = utilities.pyodbc2rec(records, descr)
        return species_plot_counts

    def get_species_table_names(self):
        """
        Get the name of the generic species tables in MS-SQL
        For wdycov models there are two tables, one for trees
        and one for shrubs
        Other model types have just one species table

        Parameters
        ----------
        none

        Returns
        -------
        table_names: numpy.array
        """

        sql = """
            EXEC lemma.GET_SPECIES_TABLE_NAME
                @model_type = '%s'
        """
        sql = sql % (self.model_type)
        records, descr = self.get_data(sql)
        #recarray = utilities.pyodbc2rec(records, descr)
        #return recarray
        table_names = ''
        for x in records:
            table_names += str(x[0]) + ','
        return table_names
        #table_names = [str(x[0]) for x in records]
        #return numpy.array(table_names)

    def get_structure_table_name(self):
        """
        Get the name of the structure attribute table in MS-SQL

        Returns
        -------
        table_name: String
        """

        sql = """
            EXEC lemma.GET_STRUCTURE_TABLE_NAME
                @model_type = '%s'
        """
        sql = sql % (self.model_type)
        records, descr = self.get_data(sql)
        table_name = records[0][0]
        return table_name

    def get_structure_metadata(self, project_id):
        """
        Get metadata for all of the structure attributes
        
        Parameters
        ----------
        None
            
        Returns
        -------
            fields: numpy.recarray
                field names and definitions
            codes: numpy.recarray
                code names and definitions
        """

        sql = """
            EXEC lemma.GET_STRUCTURE_METADATA_FIELDS
                @model_type = '%s',
                @model_region = %d,
                @project_id = '%s'
        """
        sql = sql % (self.model_type, \
                     self.model_region, project_id)
        records, descr = self.get_data(sql)
        metadata_fields = utilities.pyodbc2rec(records, descr)

        structure_table = self.get_structure_table_name()
        metadata_codes = self.get_metadata_codes(structure_table)
        return metadata_fields, metadata_codes

    def get_validation_attributes(self):
        """
        Get structure and species attributes for validation plots

        Returns
        -------
            validation_attributes: numpy.recarray
        """

        sql = """
            EXEC lemma.GET_VALIDATION_ATTRIBUTES
                @model_region = %d,
                @model_type = '%s'
        """
        sql = sql % (self.model_region, self.model_type)
        records, descr = self.get_data(sql)
        validation_attributes = utilities.pyodbc2rec(records, descr)
        return validation_attributes

    def get_duplicate_plots_to_remove(self, id_str):
        """
        Get a list of plots to remove from the plot pool after
        ordination. We only want to keep one plot at locations with
        fixed spectal values for imagery (i.e. the location has been
        stabilized through LandTrendr procedures).

        Returns
        -------
        id_list: list of plot IDs to delete
        """
        sql = """
            EXEC lemma.DUPLICATE_PLOTS_TO_REMOVE
                @fcids = '%s',
                @image_source = '%s',
                @image_version = %f,
                @model_year = %d
         """

        sql = sql % (id_str, self.image_source, self.image_version,
            self.model_year)
        records, descr = self.get_data(sql)
        id_list = []
        for r in records:
            id_list.append(r.FCID)
        return id_list

    def load_outlier(self, fcid, rule_id, prediction_type,
                     variable_filter, observed_value,
                     predicted_value, vc_outlier_type, average_position):
        """
        Load a plot outlier into dbo.PLOT_SCREEN_OUTLIERS table

        Returns
        -------
        None (function inserts records into DB table)
        """

        sql = """
            EXEC lemma.LOAD_OUTLIER
                @model_region = %d,
                @fcid = %d,
                @rule_id = '%s',
                @prediction_type = '%s',
                @variable_filter = '%s',
                @image_version = %f,
                @observed_value = %d,
                @predicted_value = %d,
                @vc_outlier_type = '%s',
                @average_position = %d
        """

        sql = sql % (self.model_region, fcid, rule_id, prediction_type,
                     variable_filter, self.image_version, observed_value,
                     predicted_value, vc_outlier_type, average_position)
        self.update_data(sql)
