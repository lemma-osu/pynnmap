from pynnmap.parser import xml_parser


class XMLContact:
    def __init__(self, elem):
        self.root = elem

    @property
    def name(self) -> str:
        return str(self.root.name)

    @property
    def position_title(self):
        return str(self.root.position_title)

    @property
    def affiliation(self):
        return str(self.root.affiliation)

    @property
    def phone_number(self):
        return str(self.root.phone_number)

    @property
    def email_address(self):
        return str(self.root.email_address)


class XMLAssessmentYear:
    def __init__(self, elem):
        self.root = elem

    @property
    def assessment_year(self):
        return str(self.root.assessment_year)

    @property
    def plot_count(self):
        return int(self.root.plot_count)


class XMLPlotDataSource:
    def __init__(self, elem):
        self.root = elem

    @property
    def data_source(self):
        return str(self.root.data_source)

    @property
    def description(self):
        return str(self.root.description)

    @property
    def assessment_years(self):
        try:
            years_elem = self.root.assessment_years
            return [XMLAssessmentYear(x) for x in years_elem.iterchildren()]
        except AttributeError:
            return []


class XMLSpeciesName:
    def __init__(self, elem):
        self.root = elem

    @property
    def spp_symbol(self):
        return str(self.root.spp_symbol)

    @property
    def scientific_name(self):
        return str(self.root.scientific_name)

    @property
    def common_name(self):
        return str(self.root.common_name)


class XMLOrdinationVariable:
    def __init__(self, elem):
        self.root = elem

    @property
    def field_name(self):
        return str(self.root.field_name)

    @property
    def description(self):
        return str(self.root.description)

    @property
    def source(self):
        return str(self.root.source)


class XMLReportMetadataParser(xml_parser.XMLParser):
    def __init__(self, xml_file_name):
        """
        Initialize the XMLReportMetadataParser object by setting a
        reference to the XML parameter file.

        Parameters
        ----------
        xml_file_name : file
            name and location of XML report metadata parameter file

        Returns
        -------
        None
        """

        super(XMLReportMetadataParser, self).__init__(xml_file_name)

    @property
    def model_region_area(self):
        return float(self.root.overview.model_region_area)

    @property
    def forest_area(self):
        return float(self.root.overview.forest_area)

    @property
    def model_region_name(self):
        return str(self.root.overview.model_region_name)

    @property
    def model_region_overview(self):
        return str(self.root.overview.model_region_overview)

    @property
    def image_path(self):
        return str(self.root.overview.image_path)

    @property
    def contacts(self):
        try:
            ci_elem = self.root.contact_information
            return [XMLContact(x) for x in ci_elem.iterchildren()]
        except AttributeError:
            return []

    @property
    def plot_data_sources(self):
        try:
            data_sources_elem = self.root.plot_data_sources
            return [
                XMLPlotDataSource(x) for x in data_sources_elem.iterchildren()
            ]
        except AttributeError:
            return []

    @property
    def species_names(self):
        try:
            species_names_elem = self.root.species_names
            return [
                XMLSpeciesName(x) for x in species_names_elem.iterchildren()
            ]
        except AttributeError:
            return []

    def get_species(self, species):
        expr = 'species_names/species[spp_symbol="' + species + '"]'
        elem = self.root.xpath(expr)
        return XMLSpeciesName(elem[0])

    @property
    def ordination_variables(self):
        try:
            ord_var_elem = self.root.ordination_variables
            return [
                XMLOrdinationVariable(x) for x in ord_var_elem.iterchildren()
            ]
        except AttributeError:
            return []
