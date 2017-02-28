from pynnmap.parser import xml_parser


class XMLStandMetadataParser(xml_parser.XMLParser):

    def __init__(self, xml_file_name):
        """
        Initialize the XMLStandMetadataParser object by setting a reference
        to the XML parameter file.

        Parameters
        ----------
        xml_file_name : file
            name and location of XML metadata parameter file

        Returns
        -------
        None
        """

        super(XMLStandMetadataParser, self).__init__(xml_file_name)

    def get_attribute(self, field_name):
        elem = self.root.xpath('attribute[field_name="' + field_name + '"]')
        return XMLAttributeField(elem[0])

    @property
    def attributes(self):
        return [XMLAttributeField(x) for x in self.root.iterchildren()]


class XMLAttributeField(object):

    def __init__(self, elem):
        self.root = elem

    @property
    def field_name(self):
        return str(self.root.field_name)

    @property
    def field_type(self):
        return str(self.root.field_type)

    @property
    def units(self):
        return str(self.root.units)

    @property
    def description(self):
        return str(self.root.description)

    @property
    def short_description(self):
        return str(self.root.short_description)

    @property
    def species_attr(self):
        return int(self.root.species_attr)

    @property
    def project_attr(self):
        return int(self.root.project_attr)

    @property
    def accuracy_attr(self):
        return int(self.root.accuracy_attr)

    @property
    def codes(self):
        try:
            codes_elem = self.root.codes
            return [XMLAttributeCode(x) for x in codes_elem.iterchildren()]
        except AttributeError:
            return []


class XMLAttributeCode(object):

    def __init__(self, elem):
        self.root = elem

    @property
    def code_value(self):
        return str(self.root.code_value)

    @property
    def description(self):
        return str(self.root.description)

    @property
    def label(self):
        return str(self.root.label)
