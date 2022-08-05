from pynnmap.parser import xml_parser


class Flags(object):
    CONTINUOUS = 1
    CATEGORICAL = 2
    CHARACTER = 4
    ID = 8
    PROJECT = 16
    ACCURACY = 32
    SPECIES = 64


class XMLStandMetadataParser(xml_parser.XMLParser):
    def __init__(self, xml_file_name):
        """
        Initialize the XMLStandMetadataParser object by setting a reference
        to the XML parameter file.

        Parameters
        ----------
        xml_file_name : file
            name and location of XML metadata parameter file
        """
        super(XMLStandMetadataParser, self).__init__(xml_file_name)

    def get_attribute(self, field_name):
        if elem := self.root.xpath('attribute[field_name="' + field_name + '"]'):
            return XMLAttributeField(elem[0])
        else:
            raise ValueError(f'Missing attribute: {field_name}')

    @property
    def attributes(self):
        return [XMLAttributeField(x) for x in self.root.iterchildren()]

    def attr_names(self):
        return [x.field_name for x in self.attributes]

    def filter(self, flags=None):
        fcn_crosswalk = {
            Flags.CONTINUOUS: XMLAttributeField.is_continuous_attr,
            Flags.CATEGORICAL: XMLAttributeField.is_categorical_attr,
            Flags.ID: XMLAttributeField.is_id_attr,
            Flags.CHARACTER: XMLAttributeField.is_character_attr,
            Flags.PROJECT: XMLAttributeField.is_project_attr,
            Flags.ACCURACY: XMLAttributeField.is_accuracy_attr,
            Flags.SPECIES: XMLAttributeField.is_species_attr,
        }

        # Get the checks to process
        checks = [v for k, v in fcn_crosswalk.items() if k & flags]

        # Return the fields that pass checks
        return [
            x.field_name
            for x in self.attributes
            if all(f(x) for f in checks)
        ]


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
    def codes(self):
        try:
            codes_elem = self.root.codes
            return [XMLAttributeCode(x) for x in codes_elem.iterchildren()]
        except AttributeError:
            return []

    def is_species_attr(self):
        return bool(self.root.species_attr)

    def is_project_attr(self):
        return bool(self.root.project_attr)

    def is_accuracy_attr(self):
        return bool(self.root.accuracy_attr)

    def is_continuous_attr(self):
        return self.field_type == 'CONTINUOUS'

    def is_categorical_attr(self):
        return self.field_type == 'CATEGORICAL'

    def is_character_attr(self):
        return self.field_type == 'CHARACTER'

    def is_id_attr(self):
        return self.field_type == 'ID'

    def is_continuous_accuracy_attr(self):
        return self.is_continuous_attr() and self.is_accuracy_attr()

    def is_continuous_species_attr(self):
        return self.is_continuous_attr() and self.is_species_attr()


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
