from lxml import objectify

from pynnmap.misc import utilities


class XMLParser(object):

    def __init__(self, xml_file_name):
        """
        Initialize the XMLParser object by setting a reference to the
        XML file to parse.

        Parameters
        ----------
        xml_file_name : file
            name and location of XML file

        Returns
        -------
        None
        """

        # Read the XML file into an objectified lxml tree
        self.xml_tree = objectify.parse(xml_file_name)
        self.root = self.xml_tree.getroot()

        # From the XML file, extract the name of the XML schema file which
        # is used to validate the current XML
        expr = '@*[local-name()="noNamespaceSchemaLocation"]'
        schema_list = self.root.xpath(expr)
        if len(schema_list) == 1:
            self.xml_schema_file = schema_list[0]
            self.validate()

    def __repr__(self):
        """
        Return a string representation of the this XML file starting at the
        root

        Parameters
        ----------
        None

        Returns
        -------
        return_str: str
            Pretty printed representation of XML tree
        """
        return utilities.pretty_print(self.root)

    def validate(self):
        # Validate the XML schema - if the current tree doesn't validate
        # against the XML schema, this will raise an exception
        utilities.validate_xml(self.xml_tree, self.xml_schema_file)

    @property
    def tree(self):
        return self.root.getroottree()
