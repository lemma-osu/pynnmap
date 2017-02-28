import unittest
from pynnmap.parser import xml_stand_metadata_parser as xsmp


class XMLMetadataParserTest(unittest.TestCase):

    def setUp(self):
        xml_file_name = 'data/stand_attr.xml'
        self.xmp = xsmp.XMLStandMetadataParser(xml_file_name)

    def test_fcid(self):
        field = 'FCID'
        elem = self.xmp.get_attribute(field)
        self.assertEqual(elem.field_name, field)
        self.assertEqual(elem.field_type, 'ID')
        self.assertEqual(elem.units, 'none')
        self.assertEqual(elem.description,
            'Forest class identification number assigned by LEMMA')
        self.assertEqual(elem.short_description,
            'Forest class identification number assigned by LEMMA')
        self.assertEqual(elem.codes, [])

    def test_vegclass(self):
        field = 'VEGCLASS'
        elem = self.xmp.get_attribute(field)
        self.assertEqual(elem.field_name, field)
        self.assertEqual(elem.field_type, 'CATEGORICAL')
        self.assertEqual(elem.units, 'none')
        self.assertEqual(elem.description,
            'Vegetation class based on CANCOV, BAH_PROP, QMDA_DOM')
        self.assertEqual(elem.short_description,
            'Vegetation class based on CANCOV, BAH_PROP, QMDA_DOM')

        check_codes = [
            ('1', 'Sparse'),
            ('2', 'Open'),
            ('3', 'Blf - Sm'),
            ('4', 'Blf - Md/Lg'),
            ('5', 'Mix - Sm'),
            ('6', 'Mix - Md'),
            ('7', 'Mix - Lg'),
            ('8', 'Con - Sm'),
            ('9', 'Con - Md'),
            ('10', 'Con - Lg'),
            ('11', 'Con - VLg'),
        ]

        for (i, code) in enumerate(elem.codes):
            self.assertEqual(code.code_value, check_codes[i][0])
            self.assertEqual(code.label, check_codes[i][1])

if __name__ == '__main__':
    unittest.main()
