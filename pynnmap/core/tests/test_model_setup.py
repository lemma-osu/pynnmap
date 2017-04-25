import unittest
from pynnmap.core import model_setup as ms


class SppszSetupTest(unittest.TestCase):

    def setUp(self):
        sppsz_proto_name = 'L:/resources/code/xml/sppsz_parameters.xml'
        self.model = ms.ModelSetup(sppsz_proto_name)

    def test_normal_init(self):
        d = 'D:/foo'
        self.model.create_modeling_files(
            model_directory=d,
            model_region=224,
            model_year=2005,
            create_ordination_matrices=True,
            run_ordination=True,
            create_attribute_data=True,
            create_area_estimates=True,
            create_report_metadata=True,
            create_hex_attribute_file=True,
            create_validation_attribute_file=True,
        )

# class TrecovSetupTest(unittest.TestCase):
#
#     def setUp(self):
#         trecov_proto_name = 'L:/resources/code/xml/trecov_parameters.xml'
#         self.model = ms.ModelSetup(trecov_proto_name)
#
#     def test_normal_init(self):
#         self.model.create_modeling_files(
#             model_directory = 'D:/model_root/mr224/trecov_1996',
#             model_region = 224,
#             model_year = 1996,
#             create_ordination_input=True,
#         )

if __name__ == '__main__':
    unittest.main()
