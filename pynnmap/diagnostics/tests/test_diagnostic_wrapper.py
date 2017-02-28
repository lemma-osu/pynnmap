import unittest
from models.diagnostics import diagnostic_wrapper as dw
from models.parser import parameter_parser_factory as ppf


class SppszTest(unittest.TestCase):

    def setUp(self):
        parameter_file = 'D:/model_root/mr224/fc_kernel_mean_gb_5/model.xml'
        parameter_parser = ppf.get_parameter_parser(parameter_file)
        self.diagnostic_wrapper = dw.DiagnosticWrapper(parameter_parser)

    def test_accuracy_diagnostic_run(self):
        self.diagnostic_wrapper.run_accuracy_diagnostics()

    def test_outlier_diagnostic_run(self):
        self.diagnostic_wrapper.run_outlier_diagnostics()

if __name__ == '__main__':
    unittest.main()
