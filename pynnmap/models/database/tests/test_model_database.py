import unittest
from models.database import plot_database as pd


class PlotDatabaseTest(unittest.TestCase):

    def setUp(self):
        self.plot_database = pd.PlotDatabase('sppsz', dsn='rocky2test_lemma')

    def test_normal(self):
        model_region = 224
        model_year = 1996
        buffer = 0
        var_types = 'DEFAULT'
        image_years = '1996, 1997, 1998, 1999, 2000'
        image_source = 'LARSE'
        image_version = 0.0
        extraction_method = 'MEAN'
        variable_filter = 'RAW'
        junk = self.plot_database.get_ordination_variable_list(
            model_region, model_year, buffer, var_types, image_years,
            image_source, image_version, extraction_method, variable_filter)
        for rec in junk:
            print rec.VARIABLE_NAME + ' : ' + rec.VARIABLE_PATH


if __name__ == '__main__':
    unittest.main()
