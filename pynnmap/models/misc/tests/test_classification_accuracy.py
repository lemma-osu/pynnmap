import os
import unittest
import filecmp
from models.misc import classification_accuracy as ca
from models.misc import utilities


class ClassificationAccuracyTest(unittest.TestCase):

    def setUp(self):
        self.data_fn = 'data/vegclass.csv'
        self.classifier_fn = 'data/vegclass.xml'
        self.e_ref = 'data/vegclass_errmat.csv'
        self.k_ref = 'data/vegclass_kappa.csv'
        self.data = utilities.csv2rec(self.data_fn)
        self.classifier = ca.Classifier.from_xml(self.classifier_fn)

    def test_kappa(self):
        obs_data = self.data.OBSERVED
        prd_data = self.data.PREDICTED
        ca.print_kappa_file(obs_data, prd_data, self.classifier, 'k.csv')
        self.assertTrue(filecmp.cmp(self.k_ref, 'k.csv'))
        os.remove('k.csv')

    def test_errmat(self):
        obs_data = self.data.OBSERVED
        prd_data = self.data.PREDICTED
        ca.print_error_matrix_file(obs_data, prd_data, self.classifier,
            'e.csv')
        self.assertTrue(filecmp.cmp(self.e_ref, 'e.csv'))
        os.remove('e.csv')

    def test_full(self):
        ca.classification_accuracy(self.data_fn, self.classifier_fn, 'k.csv',
            'e.csv', 'OBSERVED', 'PREDICTED')
        self.assertTrue(filecmp.cmp(self.k_ref, 'k.csv'))
        self.assertTrue(filecmp.cmp(self.e_ref, 'e.csv'))
        os.remove('k.csv')
        os.remove('e.csv')


if __name__ == '__main__':
    unittest.main()
