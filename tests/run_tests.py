"""
This module runs unit tests for the package.
Some tests might take longer since they work on a test dataset.

"""
import unittest
import coelurus
import ConfigParser
import shutil
config = ConfigParser.ConfigParser()
config.readfp(open('./test_config.ini', 'r'))


class MalformedInputTest1(unittest.TestCase):
    def setUp(self):
        # Create a temp malformed input.
        import pandas as pd
        test_data = pd.read_csv(config.get('data_sources', 'input_data_path'))
        del test_data['protein_id']
        test_data.to_csv('./sample_data/input_data_malformed.csv')

    def test_protein_ids(self):
        loader = coelurus.Loader()
        loader.load_data()
        self.assertFalse(loader.quality_check(), 'Protein IDs missing detected. OK.')

    def tearDown(self):
        # Remove the temp file.
        shutil.rmtree('./sample_data/input_data_malformed.csv')


if __name__ == '__main__':
    unittest.main()