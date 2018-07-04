"""
This module runs unit tests for the package.
Some tests might take longer since they work on a test dataset.

"""
import os
import pandas as pd
import unittest
import coelurus
import ConfigParser
import shutil
config = ConfigParser.ConfigParser()
config.readfp(open('./tests/test_config.ini', 'r'))


class MalformedInputTest1(unittest.TestCase):

    def setUp(self):
        # Create a temp malformed input.

        tempdir = './tests/sample_data/temp/'
        if not os.path.exists(tempdir):
            os.mkdir(tempdir)

        test_data = pd.read_csv(config.get('data_sources', 'good_input_data_path'))
        del test_data['protein_id']
        test_data.to_csv('./tests/sample_data/temp/input_data_malformed.csv', index=False)

    def test_protein_ids(self):
        loader = coelurus.Loader('./tests/test_config.ini')
        loader.load_data()
        self.assertFalse(loader.quality_check(), 'Protein ID columns missing NOT detected.')

    def tearDown(self):
        # Remove the temp file.
        shutil.rmtree('./tests/sample_data/temp/')


class MalformedInputTest2(unittest.TestCase):

    def setUp(self):
        # Create a temp malformed input.

        tempdir = './tests/sample_data/temp/'
        if not os.path.exists(tempdir):
            os.mkdir(tempdir)

        test_data = pd.read_csv(config.get('data_sources', 'good_input_data_path'))
        del test_data[test_data.columns[1]]
        test_data.to_csv('./tests/sample_data/temp/input_data_malformed.csv', index=False)

    def test_protein_ids(self):
        loader = coelurus.Loader('./tests/test_config.ini')
        loader.load_data()
        self.assertFalse(loader.quality_check(), 'Wrong number of columns not detected.')

    def tearDown(self):
        # Remove the temp file.
        shutil.rmtree('./tests/sample_data/temp/')


if __name__ == '__main__':
    unittest.main()
