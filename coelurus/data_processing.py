# -*- coding: utf-8 -*-
"""
This module takes care of steps related to data reading, checking and pre-processing.
The output from the module is ready to be used by the machine learning part of the package.



"""
import pandas as pd
import ConfigParser
config = ConfigParser.ConfigParser()
config.readfp(open('./config.ini','r'))

class Loader():
    """

    """
    def __init__(self):
        self.data_source = config.get('data_sources', 'data_source')

    def load_data(self):
        if self.data_source == 'local':
            self.input_path = config.get('data_sources', 'input_data_path')
            self.input_data = pd.read_csv(self.input_path)
        elif self.data_source == 'AWS':
            # import boto3
            pass  # add AWS S3 support using boto3
