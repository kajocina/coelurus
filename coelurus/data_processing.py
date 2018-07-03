# -*- coding: utf-8 -*-
"""
This module takes care of steps related to data reading, checking and pre-processing.
The output from the module is ready to be used by the machine learning part of the package.



"""
import pandas as pd
import ConfigParser
config = ConfigParser.ConfigParser()
config.readfp(open('./config.ini', 'r'))

class Loader():
    """ Input data loader.

    This class takes care of data reading and performing basic data quality checks.

    """
    def __init__(self):
        """

        self.data_source reads the config entry in config.ini regarding the source of the data.
            Entry 'local' means that the file is a resource on a local disk.
            Entry 'AWS' means that the input data is hosted using the S3 cloud service.

        """
        self.data_source = config.get('data_sources', 'data_source')
        self.input_path = None
        self.input_data = None

    def load_data(self):
        """

        Uses pandas to read the input data with profiles and stores as input_data attribute.

        """
        if self.data_source == 'local':
            self.input_path = config.get('data_sources', 'input_data_path')
            self.input_data = pd.read_csv(self.input_path)
        elif self.data_source == 'AWS':
            # import boto3
            pass  # add AWS S3 support using boto3

    def quality_check(self):
        
