# -*- coding: utf-8 -*-
"""
This module takes care of steps related to data reading, checking and pre-processing.
The output from the module is ready to be used by the machine learning part of the package.



"""
import pandas as pd
import ConfigParser
import numpy as np


class Loader:
    """ Input data loader.

    This class takes care of data reading and performing basic data quality checks.

    """
    def __init__(self, config_path):
        """
        :param config_path: path to the .ini file for the ConfigParser
        """

        self.config = ConfigParser.SafeConfigParser()
        try:
            handle = open(config_path, 'r')
            self.config.readfp(handle)
        except IOError:
            print('Cant find the config file')
            raise

        self.data_source = self.config.get('data_sources', 'data_source')
        self.input_path = None
        self.input_data = None
        self.basic_quality_passed = False

    def load_data(self):
        """

        Uses pandas to read the input data with profiles and stores as input_data attribute.

        """
        if self.data_source == 'local':
            self.input_path = self.config.get('data_sources', 'input_data_path')
            self.input_data = pd.read_csv(self.input_path)
        elif self.data_source == 'AWS':
            # import boto3
            pass  # add AWS S3 support using boto3
        else:
            self.input_data = None
            print('The config doesnt specify local/AWS data_source!')

    def quality_check(self):
        """

        Performs simple quality check for the loaded input data (profiles).

        :return: Boolean indicating if the profile input data passed initial quality checks.
        """

        if self.input_data is None:
            print("The data seems to be missing. Did you call load_data() first?")
            return False

        expected_protein_id = self.config.get('data_sources', 'input_data_protein_id')
        if not np.any(self.input_data.columns.str.contains(expected_protein_id)):
            print("The data seems to be missing the specific protein ID column. Check the config.ini.")
            return False

        num_of_columns = self.input_data.shape[1]
        num_of_fractions = self.config.getint('data_sources', 'number_of_fractions')
        num_of_replicates = self.config.getint('data_sources', 'number_of_replicates')

        if num_of_columns != 1+(num_of_fractions*num_of_replicates):
            print("Check the numbers of column in the input data."
                  "It should be 1 + (number of fractions * number of replicates)")
            return False

        numeric_cols = self.input_data.iloc[:,1:].select_dtypes(include=[np.number])

        if numeric_cols.shape[1] != num_of_fractions*num_of_replicates:
            print("Seems that some of the putative profile columns are not numeric. "
                  "Check the input file for non-numeric entries.")
            return False

        self.basic_quality_passed = True
        # Checks passed OK.
        return True

    def calc_missing_data(self):
        """
        Calculate the number of missing data in the loaded dataset.


        :return: Number of missing values in the loaded dataset.
        """
        if not self.basic_quality_passed:
            print("The basic data quality check was either not ran or didnt pass. Did you call .quality_check()?")
        else:
            return self.input_data.iloc[:, 1:].isnull().sum().sum()

