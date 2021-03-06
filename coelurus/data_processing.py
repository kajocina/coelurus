# -*- coding: utf-8 -*-
"""
This module takes care of steps related to data reading, checking and pre-processing.
The output from the module is ready to be used by the machine learning part of the package.


"""
import pandas as pd
import ConfigParser
import numpy as np
from multiprocessing.pool import ThreadPool


class Loader(object):
    """ Input data loader.

    This class takes care of data reading and performing basic data quality checks.
    todo: add automatic re-ordering of columns (if ID columns is not first)

    """
    def __init__(self, config_path):
        """
        :param config_path: path to the .ini file for the ConfigParser
        """

        self.config = ConfigParser.SafeConfigParser()
        try:
            with open(config_path, 'r') as handle:
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


class Validator(object):
    """
    Performs initial quality check on the dataset.
    """
    def __init__(self, loader):
        self.loader = loader
        self.config = loader.config
        self.input_data = loader.input_data.copy()
        self.basic_quality_passed = False

    def get_expected_colnames(self):
        """
        Based on the config, check expected column names.
        :return: A list of expected columns names.
        """
        import string

        num_of_fractions = self.config.getint('data_sources', 'number_of_fractions')
        num_of_replicates = self.config.getint('data_sources', 'number_of_replicates')
        name_numbers = [[x for y in range(num_of_replicates)] for x in range(num_of_fractions)]
        name_numbers = [x + 1 for y in name_numbers for x in y]  # flatten
        name_reps = [string.ascii_uppercase[x] for x in range(num_of_replicates) * num_of_fractions]
        expected_names = map(lambda x: 'F' + str(x[0]) + x[1], zip(name_numbers, name_reps))

        return expected_names

    def quality_check(self):
        """

        Performs simple quality check for the loaded input data (profiles).

        :return: Boolean indicating if the profile input data passed initial quality checks.
        """

        if self.input_data is None:
            print("The data seems to be missing. Did you call load_data() in the Loader first?")
            return False

        if self.input_data.shape[0] < 10:
            print("The input data has less than 10 rows/profiles. Check the file again?")
            return False

        if self.input_data.shape[1] <= self.config.getint('filter_options', 'remove_n_last_fracs'):
            print("Config.ini is set to remove too many right-most (last) factions (remove_n_last_fracs). Change it!")
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

        numeric_cols = self.input_data.iloc[:, 1:].select_dtypes(include=[np.number])

        if numeric_cols.shape[1] != num_of_fractions*num_of_replicates:
            print("Seems that some of the putative profile columns are not numeric. "
                  "Check the input file for non-numeric entries.")
            return False

        # Check feature column naming
        colnames = self.input_data.columns[1:].tolist()
        expected_names = self.get_expected_colnames()

        if colnames != expected_names:
            print("Wrong feature (fraction) column naming. It should be F+frac_number+replicate, e.g. F1A,F2B,F30C.")
            return False

        self.basic_quality_passed = True

        print("Initial data checks passed OK.")
        return True

    def enforce_column_names(self):
        """
        Enforces proper column names based on the number of fractions and replicates specified in the config file.
        """
        enforced_labels = self.get_expected_colnames()
        try:
            self.input_data.columns = [self.config.get('data_sources','input_data_protein_id')] + enforced_labels
            print("Labels enforced. New column names are: %s" % self.input_data.columns)
        except ValueError:
            print("Can't enforce labels. Are the numbers of fractions and replicates in the config.ini correct?")
            raise


class DataProcessor(object):
    """
    Prepares input data for Gaussian fitting using option from the specified config file.
    1. Splits data by replicates into a list.
    2. Sets NAs to 0s
    3. Filters-out profiles with less than 'min_consecutive_fractions' consecutive non-0 datapoints.
    4. Filters-out profiles with less than 'min_signal_to_noise_ratio' signal to noise ratio.
    """
    def __init__(self, validator):

        from copy import copy
        self.validator = copy(validator)
        self.input_data = validator.input_data.copy()
        self.config = validator.config

        if not self.validator.basic_quality_passed:
            print("Data has not being checked for consistency with config. Running Validator.quality_check() first!")
            self.validator.quality_check()

        self.replicate_data = self.split_reps_to_list(self.input_data)
        self.replicate_data_transformed = None
        self.data_imputed = False

    def split_reps_to_list(self, data):
        """
        Split the input data into separate replicates (based on the letter at the end of the column name)
        :param data: Input data (profiles), pandas DataFrame.
        :return: List of pandas DataFrames, each containing a separate replicate
        """
        data_list = []
        colnames = self.input_data.columns[1:].tolist()
        reps = [name[-1] for name in colnames]
        for rep in sorted(set(reps)):
            rep_ixs = [i+1 for i, x in enumerate(reps) if x == rep]
            rep_ixs = [0] + rep_ixs
            rep_data = data.iloc[:, rep_ixs]
            data_list.append(rep_data)

        return data_list

    def impute_missing_values(self, input_data):
        """
        Imputes missing values if an NA or 0 is present between two non-null values using mean imputation.
        """
        profiles = input_data.iloc[:, 1:].copy()
        profiles[profiles == 0] = np.nan

        def mean_impute(arr):
            """
            Imputes mean value in the array.
            :param arr: 3 element array, middle value to be imputed using a mean
            :return: imputed array
            """
            if arr[1] == np.nan:
                print("Mid value in the array was not NaN, something went wrong with imputation.")
                return arr

            arr[1] = np.round((arr[0]+arr[2])/2, 3)
            return arr

        for i in range(profiles.shape[1] - 2):
            window = profiles.iloc[:, i:i+3]
            # which rows should be imputed (only if middle value is the only missing)
            imp_row = np.all(window.isna().apply(lambda x: x == [False, True, False], axis=1), axis=1)
            imp_row = np.where(imp_row)[0]
            profiles.iloc[imp_row, i:i+3] = window.loc[imp_row].apply(lambda x: mean_impute(x), axis=1)

        input_data.iloc[:, 1:] = profiles
        input_data = self.set_nas_to_0(input_data)
        self.data_imputed = True

        return input_data

    def smooth_profiles(self, input_data):
        """
        Performs profile smoothing using rolling window of size 3 and mean values.
        :return: pandas DataFrame with smoothened input profiles.
        """
        winsize = self.config.getint('filter_options', 'smooth_window_size')
        min_periods = winsize - 1
        input_data.iloc[:, 1:] = input_data.iloc[:, 1:].rolling(winsize, min_periods=min_periods, axis=1).median()

        return input_data

    def set_nas_to_0(self, input_data):
        """
        Sets all NA values to 0.
        :param input_data: pandas DataFrame with input profiles.
        :return: pandas DataFrame with input profiles.
        """
        input_data.iloc[:, 1:] = input_data.iloc[:, 1:].fillna(value=0)
        return input_data

    def filter_missing_profiles(self, input_data):
        """
        Removes rows that do not have at least 'min_consecutive_fractions' consecutive fractions above 0.
        todo: write test for this function!
        :param input_data: pandas DataFrame with input profiles.
        :return: pandas DataFrame with filtered input profiles.
        """
        profiles = input_data.iloc[:, 1:]
        # set to remove all profiles by default, unless 5 consec. values found
        rows_out = np.full(profiles.shape[0], False, dtype=bool)
        # use a sliding window approach
        window_width = self.config.getint('filter_options', 'min_consecutive_fractions')
        num_windows = (profiles.shape[1]) - window_width + 1
        for i in range(num_windows):
            r_idx = i + window_width
            nonzero_boolean = profiles.iloc[:, i:r_idx] != 0
            rows_to_keep = np.all(nonzero_boolean, axis=1)
            rows_out = rows_out | rows_to_keep

        rows_stay = rows_out
        input_data = input_data.loc[rows_stay, :]

        return input_data

    def filter_signal_to_noise(self, input_data):
        """
        Sets values that are below configured threshold * max value (for each profile), to zeros.
        :return: pandas DataFrame with filtered input profiles.
        todo: seems that the filter is not very handy yet? remove?
        """
        threshold = self.config.getfloat('filter_options', 'min_signal_to_noise')
        profiles = input_data.iloc[:, 1:]
        max_values = profiles.apply(max, 1)
        thresh_values = threshold*max_values
        profiles[profiles.apply(lambda x: x < thresh_values, axis=0)] = 0
        input_data.iloc[:, 1:] = profiles

        return input_data

    def remove_singletons(self, input_data):
        """
        Sets features surrounded by 0's to 0.
        :param input_data:
        :return:
        todo: fill docstrings
        """
        profiles = input_data.iloc[:, 1:]

        for i in range(profiles.shape[1] - 2):
            window = profiles.iloc[:, i:i + 3]
            win_bool = window == 0
            idx_to_flatten = win_bool.apply(lambda x: np.all(x == [True, False, True]), axis=1)
            window.loc[idx_to_flatten, window.columns[1]] = 0
            profiles.iloc[:, i:i + 3] = window
        input_data.iloc[:, 1:] = profiles

        return input_data

    def remove_n_last_fractions(self, input_data):
        """
        Removes N last fractions if there was a cut off in the experiment (it contains a lot of proteins
        that would elute later).
        :return: pandas DataFrame with N last fractions removed
        """
        new_rightmost_ix = input_data.shape[1] - self.config.getint('filter_options', 'remove_n_last_fracs')
        return input_data.iloc[:, 0:new_rightmost_ix]

    def transform_wrapper(self, data):
        """
        A wrapper to pool tasks to be run in parallel by apply_transformations().
        :param data: A pandas DataFrame with given replicate subset of the profiles.
        :return: A pandas DataDrame with transformations applied to the data.
        """

        data = self.set_nas_to_0(data)

        if self.config.getint('filter_options', 'remove_n_last_fracs') > 0:
            data = self.remove_n_last_fractions(data)

        data = self.impute_missing_values(data)
        data = self.remove_singletons(data)
        data = self.filter_missing_profiles(data)

        if self.config.getint('filter_options', 'enable_smoothing'):
            data = self.smooth_profiles(data)

        if self.config.getfloat('filter_options', 'min_signal_to_noise') > 0:
            data = self.filter_signal_to_noise(data)

        return data

    def apply_transformations(self):
        """
        Applies the filter functions to the data.
        :return:
        """
        nthreads = self.config.getint('system_options', 'num_threads')
        pool = ThreadPool(nthreads)

        results = pool.map(self.transform_wrapper, self.replicate_data)
        pool.close()
        pool.join()
        self.replicate_data_transformed = results
        self.data_imputed = True
        print("Filters applied to .replicate_data list.")