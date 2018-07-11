# -*- coding: utf-8 -*-
"""
This module takes care of steps related to machine learning and statistics.

"""
import pandas as pd
import ConfigParser
import numpy as np
from coelurus.data_processing import Loader, Validator, DataProcessor
from sklearn.mixture import GaussianMixture
from multiprocessing.pool import ThreadPool


class FeatureIntegrator(object):
    """
    A class that integrates multiple ML/statistics classes defined in this module.
    It takes care of applying them in proper order to the data set.
    """
    def __init__(self, processor):
        self.processor = processor
        self.config = processor.config
        self.replicate_data_transformed = processor.replicate_data_transformed
        self.data_new_features = None  # this will hold newly extracted features

    def fit_gaussians(self, profiles):
        """
        Fits Bayesian Gaussian mixture model to the profile set.
        :return: Pandas DF with probabilities for each profile that it belongs to a given component.
        todo: set n_components to be a config parameter
        """
        if np.all(profiles.iloc[:, 1].isnull()):
            profiles = profiles.drop([profiles.columns[1]], axis=1)
        profiles = profiles.set_index(self.config.get('data_sources', 'input_data_protein_id'), drop=True, inplace=False)
        maxes = np.max(profiles, axis=1)
        profiles_norm = profiles.apply(lambda x: x / maxes, axis=0)  # scale 0-1 reach row

        # ML part
        # select the best number of components
        n_component_testing = range(2,
                                    self.config.getint('data_sources', 'number_of_fractions') -
                                    self.config.getint('filter_options', 'min_consecutive_fractions'))
        component_bics = []
        for n in n_component_testing:
            bmm_model = GaussianMixture(n_components=n)
            bmm_model = bmm_model.fit(profiles_norm)
            current_bic = bmm_model.bic(profiles_norm)
            component_bics.append((n, current_bic))
        component_bics.sort(key=lambda x: x[1])
        best_n = component_bics[0][0]

        # use the model with the lowest Bayesian information criterion
        print("Using {0} components, selected by BIC.".format([best_n]))
        bmm_final_model = GaussianMixture(n_components=best_n)
        bmm_final_model = bmm_final_model.fit(profiles_norm)
        bmm_final = bmm_final_model.predict_proba(profiles_norm)
        bmm_df = pd.DataFrame(bmm_final, index=profiles_norm.index)

        return bmm_df

    def extract_wrapper(self, data):
        """
        A wrapper to run feature extraction in parallel on each replicate set.
        #todo: add other extraction approaches
        """
        features = self.fit_gaussians(data)
        return [features]


    def extract_features(self):
        """
        Applies specified algorithms on the data and extracts features for integration.
        :return: Pandas DataFrame with rows being profiles and columns new features
        """
        nthreads = self.config.getint('system_options', 'num_threads')
        pool = ThreadPool(nthreads)
        results = pool.map(self.extract_wrapper, self.replicate_data_transformed)
        pool.close()
        pool.join()

        # join together different feature sources for each replicated
        results_joined = [reduce(lambda x, y: x.join(y), z) for z in results]
        self.data_new_features = results_joined


# foo = Loader("./config.ini")
# foo.load_data()
# val = Validator(foo)
# val.enforce_column_names()
# val.quality_check()
# data_filter = DataProcessor(val)
# data_filter.apply_transformations()
integrator = FeatureIntegrator(data_filter)
integrator.extract_features()

import matplotlib
matplotlib.use()