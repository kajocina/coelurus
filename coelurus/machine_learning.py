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
        if self.config.getint('system_options','debug'):
            import logging as log
            log.basicConfig(filename='debug.log', level=log.DEBUG, format='%(asctime)s %(message)s',
                            datefmt="%Y-%m-%d %H:%M")

    def fit_gaussians(self, profiles):
        """
        Fits Gaussian mixture models to each profile with automatic component selection.
        :return: dictionary with each profile name and assigned lists of tuples
        holding means and std. dev. of fitted Gaussians
        """
        if np.all(profiles.iloc[:, 1].isnull()):
            profiles = profiles.drop([profiles.columns[1]], axis=1)
        profiles = profiles.set_index(self.config.get('data_sources', 'input_data_protein_id'), drop=True,
                                      inplace=False)
        # add negligible Gaussian noise close to 0 for each feature
        profiles[profiles == 0] = np.abs(np.random.normal(10, 1, size=profiles[profiles == 0].shape))

        # create a probability matrix to sample data for each profile
        sig_sums = profiles.sum(axis=1)
        profile_probs = profiles.apply(lambda x: x / sig_sums, axis=0)

        for profile in profile_probs:
            sampled_data = np.random.choice(np.arange(2, profile.shape[0] + 2), size=100000, p=profile.tolist())

            # temp: add noise to the sampled data
            sampled_data = sampled_data + np.random.normal(0.1,0.5,size=len(sampled_data))

            # select a Gaussian mixture model using the sampled data
            models = [GaussianMixture(n_components=i, covariance_type='spherical', reg_covar=5e6) for i in range(1, 6)]
            models = [model.fit(sampled_data.reshape(-1, 1)) for model in models]
            bics = [model.bic(sampled_data.reshape(-1, 1)) for model in models]

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


foo = Loader("./config.ini")
foo.load_data()
val = Validator(foo)
val.enforce_column_names()
val.quality_check()
data_filter = DataProcessor(val)
data_filter.apply_transformations()
integrator = FeatureIntegrator(data_filter)

# test the approach
profiles = data_filter.replicate_data_transformed[0].copy()
if np.all(profiles.iloc[:, 1].isnull()):
    profiles = profiles.drop([profiles.columns[1]], axis=1)
profiles = profiles.set_index('protein_id', drop=True,
                              inplace=False)

# create a probability matrix to sample data for each profile
sig_sums = profiles.sum(axis=1)
profile_probs = profiles.apply(lambda x: x / sig_sums, axis=0)
import matplotlib
matplotlib.use('Agg')
from plotnine import *
import warnings
warnings.filterwarnings('ignore')

example = 3
profile = profile_probs.iloc[example,:]
sampled_data = np.random.choice(np.arange(1, profile.shape[0] + 1), size=10000, p=profile.tolist())
sampled_data = sampled_data + np.random.normal(0.75, 1.0, size=len(sampled_data))

# select a Gaussian mixture model using the sampled data
models = [GaussianMixture(n_components=i, covariance_type='full',
                          tol=1e-2, n_init=1) for i in range(1,6)]
models = [model.fit(sampled_data.reshape(-1, 1)) for model in models]
bics = [model.bic(sampled_data.reshape(-1, 1)) for model in models]
models[np.argmin(bics)].means_
models[np.argmin(bics)].covariances_

plot = ggplot(pd.DataFrame({'data':sampled_data})) + geom_histogram(aes('data'))
for mean in models[np.argmin(bics)].means_:
    plot = plot + geom_vline(xintercept=mean)
plot.save('./plot.pdf')

## compare to original distribution...
original = data_filter.replicate_data[0].copy()
original.set_index('protein_id', drop=True, inplace=True)
original = pd.DataFrame(original.loc[profile.name])
original['frac'] = [i for i in range(len(original.index))]
orgplot = ggplot(original) + geom_bar(aes(x='frac', y=profile.name), stat='identity')
orgplot.save('./orgplot.pdf')

## compare to processed distribution...
processed = data_filter.replicate_data_transformed[0].copy()
processed.set_index('protein_id', drop=True, inplace=True)
processed = pd.DataFrame(processed.loc[profile.name])
processed['frac'] = [i for i in range(len(processed.index))]
processedplot = ggplot(processed) + geom_bar(aes(x='frac', y=profile.name), stat='identity')
processedplot.save('./processedplot.pdf')