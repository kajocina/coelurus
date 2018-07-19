# sandbox for deciding on the approach
import pandas as pd
import ConfigParser
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from coelurus.data_processing import Loader, Validator, DataProcessor
from sklearn.mixture import GaussianMixture as GMM

foo = Loader("./config.ini")
foo.load_data()
val = Validator(foo)
val.enforce_column_names()
val.quality_check()
data_filter = DataProcessor(val)
data_filter.apply_transformations()

tdata = data_filter.replicate_data_transformed[0].copy()
tdata = tdata.drop([tdata.columns[1]], axis=1)
tdata = tdata.set_index('protein_id', drop=True, inplace=False)

## test a way to generate fraction samples
example = tdata.iloc[1, :].copy()
example[example == 0] = np.abs(np.random.normal(10, 1, size=example[example == 0].shape[0]))

sig_sum = example.sum()
probs = example / sig_sum
sampled_data = np.random.choice(np.arange(2, example.shape[0] + 2), size = 10000, p=probs.tolist())
sns.distplot(sampled_data, kde=False)

## test fitting GMM on the sampled data
mix1 = GMM(1)
mix1 = mix1.fit(sampled_data.reshape(-1,1))
mix1.means_
mix1.bic(sampled_data.reshape(-1,1))