# sandbox for deciding on the approach
import pandas as pd
import ConfigParser
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("TkAgg")
from coelurus.data_processing import Loader, Validator, DataProcessor
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture

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

mix = GaussianMixture(n_components=1)
test1 = mix.fit(tdata.iloc[0, :].as_matrix().reshape(-1, 1))
test1.means_
