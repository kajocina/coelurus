# -*- coding: utf-8 -*-
"""
This module takes care of steps related to machine learning and statistics.

"""
import pandas as pd
import ConfigParser
import numpy as np
from data_processing import *


class GaussianFitter(object):
    """
    A class that handles everything related to fitting Gaussians to the loaded and processed data.
    Takes DataProcessor object during it's initialization.
    """

    def __init__(self, processor):
        self.processor = processor
        self.config = processor.config
        self.replicate_data_transformed = processor.replicate_data_transformed


foo = Loader("./config.ini")
foo.load_data()
val = Validator(foo)
val.enforce_column_names()
val.quality_check()
data_filter = DataProcessor(val)
data_filter.apply_transformations()
