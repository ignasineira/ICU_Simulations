#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 14:21:00 2022

@author: ineira
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import poisson, geom, nbinom
from numpy.lib.stride_tricks import sliding_window_view
from datetime import  datetime
import json
import time


    
path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
path_ETL='Code/ETL'

module_path=path_projet+path_ETL
if module_path not in sys.path:
    sys.path.append(module_path)

from data_processing import read_data,prepare_population

import seaborn as sns ## These Three lines are necessary for Seaborn to work   
import matplotlib.pyplot as plt 
from matplotlib.dates import date2num
import matplotlib.dates as mdates


if __name__ == "__main__":
    data = read_data()
    
