#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 21:23:56 2022

@author: ineira

"""

import requests
import io
import os

import pandas as pd
import  numpy as np

import json
from urllib.request import urlretrieve #open url


from datetime import datetime,timedelta
pd.set_option('display.float_format', lambda x: '%.9f' % x)


from matplotlib import cm
cmap = cm.get_cmap('Spectral')
import matplotlib.pyplot as plt # plotting data 
import matplotlib.dates as mdates
from matplotlib.dates import date2num



def download_owid_data(update_data=False):
    owid_csv_file = "owid-covid-data.csv"
    owid_csv_input_file =  f"https://covid.ourworldindata.org/data/{owid_csv_file}"
    
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/World/'
    
    owid_csv_save_file = path_projet+path_data+owid_csv_file
    os.makedirs(os.path.dirname(owid_csv_save_file), exist_ok=True)
    
    last_update_file=path_projet+path_data+"last_update_file_owid_data.json"
    os.makedirs(os.path.dirname(last_update_file), exist_ok=True)
    
    if update_data:
    
        urlretrieve(owid_csv_input_file, owid_csv_save_file)
        today= datetime.now().strftime('%Y-%m-%d')
        print(f"Update the file: {today}")
        
        with open(last_update_file, "w") as out:
            json.dump({'last_update':today},out)
    else:
        
        with open(last_update_file) as f:
            last_update=json.load(f)
        print(f"The last Update was: {last_update['last_update']}")
    
    return owid_csv_save_file


def get_owid_dataframe(columns= ["continent", "location", "date", "new_cases", "new_cases_per_million"],update_data=False):
    """
    Description of columns: https://github.com/owid/covid-19-data/blob/master/public/data/README.md
    
    
    
    Parameters
    ----------
    columns : TYPE, optional
        DESCRIPTION. The default is ["continent", "location", "date", "new_cases", "new_cases_per_million"].
    update_data : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    TYPE
        DESCRIPTION.
    
    example of analysis    
    all_data = df[["iso_code", "date", "location", "total_cases_per_million"]]

    # Drop rows starting with "OWID_". They are not for countries.
    where_iso_code_is_not_a_country = df['iso_code'].str.startswith("OWID_")
    all_data = all_data[~where_iso_code_is_not_a_country]

    """
    
    owid_csv_save_file=download_owid_data(update_data)
    if columns == None:
        return pd.read_csv(owid_csv_save_file)
    else: 
        return pd.read_csv(owid_csv_save_file, usecols=columns)