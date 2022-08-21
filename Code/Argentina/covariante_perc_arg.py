#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 18:21:00 2022

@author: ineira
"""
import sys
import os

import pandas as pd
import numpy as np
from datetime import datetime,timedelta

path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
path_world='Code/World'

module_path=path_projet+path_world
if module_path not in sys.path:
    sys.path.append(module_path)

from covariants_data import covariants_severidad

from matplotlib import cm
cmap = cm.get_cmap('Spectral')
import matplotlib.pyplot as plt # plotting data 
import matplotlib.dates as mdates
from matplotlib.dates import date2num


if __name__ == "__main__":
    country='Argentina'
    variant,variant_per=covariants_severidad(update_data=False,country='Argentina')
    gamma_columns=[]
    alpha_columns=[]
    original_columns=[]

    for column in variant_per.columns:
        if 'Gamma' in column:
            print("gamma "+column)
            gamma_columns.append(column)
        if 'Alpha' in column:
            print("Alpha "+column)
            alpha_columns.append(column)
        else:
            original_columns.append(column)
            
            
    variant_arg=pd.DataFrame(index=variant_per.index)
    variant_arg['Gamma']=variant_per[gamma_columns].sum(axis=1)
    variant_arg['Alpha']=variant_per[alpha_columns].sum(axis=1)
    variant_arg['Others']=variant_per[original_columns].sum(axis=1)
    
    start_date='2020-12-07'
    end_date='2021-06-14'
    variant_arg=variant_arg[(variant_arg.index>=start_date)&(variant_arg.index<=end_date)]
    
    #[data['date_to_dateID'][np.datetime64(date.strftime('%Y-%m-%d')+"T00:00:00.000000000")] for date in variant_arg.index.tolist() ]
    #variant_arg.Gamma.values
    #variant_arg.Alpha.values
    
    
    fig,ax = plt.subplots(figsize=(15,9))
    variant_arg.plot.area(ax=ax, cmap=cmap)
    plt.ylim(0,1)
    ax.set_yticklabels(['0','20%','40%','60%','80%','100%'])

    #put the lengend outside the plot
    plt.legend(bbox_to_anchor =(1.05,1))
    plt.title(f'Market share for covariants ({country})', fontsize=20)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Percentaje", fontsize=12)
    ax.set_xlim([variant_arg.index.min()+pd.DateOffset(-2),variant_arg.index.max()+pd.DateOffset(2)])
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # formatter for major axis only
    # Also moves the bottom of the axes up to make room for them.
    fig.autofmt_xdate()
    #plt.gcf().autofmt_xdate()
    
    ax.axvspan(date2num(datetime(2021,5,15)), date2num(datetime(2021,5,16)), 
           label="LAST DATE",color="red", alpha=0.5)
    ax.axvspan(date2num(datetime(2021,1,17)), date2num(datetime(2021,1,18)), 
           label="first dosis",color="red", alpha=0.5)
    
    plt.show()