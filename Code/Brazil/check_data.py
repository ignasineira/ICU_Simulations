#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 22:10:32 2022

@author: ineira

Explore data from Brazil! 

# Number of confirmed cases of COVID-19 in Brazil

Main repository: https://github.com/wcota/covid19br/

Description of the data: https://github.com/wcota/covid19br/blob/master/DESCRIPTION.en.md

"""

# import packages
import requests
import io

import pandas as pd
import  numpy as np

from matplotlib import cm
cmap = cm.get_cmap('Spectral')
import matplotlib.pyplot as plt # plotting data 
import matplotlib.dates as mdates
from datetime import datetime

pd.set_option('display.float_format', lambda x: '%.9f' % x)



#import data directly from Github
df = pd.read_csv("https://raw.githubusercontent.com/wcota/covid19br/master/cases-brazil-states.csv")
# change date column to datetime format
df['date'] = pd.to_datetime(df['date'], format= '%Y-%m-%d')

check_if=True
if check_if:
    # print all available columns
    print(df.columns)
    print(" This is data from brazil!")
    print("Nª of state: {}".format(len(df.state.unique())))
    print("Dates!")
    print("Nª dates: {}".format(len(df.date.unique())))
    print("Frist and last date: {} to {}".format(np.datetime_as_string(df.date.unique()[0], unit='D'),
                                                np.datetime_as_string(df.date.unique()[-1], unit='D')))

# filter data for Brazil, and show only the specified columns
df_br= df[df.state == 'TOTAL'][['date', 'totalCases', 'newCases','deaths','newDeaths', 'suspects', 'tests', 'vaccinated', 'vaccinated_second','vaccinated_third']].copy().reset_index()

#Check if list of dates is complete
print("Check if list of dates is complete!") 
print(df_br.date.diff().value_counts())
# create new columns
df_br['newVaccinated'] = df_br['vaccinated'].diff()
df_br['newVaccinated_second'] = df_br['vaccinated_second'].diff()
df_br['newVaccinated_third'] = df_br['vaccinated_third'].diff()
#change some values:
index_frist_dosis=df_br.loc[~df_br.vaccinated.isnull()].index[0]
index_second_dosis=df_br.loc[~df_br.vaccinated_second.isnull()].index[0]
index_third_dosis=df_br.loc[~df_br.vaccinated_third.isnull()].index[0]
print("Frist dosis was: {}".format(df_br.iloc[[index_frist_dosis]].date))#df_br.iloc[[index_second_dosis]]['date'].values[0]
print("Second dosis was: {}".format(df_br.iloc[[index_second_dosis]].date))
print("Third dosis was: {}".format(df_br.iloc[[index_third_dosis]].date))
df_br.loc[[index_frist_dosis], 'newVaccinated']=df_br.loc[[index_frist_dosis], 'vaccinated']
df_br.loc[[index_second_dosis], 'newVaccinated_second']=df_br.loc[[index_second_dosis], 'vaccinated_second']
df_br.loc[[index_third_dosis], 'newVaccinated_third']=df_br.loc[[index_third_dosis], 'vaccinated_third']

#check some data
list_sorted=list(df_br['date'].dt.strftime('%b/%y').unique())
resumen=df_br.groupby(df_br['date'].dt.strftime('%b/%y'))[['newCases','vaccinated','newDeaths']].sum().loc[list_sorted]
resumen=resumen.astype(dict.fromkeys(resumen.select_dtypes(np.floating), 'int'))




ax = plt.gca()

df_br["newDeaths_7d"] = df_br["newDeaths"].rolling(7).mean()

df_br.plot(x = "date", y="newDeaths",marker=".",lw=0, ax=ax)
df_br.plot(x = "date", y="newDeaths_7d",ax=ax,lw=3)
#df_br.plot(x = "date", y="newVaccinated",marker=".",lw=0, ax=ax)
#df_br.plot(x = "date", y="newCases",ax=ax,lw=3)

"""
falta agregar camas UCI, participacion de mercado 

"""



df_aux=df_br[(df_br.date<'2021-12-13')][['date','newVaccinated','newVaccinated_second','newVaccinated_third']].copy().set_index('date')
df_aux.dropna(inplace=True)
df_aux=df_aux.apply(lambda x: x/x.sum(), axis=1)




fig,ax = plt.subplots(figsize=(9,6))
df_aux.plot.area(ax=ax, cmap=cmap)
plt.ylim(0,1)
ax.set_yticklabels(['0','20%','40%','60%','80%','100%'])

#put the lengend outside the plot
plt.legend(bbox_to_anchor =(1.05,1))
plt.title('100 % stacked area chart')
plt.xlabel("")
half_year_locator = mdates.MonthLocator(interval=1)
year_month_formatter = mdates.DateFormatter("%Y-%m") # four digits for year, two for mon
ax.xaxis.set_major_locator(half_year_locator) # Locator for major axis only.
ax.xaxis.set_major_formatter(year_month_formatter) # formatter for major axis only
# Also moves the bottom of the axes up to make room for them.
fig.autofmt_xdate()
plt.show()





















