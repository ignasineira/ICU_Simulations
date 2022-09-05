#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 15:29:08 2022

@author: ineira
"""
import pandas as pd



import seaborn as sns ## These Three lines are necessary for Seaborn to work   
import matplotlib.pyplot as plt 
from matplotlib.dates import date2num
import matplotlib.dates as mdates

plt.figure(dpi=1200)


from  products_min_ciencia import producto_10, prodcuto_57

from data_processing import prepare_producto_10


df_10=producto_10()

keep_columns=['start_date', 'accumulated_dead', 'dead_today']
df_10=df_10[keep_columns].groupby(keep_columns[0]).agg('sum')
df_57=prodcuto_57()



df=pd.merge(df_57,df_10, on='start_date', how='left')
#plot 
plt.plot(df['start_date'],df['dead_today'], drawstyle='steps', label='dead product 10')
plt.plot(df['start_date'],df['total_deads_p_57'], drawstyle='steps', label='dead product 57')
plt.grid(axis='x', color='0.95')
plt.legend(title='git hub product')
plt.title('National dead time series')
plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.show()

df_10=prepare_producto_10()
keep_columns=['start_date', 'dead_today']
df_10=df_10[keep_columns].groupby(keep_columns[0]).agg('sum')
df=pd.merge(df_57,df_10, on='start_date', how='left')
df = df[df.start_date<'2021-06-01']
df['total_deads_p_57']=df['total_deads_p_57'].rolling(21,center=True, min_periods=7).mean()
df['dead_hospitalizados']=df['hospitalizados'].rolling(21,center=True, min_periods=7).mean()

plt.plot(df['start_date'],df['dead_today'], drawstyle='steps', label='dead product 10')
plt.plot(df['start_date'],df['total_deads_p_57'], drawstyle='steps', label='dead product 57')
plt.plot(df['start_date'],df['dead_hospitalizados'], drawstyle='steps', label='hospitalized dead product 57')

plt.grid(axis='x', color='0.95')
plt.legend( loc='upper right',fontsize=7)#(title='git hub product'
plt.title('National dead time series')
plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.show()

