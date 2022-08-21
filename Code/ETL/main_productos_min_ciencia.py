#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:45:50 2022

@author: ineira
"""

import os
import sys

import pandas as pd
import numpy as np 
import itertools
import scipy.stats as st


from products_min_ciencia import producto_5,producto_7_8, producto_9, producto_10,producto_16,producto_21,producto_22, producto_24,producto_26, producto_39,get_df_date_16
from data_processing import prepare_producto_16


def time_series_inf():
    """
    Call products that contains infected data

    Returns
    -------
    df_5 : TYPE
        DESCRIPTION.
    df_16 : TYPE
        DESCRIPTION.
    df_16_26 : TYPE
        DESCRIPTION.
    df_21 : TYPE
        DESCRIPTION.
    df_26 : TYPE
        DESCRIPTION.
    df_39 : TYPE
        DESCRIPTION.

    """
    
    df_5 = producto_5()
    
    df= producto_16()
    
    #firts drop
    df = df[df.start_date!=np.datetime64('2020-10-05')]
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/github_product/'
    path=path_projet+path_data
    
    dict_between_df_9_16 = pd.read_csv(path+'dict_between_df_9_16_over_15_years.csv')
    #product 16: delete gender column and group
    df.drop(columns=['Sexo'], inplace= True)
    df.dropna(inplace=True)
    df = df.groupby(['Grupo de edad', 'start_date']).agg('sum').reset_index()
    df=pd.merge(df,dict_between_df_9_16, how='left', left_on=['Grupo de edad'], right_on=['1'])
    df.drop(columns=['1','Grupo de edad'], inplace= True)
    df = df.groupby(['0', 'start_date']).agg('sum').reset_index()
    df.rename(columns={'0': 'Grupo de edad'}, inplace=True)
    
    df_date, df_date_columns = get_df_date_16(df, missing_values=True)
    df= pd.concat([ df, df_date], ignore_index=True)
    df = df.sort_values(by=df_date_columns).reset_index(drop=True)
    df['accumulated_infected']=df.groupby(
        df_date_columns[:-1])['accumulated_infected'].apply(lambda group: group.interpolate(limit=30)).apply(np.floor)
    
    
    df['infected_today'] = df.groupby(df_date_columns[:-1])['accumulated_infected'].diff()
    
    df_16=df.groupby(['start_date']).agg('sum').reset_index()
    
    
    df_16_26 = prepare_producto_16()
    df_16_26['infected_asintomatic_today'] = df_16_26['infected_today']*(
        df_16_26['asintomatic_today']/(df_16_26['sintomatic_today']+df_16_26['asintomatic_today']))
    df_16_26=df_16_26[['Grupo de edad', 'start_date',
       'infected_today', 'infected_sintomatic_today','infected_asintomatic_today']].groupby(['start_date']).agg('sum').reset_index()
    df_21 = producto_21()
    df_26 = producto_26()
    df_26 = df_26[df_26.Region=='Total'][['start_date', 'sintomatic_today', 'asintomatic_today']].copy()
    df_26['infected_today']= df_26['sintomatic_today']+df_26['asintomatic_today']
    df_26.columns=['start_date', 'sintomatic_today_26', 'asintomatic_today_26','infected_today_26']
    df_26['accumulated_infected_26']=df_26['infected_today_26'].cumsum()
    df_39 = producto_39()
    return df_5, df_16,df_16_26, df_21,df_26,df_39






def call_df():
    """
    Main idea: 
        Call all the df of interest

    Returns
    -------
    df : TYPE
        DESCRIPTION.
    df_v0 : TYPE
        DESCRIPTION.
    df_media_movil : TYPE
        DESCRIPTION.
    df_9 : TYPE
        DESCRIPTION.
    df_10 : TYPE
        DESCRIPTION.
    df_16 : TYPE
        DESCRIPTION.
    df_22 : TYPE
        DESCRIPTION.
    dict_between_df_9_16 : TYPE
        DESCRIPTION.
    dict_between_df_9_10 : TYPE
        DESCRIPTION.

    """
    df_7,df_8 = producto_7_8()
    df_9  = producto_9()
    df_10 = producto_10()
    df_16 = producto_16()
    df_22 = producto_22()
    
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/github_product/'
    dict_between_df_9_16 = pd.read_csv(path_projet+path_data+'dict_between_df_9_16.csv')
    dict_between_df_9_10 = pd.read_csv(path_projet+path_data+'dict_between_df_9_10.csv')
    dict_between_df_9_22 = pd.read_csv(path_projet+path_data+'dict_between_df_9_22.csv')
    #product 16: delete gender column and group
    df_16.drop(columns=['Sexo'], inplace= True)
    df_16.dropna(inplace=True)
    df_16 = df_16.groupby(['Grupo de edad', 'start_date']).agg('sum').reset_index()
    df_16=pd.merge(df_16,dict_between_df_9_16, how='left', left_on=['Grupo de edad'], right_on=['1'])
    df_16.drop(columns=['1','Grupo de edad'], inplace= True)
    df_16 = df_16.groupby(['0', 'start_date']).agg('sum').reset_index()
    
    df_10.dropna(inplace=True)
    df_10=pd.merge(df_10,dict_between_df_9_10, how='left', left_on=['Grupo de edad'], right_on=['1'])
    df_10.drop(columns=['1','Grupo de edad'], inplace= True)
    df_10 = df_10.groupby(['0', 'start_date']).agg('sum').reset_index()
    
    df=pd.merge(df_9,df_16, how='left', left_on=['start_date','Grupo de edad'], right_on=['start_date','0'])
    df.drop(columns=['0'],inplace=True)
    df=pd.merge(df,df_10, how='left', left_on=['start_date','Grupo de edad'], right_on=['start_date','0'])
    df.drop(columns=['0'],inplace=True)
    df.dropna(inplace=True)
    df_v0 = df.copy()


    
    df_22.drop(columns=['Sexo'], inplace= True)
    df_22 = df_22.groupby(['Grupo de edad', 'start_date']).agg('sum').reset_index()
    df_22=pd.merge(df_22,dict_between_df_9_22[['1','2']], how='left', left_on=['Grupo de edad'], right_on=['1'])
    df_22.drop(columns=['1','Grupo de edad'], inplace= True)
    df_22 = df_22.groupby(['2', 'start_date']).agg('sum').reset_index()
    df_22['uci_today'] = df_22.groupby(['2'])['accumulated_uci_beds'].diff()/(df_22.groupby(['2'])['start_date'].diff()/ np.timedelta64(1, 'D'))
    
    df=pd.merge(df,dict_between_df_9_22[['0','2']].drop_duplicates(subset='0', keep="first"), how='left', left_on=['Grupo de edad'], right_on=['0'])
    df.drop(columns=['0', 'Grupo de edad'],inplace=True)
    df = df.groupby(['2', 'start_date']).agg('sum').reset_index()
    df=pd.merge(df,df_22, how='left', on=['start_date','2'])
    df.rename(columns={'2':'Grupo de edad'},inplace=True)
    df=df[df.start_date>='2020-04-22'].copy()
    df=df[df.start_date<='2021-02-08'].copy()
    
    #df_media_movil = meadia_movil_function(df,distribucion='norm', loc=7, scale=1, lag=14)
    
    return df,df_v0,df_7,df_8,df_9,df_10,df_16,df_22,dict_between_df_9_16,dict_between_df_9_10


    
def meadia_movil_function(df,distribucion='norm', loc=7, scale=1, lag=14):
    """
    Deprecated

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    distribucion : TYPE, optional
        DESCRIPTION. The default is 'norm'.
    loc : TYPE, optional
        DESCRIPTION. The default is 7.
    scale : TYPE, optional
        DESCRIPTION. The default is 1.
    lag : TYPE, optional
        DESCRIPTION. The default is 14.

    Returns
    -------
    aux : TYPE
        DESCRIPTION.

    """
    df_24 = producto_24()
    DISTRIBUTIONS = {
    'uniform':st.uniform,'norm':st.norm}#'lognorm':st.lognorm
    aux= df[['Grupo de edad', 'start_date','uci_today','infected_today']].copy()
    #date_df=pd.DataFrame(pd.date_range(aux.start_date.unique()[0],aux.start_date.unique()[-1] , freq='1d'))
    df_date=pd.DataFrame(list(itertools.product(pd.date_range(aux.start_date.unique()[0],aux.start_date.unique()[-1] , freq='1d'), aux['Grupo de edad'].unique())))
    df_date.columns=['start_date', 'Grupo de edad']

    aux= pd.merge(df_date, aux, how='left', on=['start_date', 'Grupo de edad'])
    
    #aux['uci_beds2']=aux.groupby(['Grupo de edad'])['uci_beds'].transform(lambda v: v.ffill())
    aux['uci_today']=aux.groupby(['Grupo de edad'])['uci_today'].transform(lambda v: v.bfill())
    aux['infected_today']=aux.groupby(['Grupo de edad'])['infected_today'].transform(lambda v: v.bfill())
    
    x , step = np.linspace(DISTRIBUTIONS[distribucion].ppf(0.01,loc,scale),
                DISTRIBUTIONS[distribucion].ppf(0.99,loc,scale), lag,retstep=True)
    
    w = DISTRIBUTIONS[distribucion].pdf(x,loc,scale) * step
    #print(w.sum())
    #shift me desplazo un periodo
    f1= lambda x:  x.rolling(lag, min_periods=lag).apply(lambda x: (x * w).sum()).shift()
    aux['infedted_media_movil']=aux.groupby(['Grupo de edad'])['infected_today'].apply(f1)
    
    aux = pd.merge(aux,df_24, how='right', on=['start_date'])
    aux['prob_uci']=100*aux['uci_today']*aux['proporcion']/aux['infedted_media_movil']
    
    return aux

if __name__ == "__main__":
    
    # df,df_v0,df_7,df_8, df_9,df_10,df_16,df_22,dict_between_df_9_16,dict_between_df_9_10=call_df()
    # df_5, df_16,df_16_26, df_21,df_26,df_39= time_series_inf()
    None