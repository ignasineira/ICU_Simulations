#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 13:32:20 2022

@author: ineira

check the sourse of this data 
"""
import os
import sys

import pandas as pd
import numpy as np 
import itertools




def Avance_vacunacion():
    """
    Main idea: 
        load vaccination data
    make explicit the sourse of this data 

    Returns
    -------
    aux_df1 : Datframe
        DESCRIPTION.

    """
    
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/'
    df=pd.read_csv(path_projet+path_data+'Avance_vacunación_Campaña_SARS_COV_2_final.csv')
    df.rename(columns={'Fecha inmunización':'start_date'},inplace=True)
    df['start_date']=pd.to_datetime(df['start_date'])
    df = df.sort_values(by=['Grupo de edad', 'Laboratorio', 'start_date']).reset_index(drop=True)
    df_date, df_date_columns=get_df_date_vacc(df, missing_values=False,start_date=np.sort(df.start_date.unique())[0], end_date=np.sort(df.start_date.unique())[-1])
    aux_df=pd.merge(df_date,df, how='left', on=df_date_columns)
    aux_df.fillna(0,inplace=True)


    aux_df['2° Dosis']=aux_df.groupby(['Grupo de edad', 'Laboratorio'])['1° Dosis'].shift(periods=28, fill_value=0)
    
    aux_df.columns=['Grupo de edad', 'Laboratorio', 'start_date', 'Primera', 'Segunda']
    aux_df1=pd.melt(aux_df, id_vars=['Grupo de edad', 'start_date','Laboratorio'], value_vars=['Primera', 'Segunda'],var_name='Dosis', value_name='vaccinated_today')
    aux_df1['accumulated_vaccinated']=aux_df1.groupby(['Grupo de edad', 'Laboratorio','Dosis'])['vaccinated_today'].cumsum()
    aux_df1['Region']='Total'
    return aux_df1



def get_df_date_vacc(df, missing_values=True,start_date='2021-02-02', end_date='2021-02-21'):
    """
    Main idea: 
        generate all the combinations for the set  'Grupo de edad', 'Laboratorio', 'start_date'

    Parameters
    ----------
    df : Dataframe
        df from producto 77.
    missing_values : bolean, optional
        if we are interested in all possible dates True, otherwise False ,
        combinations are generated from start_date and end_date dates. 
        The default is True.
    start_date : string datetime, optional
        Suppose the day starts before the vaccine starts. The default is '2021-02-02'.
    end_date : string datetime, optional
        include that day . The default is '2021-02-21'.

    Returns
    -------
    df_date : Dataframe
        DESCRIPTION.
    df_date_columns : list of string
       ['Grupo de edad', 'Laboratorio', 'start_date']
    
    
    """
    if missing_values:
        date_arr=np.setdiff1d(
            pd.date_range(
                np.sort(df.start_date.unique())[0],
                np.sort(df.start_date.unique())[-1],
                freq='1d'),
            df.start_date.unique())
    else: 
        date_arr=pd.date_range(
            np.datetime64(start_date),#- (np.datetime64(end_date)-np.datetime64(start_date))
            np.datetime64(end_date),
            freq='1d')
    
    df_date=pd.DataFrame(
            list(itertools.product(
                    df['Grupo de edad'].unique(),
                    df['Laboratorio'].unique(),
                    date_arr)))
    df_date_columns = ['Grupo de edad', 'Laboratorio', 'start_date']
    df_date.columns=df_date_columns
    
    return df_date, df_date_columns