#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:17:45 2022

@author: ineira

analysis of the vaccine campaign

"""
import os
import warnings
import numpy as np
import pandas as pd
import itertools
import time
import urllib
import  json

from data_processing import prepare_population

def main_vac_table(df):
    """
    Main idea: 
         execute all the function in orderto accomplish the task.

    Parameters
    ----------
    df : dataframe
        DESCRIPTION.

    Returns
    -------
    None.

    """
    flattened = porc_vac_table(df)
    porc_vac_table_reg(df)



def porc_vac_table(df):
    """
    Main idea: 
        by  'Dosis', 'Grupo de edad', 'start_date'  and the 
        accumulated_vaccinated_2
        
    generar una tabla con: Dia | Grupo Edad | PV1-1S | PV2-1S | PV1-2S | PV2-2S
    Donde PVK-JS = porcentaje de vacunados hace al menos J semanas con la K dosis

    Parameters
    ----------
    df : Dataframe
        DESCRIPTION.

    Returns
    -------
    flattened : TYPE
        DESCRIPTION.

    """
    df = df[df.Region=='Total'].copy()
    f1 = lambda x:  x.rolling(1, min_periods=1).sum().shift(periods=7, fill_value=0)
    f2 = lambda x:  x.rolling(1, min_periods=1).sum().shift(periods=14, fill_value=0)
    
    df['accumulated_vaccinated_2']=df.groupby(
        ['Region', 'Dosis', 'Grupo de edad',])['accumulated_vaccinated'].apply(f1)
    
    df['accumulated_vaccinated_3']=df.groupby(
        ['Region', 'Dosis', 'Grupo de edad',])['accumulated_vaccinated'].apply(f2)
    
    df_pop = prepare_population()
    df = pd.merge(df, df_pop, how='left', on=['Grupo de edad'])
    
    
    df['accumulated_vaccinated_2'] = df['accumulated_vaccinated_2']/df['Personas']
    df['accumulated_vaccinated_3'] = df['accumulated_vaccinated_3']/df['Personas']
    
    df = df[[ 'Dosis', 'Grupo de edad', 'start_date',
       'accumulated_vaccinated_2', 'accumulated_vaccinated_3']]
    table = pd.pivot_table(df,values=['accumulated_vaccinated_2', 'accumulated_vaccinated_3'],
                           index=['Grupo de edad', 'start_date'],
                           columns=['Dosis'])#.reset_index()
    
    flattened = pd.DataFrame(table.to_records())
    flattened.columns=['Grupo de edad', 'start_date','PV1-1S','PV2-1S','PV1-2S','PV2-2S']
    
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    save_folder=path_projet+"Data/Output/Table_marcel/"
    flattened.to_csv(save_folder+"tabla_vacunados.csv",index=False)
    return flattened



def porc_vac_table_reg(df):
    """
    Main idea: 
        by region: save a file with 'Dosis', 'Grupo de edad', 'start_date'  and the 
        accumulated_vaccinated_2
    Parameters
    ----------
    df : Dataframe
        DESCRIPTION.

    Returns
    -------
    None.

    """
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/poblacion/'
    path=path_projet+path_data
    with open(path+'dict_reg.json', 'r') as fp:
        regions = json.load(fp)
    regions= {int(k):v for k,v in regions.items()}
    
    df_pop = prepare_population_reg()
    
    for key, value in regions.items():
        df_reg = df[df.Region==value].copy()
        f1 = lambda x:  x.rolling(1, min_periods=1).sum().shift(periods=7, fill_value=0)
        f2 = lambda x:  x.rolling(1, min_periods=1).sum().shift(periods=14, fill_value=0)
        
        df_reg['accumulated_vaccinated_2']=df_reg.groupby(
            ['Region', 'Dosis', 'Grupo de edad',])['accumulated_vaccinated'].apply(f1)
        
        df_reg['accumulated_vaccinated_3']=df_reg.groupby(
            ['Region', 'Dosis', 'Grupo de edad',])['accumulated_vaccinated'].apply(f2)
        
        df_pop_reg = df_pop[df_pop.Region==key][['Grupo de edad','Personas']].copy()
        df_reg = pd.merge(df_reg, df_pop_reg, how='left', on=['Grupo de edad'])
        
        
        df_reg['accumulated_vaccinated_2'] = df_reg['accumulated_vaccinated_2']/df_reg['Personas']
        df_reg['accumulated_vaccinated_3'] = df_reg['accumulated_vaccinated_3']/df_reg['Personas']
        
        df_reg = df_reg[[ 'Dosis', 'Grupo de edad', 'start_date',
           'accumulated_vaccinated_2', 'accumulated_vaccinated_3']]
        table = pd.pivot_table(df_reg,values=['accumulated_vaccinated_2', 'accumulated_vaccinated_3'],
                               index=['Grupo de edad', 'start_date'],
                               columns=['Dosis'])#.reset_index()
        
        flattened = pd.DataFrame(table.to_records())
        flattened.columns=['Grupo de edad', 'start_date','PV1-1S','PV2-1S','PV1-2S','PV2-2S']
        
        save_folder=path_projet+"Data/Output/Table_marcel/"
        flattened.to_csv(save_folder+"tabla_vacunados_reg_"+str(key)+".csv",index=False)
    
    


def prepare_population_reg():
    """
    Main idea: Prepare population by 'Region', 'Grupo de edad'
    

    Returns
    -------
    df : Dataframe
        columns: ['Region', 'Grupo de edad', 'Personas'].

    """
    
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/poblacion/'
    path=path_projet+path_data
    df = pd.read_csv(path+'estimacion_poblacion_ine_2020_por_region.csv',sep=';')
    df['Grupo de edad']=None #np.nan
    df['Grupo de edad']=np.where((df['edad']>14)&(df['edad']<=39),'<=39',df['Grupo de edad'] )
    df['Grupo de edad']=np.where((df['edad']>39)&(df['edad']<=49),'40-49',df['Grupo de edad'] )
    df['Grupo de edad']=np.where((df['edad']>49)&(df['edad']<=59),'50-59',df['Grupo de edad'] )
    df['Grupo de edad']=np.where((df['edad']>59)&(df['edad']<=69), '60-69',df['Grupo de edad'] )
    df['Grupo de edad']=np.where(df['edad']>69,'>=70',df['Grupo de edad'] )
    
    df.dropna(inplace=True)
    df.drop(columns=['edad','Sexo'], inplace= True)
    df = df.groupby(['Region','Grupo de edad']).agg('sum').reset_index()
    

    return df

if __name__ == "__main__":
    #df = producto_77()        
    #flattened = porc_vac_table(df)
    None