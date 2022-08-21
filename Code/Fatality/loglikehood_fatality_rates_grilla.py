#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  6 12:23:56 2022

@author: ineira


Experimento de log likehood normal distribution
calculado los parametros desde una funcion "cerrada" y usando una grilla
"""
import requests
import io
import os
import json
from urllib.request import urlretrieve #open url

import pandas as pd
import numpy as np
from math import pi

from scipy.stats.distributions import  t
from scipy import stats 
from scipy.stats import nbinom,chi2


from data_processing import prepare_producto_10,prepare_producto_16
from datetime import datetime,timedelta
pd.set_option('display.float_format', lambda x: '%.9f' % x)


from matplotlib import cm
cmap = cm.get_cmap('Spectral')
import matplotlib.pyplot as plt # plotting data 
import matplotlib.dates as mdates
from matplotlib.dates import date2num

import seaborn as sns ## These Three lines are necessary for Seaborn to work  


def loglikehood_normal_distribution(alpha_original,alpha_gamma,variance,T,dead,icu_original,icu_gamma):
    
    aux_numerador=dead-alpha_original*icu_original-alpha_gamma*icu_gamma#array
    aux_square=np.square(aux_numerador)

    return -(T/2)* np.log(2*pi*variance)-(1/2*variance)*np.sum(aux_square)

def alpha_gamma_hat(alpha_original,sum_dead,sum_icu_original,sum_icu_gamma):
    return (sum_dead-alpha_original*sum_icu_original)/sum_icu_gamma

def variance_hat(alpha_original,alpha_gamma,T,dead,icu_original,icu_gamma):
    
    aux_numerador=dead-alpha_original*icu_original-alpha_gamma*icu_gamma
    aux_square=np.square(aux_numerador)
    numerador=np.sum(aux_square)
    return numerador/T


def main_loglikehood_experiment_irrestricto(dead,icu_original,icu_gamma,interval_alpha_original=[0,3], step=0.01):
    
    array_alpha_original=np.arange(interval_alpha_original[0],interval_alpha_original[1],step)
    
    result_data = {'alpha_original':[],
                   'alpha_gamma':[],
                   'variance':[],
                   'LL':[]
                   }
    #pre calculate
    T=dead.shape[0]
    sum_dead=np.sum(dead)
    sum_icu_original=np.sum(icu_original)
    sum_icu_gamma=np.sum(icu_gamma)
    for item in array_alpha_original:
        alpha_original=item
        alpha_gamma=alpha_gamma_hat(alpha_original,sum_dead,sum_icu_original,sum_icu_gamma)
        variance=variance_hat(alpha_original,alpha_gamma,T,dead,icu_original,icu_gamma)
        ll=loglikehood_normal_distribution(alpha_original,alpha_gamma,variance,T,dead,icu_original,icu_gamma)
        
        result_data['alpha_original'].append(alpha_original)
        result_data['alpha_gamma'].append(alpha_gamma)
        result_data['variance'].append(variance)
        result_data['LL'].append(ll)
    
    return result_data

def main_loglikehood_experiment_restringido(dead,icu,interval_alpha_original=[0,3], step=0.01):
    
    array_alpha_original=np.arange(interval_alpha_original[0],interval_alpha_original[1],step)
    
    result_data = {'alpha':[],
                   'variance':[],
                   'LL':[]
                   }
    #pre calculate
    T=dead.shape[0]
    sum_dead=np.sum(dead)
    sum_icu_original=np.sum(icu)
    #sum_icu_gamma=np.sum(icu_gamma)
    aux_icu=np.zeros(T)
    for item in array_alpha_original:
        alpha_original=item
        variance=variance_hat(alpha_original,0,T,dead,icu,aux_icu)
        ll=loglikehood_normal_distribution(alpha_original,0,variance,T,dead,icu,aux_icu)
        
        result_data['alpha'].append(alpha_original)
        result_data['variance'].append(variance)
        result_data['LL'].append(ll)
    
    return result_data



def main_fit_fatalitity_loglikehood(data,dict_main_dead,W=29,modelo_irrestricto=True):
    """
    1. give a range of time fit the curve 
    2. evaluate the MSE in beteween 01-07 to 31-12
    3. save result
    
    
    """
    dead=dict_main_dead['dict_dead_variant']
    date='2021-01-01'
    start_date='2021-01-01';
    end_date='2021-05-15'
    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    
    groupID_to_group = data['groupID_to_group']
    


    frames=[]
    
    for g in range(len(groupID_to_group)):
        group_name=groupID_to_group[g]
        print(group_name)
        
        dateID_date = data['date_to_dateID'][np.datetime64(date+"T00:00:00.000000000")]
        dead=data['dead'][g,dateID_date:dateID_end_date+1]
        #x= dead['Not variant'][g,dateID_date-(W-1):dateID_end_date+1-(W-1)]
        #x=np.stack([item[g,dateID_date-(W-1):dateID_end_date+1-(W-1)] for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
        variantes=[]
        no_variante=[]
        for key,item in dict_main_dead['dict_dead_variant'].items():
            if key== 'Not variant':
                no_variante=item[g,dateID_date-(W-1):dateID_end_date+1-(W-1)]
            if key in ['variant', 'b117']:
                variantes.append(item[g,dateID_date-(W-1):dateID_end_date+1-(W-1)])
                
        icu_gamma=np.array(variantes).sum(axis=0)
        icu_original=np.array(no_variante)
                
        #x=np.stack([array_no_variantes,array_variantes],axis=0)
        
        #icu_original=x[0]
        #∫icu_gamma=x[1]
        if modelo_irrestricto:
            result_data=main_loglikehood_experiment_irrestricto(dead,icu_original,icu_gamma,interval_alpha_original=[0,3], step=0.01)
        else:
            icu=icu_gamma+icu_original
            result_data=main_loglikehood_experiment_restringido(dead,icu,interval_alpha_original=[0,3], step=0.01)
        
        
        aux_frame=pd.DataFrame.from_dict(result_data)
        aux_frame['Grupo de edad']=groupID_to_group[g]
        frames.append(aux_frame)
    df=pd.concat(frames, ignore_index=True)
    list_order_group=['<=39','40-49', '50-59', '60-69', '>=70']
    df['Grupo de edad']=pd.Categorical(df['Grupo de edad'],categories=list_order_group,ordered=True)
    if modelo_irrestricto:
        df.sort_values(by=['Grupo de edad','alpha_original'], inplace=True)
    else:
        df.sort_values(by=['Grupo de edad','alpha'], inplace=True)
    df=df.round(3)
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    
    idx = df.groupby(['Grupo de edad'])['LL'].transform(max) == df['LL']
    print(df[idx])
    min_df=df[idx].copy()
    return df





def LR_test(df_irrestricto, df_restringido):
    list_order_group=['<=39','40-49', '50-59', '60-69', '>=70']
    # interpret test-statistic
    prob = 0.95
    dof=2
    critical = chi2.ppf(prob, dof)
    
    for group in list_order_group: 
        result1=df_irrestricto[df_irrestricto['Grupo de edad']==group]['LL'].max()
        result2=df_restringido[df_restringido['Grupo de edad']==group]['LL'].max()
        LR=2*abs((np.min(result1)-np.min(result2)))
        print(f"For group: {group}")
        print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, LR))
        if abs(LR) >= critical:
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')
            
        p = 1-chi2.cdf(x=LR, df=dof)    
        print("calculate p value: {}".format(round(p,4)))

"""
df_irrestricto=main_fit_fatalitity_loglikehood(data,dict_main_dead,W=29,modelo_irrestricto=True)
df_restringido=main_fit_fatalitity_loglikehood(data,dict_main_dead,W=29,modelo_irrestricto=False)

"""






