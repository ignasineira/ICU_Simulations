#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 20:17:22 2022

@author: ineira
"""
import pandas as pd
import numpy as np
import statsmodels.api as sm
import math


def main_fit_fatalitity_ols(data,dict_main_dead,W=29,modelo_irrestricto=True,end_date_list= ['2021-03-25','2021-03-12','2021-02-18','2021-05-14','2021-02-09']):
    """
    1. give a range of time fit the curve 
    2. evaluate the MSE in beteween 01-07 to 31-12
    3. save result
    
    
    """
    dead=dict_main_dead['dict_dead_variant']
    end_date_2020='2021-01-01'
    start_date_2020='2020-07-01'
    start_date_2021='2021-01-01';
    end_date_2021='2021-05-15'
    dateID_start_date_2021 = data['date_to_dateID'][np.datetime64(start_date_2021+"T00:00:00.000000000")]
    dateID_start_date_2020 = data['date_to_dateID'][np.datetime64(start_date_2020+"T00:00:00.000000000")]
    #dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    
    groupID_to_group = data['groupID_to_group']
    


    frames=[]
    
    for g in range(len(groupID_to_group)):
        group_name=groupID_to_group[g]
        print(group_name)
        #primero get ols model for 2020
        dateID_end_date_2020 = data['date_to_dateID'][np.datetime64(end_date_2020+"T00:00:00.000000000")]
        dead=data['dead'][g,dateID_start_date_2020:dateID_end_date_2020+1]
        #x= dead['Not variant'][g,dateID_date-(W-1):dateID_end_date+1-(W-1)]
        #x=np.stack([item[g,dateID_date-(W-1):dateID_end_date+1-(W-1)] for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
        variantes=[]
        no_variante=[]
        for key,item in dict_main_dead['dict_dead_variant'].items():
            if key== 'Not variant':
                no_variante=item[g,dateID_start_date_2020-(W-1):dateID_end_date_2020+1-(W-1)]
            if key in ['variant', 'b117']:
                variantes.append(item[g,dateID_start_date_2020-(W-1):dateID_end_date_2020+1-(W-1)])
                
        icu_gamma=np.array(variantes).sum(axis=0)
        icu_original=np.array(no_variante)
        
        
        result_data = {
                       'alpha_original_ols':[],
                       'alpha_gamma_ols':[],
                       'sd_alpha_original_ols':[],
                       'sd_alpha_gamma_ols':[],
                       'LL_OLS_2020':[],
                       'LL_OLS_2021':[]
                       }
        
        params_2020,str_err_2020,LL_2020=ols_2020(dead,icu_original)
        
        
        
        
        #segundo get ols model for 2021
        dateID_end_date_2021 = data['date_to_dateID'][np.datetime64(end_date_list[g]+"T00:00:00.000000000")]
        dead=data['dead'][g,dateID_start_date_2021:dateID_end_date_2021+1]
        #x= dead['Not variant'][g,dateID_date-(W-1):dateID_end_date+1-(W-1)]
        #x=np.stack([item[g,dateID_date-(W-1):dateID_end_date+1-(W-1)] for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
        variantes=[]
        no_variante=[]
        for key,item in dict_main_dead['dict_dead_variant'].items():
            if key== 'Not variant':
                no_variante=item[g,dateID_start_date_2021-(W-1):dateID_end_date_2021+1-(W-1)]
            if key in ['variant', 'b117']:
                variantes.append(item[g,dateID_start_date_2021-(W-1):dateID_end_date_2021+1-(W-1)])
                
        icu_gamma=np.array(variantes).sum(axis=0)
        icu_original=np.array(no_variante)
        dead_original=icu_original*params_2020[0]
        params_2021,str_err_2021,LL_2021=ols_2021(dead,dead_original, icu_gamma)
        
        print(icu_gamma[-10:],icu_original[-10:])
        #LL_2020=neg_loglikehood_normal_distribution_restringido(data,params_2020[0])
        
        
        result_data['alpha_original_ols'].append(params_2020[0])
        result_data['alpha_gamma_ols'].append(params_2021[0])
        result_data['sd_alpha_original_ols'].append(str_err_2020[0])
        result_data['sd_alpha_gamma_ols'].append(str_err_2021[0])
        result_data['LL_OLS_2020'].append(LL_2020)
        result_data['LL_OLS_2021'].append(LL_2021)
        
        print(result_data)
        aux_frame=pd.DataFrame.from_dict(result_data)
        aux_frame['Grupo de edad']=groupID_to_group[g]
        frames.append(aux_frame)
    df=pd.concat(frames, ignore_index=True)
    list_order_group=['<=39','40-49', '50-59', '60-69', '>=70']
    df['Grupo de edad']=pd.Categorical(df['Grupo de edad'],categories=list_order_group,ordered=True)
    if modelo_irrestricto:
        df.sort_values(by=['Grupo de edad'], inplace=True)
    else:
        df.sort_values(by=['Grupo de edad'], inplace=True)
    df=df.round(3)
    cols = list(df.columns)
    cols = [cols[-1]] + cols[:-1]
    df = df[cols]
    
    #idx = df.groupby(['Grupo de edad'])['LL'].transform(max) == df['LL']
   # print(df[idx])
   # min_df=df[idx].copy()
    return df

#df=main_fit_fatalitity_ols(data,dict_main_dead,W=29,modelo_irrestricto=True,end_date_list= ['2021-03-25','2021-03-12','2021-02-18','2021-05-14','2021-02-09'])
def ols_2020(dead,icu_original):

    data=[dead,icu_original]
    data_smpl = pd.DataFrame(data, index=["y", "ICU_orig"])
    data_smpl= data_smpl.T
    y_mdl = sm.OLS.from_formula("y ~ ICU_orig -1 ", data = data_smpl)
    y_mdl_fit = y_mdl.fit()
    LL=y_mdl_fit.llf
    print(f"\n Log likehhod {LL}")
    print(y_mdl_fit.summary().tables[1])
    params=y_mdl_fit.params.values
    str_err=y_mdl_fit.bse.values
    
    return params,str_err,LL

def ols_2021(dead,dead_original, icu_gamma):
    dead=dead-dead_original # out of time series 
    data=[dead,icu_gamma]
    data_smpl = pd.DataFrame(data, index=["y", "ICU_gamma"])
    data_smpl= data_smpl.T
    y_mdl = sm.OLS.from_formula("y ~ ICU_gamma -1 ", data = data_smpl)
    y_mdl_fit = y_mdl.fit()
    LL=y_mdl_fit.llf
    print(f"\n Log likehhod {LL}")
    print(y_mdl_fit.summary().tables[1])
    params=y_mdl_fit.params.values
    str_err=y_mdl_fit.bse.values
    
    return params,str_err,LL


def neg_loglikehood_normal_distribution_irresctricto(data,var):
    y,icu_original,icu_gamma=data[0],data[1],data[2]
    yGuess = (var[0]*icu_original) + (var[1]*icu_gamma) + var[2]
    f = calcLogLikelihood(yGuess, y, float(len(yGuess)))
    return (-1*f)

def neg_loglikehood_normal_distribution_restringido(data,var):
    y,icu=data[0],data[1]
    yGuess = (var[0]*icu) + var[1]
    f = calcLogLikelihood(yGuess, y, float(len(yGuess)))
    return (-1*f)
    
#   define a function to calculate the log likelihood
def calcLogLikelihood(guess, true, n):
    error = true-guess
    sigma = np.std(error)
    f = ((1.0/(2.0*math.pi*sigma*sigma))**(n/2))* \
        np.exp(-1*((np.dot(error.T,error))/(2*sigma*sigma)))
    return np.log(f)