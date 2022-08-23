#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 20:18:53 2022

@author: ineira


Experimento de log likehood normal distribution
minimizando 
"""

import requests
import io
import sys
import os
import json
from urllib.request import urlretrieve #open url

import pandas as pd
import numpy as np
from math import pi
import math

from scipy.stats.distributions import  t
from scipy import stats 
from scipy.stats import nbinom,chi2,norm,f
from scipy.optimize import minimize,curve_fit

import statsmodels.api as sm

path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]
if path_projet[-1]!='/':
    path_projet+='/'
path_projet+='ICU_Simulations/'
path_ETL='Code/ETL'
module_path=path_projet+path_ETL
if module_path not in sys.path:
    sys.path.append(module_path)
    
from data_processing import read_data,prepare_producto_10,prepare_producto_16

path_icu='Code/icu_simulation'
module_path=path_projet+path_icu
if module_path not in sys.path:
    sys.path.append(module_path)
from main_simulation_icu_2021 import call_icu_simulation_2021

from datetime import datetime,timedelta
pd.set_option('display.float_format', lambda x: '%.9f' % x)


from matplotlib import cm
cmap = cm.get_cmap('Spectral')
import matplotlib.pyplot as plt # plotting data 
import matplotlib.dates as mdates
from matplotlib.dates import date2num

import seaborn as sns ## These Three lines are necessary for Seaborn to work  

def neg_loglikehood_normal_distribution_irresctricto(data, log_params):
    y,icu_original,icu_gamma=data[0],data[1],data[2]
    
    alpha_original,alpha_gamma,sigma2= log_params[0],log_params[1],log_params[2]
    
    #transfor N(0,1)
    y_nor=(y-(alpha_original*icu_original+alpha_gamma*icu_gamma))/np.sqrt(sigma2)
    #Calculate the neg log-likelihood for normal distribution
    log_likehood=np.sum(np.log(norm.pdf(y_nor)))
    return -log_likehood


#   define a function to calculate the log likelihood
def calcLogLikelihood(guess, true, n):
    error = true-guess
    sigma = np.std(error)
    f = ((1.0/(2.0*math.pi*sigma*sigma))**(n/2))* \
        np.exp(-1*((np.dot(error.T,error))/(2*sigma*sigma)))
    return np.log(f)

def myFunction(data,var):
    y,icu_original,icu_gamma=data[0],data[1],data[2]
    yGuess = (var[0]*icu_original) + (var[1]*icu_gamma) + var[2]
    f = calcLogLikelihood(yGuess, y, float(len(yGuess)))
    return (-1*f)

def myFunction_restringido(data,var):
    y,icu=data[0],data[1]
    yGuess = (var[0]*icu) + var[1]
    f = calcLogLikelihood(yGuess, y, float(len(yGuess)))
    return (-1*f)



def neg_loglikehood_normal_distribution_restringido(data,log_params):
    y,icu=data[0],data[1]
    alpha,sigma2=log_params[0],log_params[1]
    #transfor N(0,1)
    y_nor=(y-icu*alpha)/np.sqrt(sigma2)
    #Calculate the neg log-likelihood for normal distribution
    log_likehood=np.sum(np.log(norm.pdf(y_nor)))
    return -log_likehood





def main_loglikehood_experiment_irrestricto(dead,icu_original,icu_gamma):
    result_data = {'alpha_original':[],
                   'alpha_gamma':[],
                   'sigma2':[],
                   'LL':[], 
                   'sd_original':[],
                   'sd_gamma':[],
                   'sd_sigma2':[],
                   'alpha_original_ols':[],
                   'alpha_gamma_ols':[],
                   'sd_alpha_original_ols':[],
                   'sd_alpha_gamma_ols':[],
                   'LL_OLS':[],
                   'f':[],
                   'p':[],
                   'NULL_HYPOTHESIS':[]
                   }
    theta_start =[0.2,0.2,1]
    data=[dead,icu_original,icu_gamma]
    res = minimize(fun=lambda log_params, data: neg_loglikehood_normal_distribution_irresctricto(data,log_params),
        x0=theta_start,args=(data,), method = 'BFGS',bounds=((0,0,0),(1,1,np.inf)),
    	       options={'disp': False})
    
    res2 = minimize(fun=lambda var, data: myFunction(data,var),
        x0=theta_start,args=(data,), method = 'BFGS',
    	       options={'disp': False})
    print("Estimated parameters:\n")
    print(res2.x)
    print(np.sqrt(np.diag(res2.hess_inv)))
    #se:=1/(hessiano)**(1/2)
    se = np.sqrt(np.diag(res.hess_inv))
    theta = res.x
    # Put Results in a DataFrame
   
    print("Number of Function Iterations: ", res.nfev)
    print("Estimated parameters:\n")
    print(theta)
    
    """
    initParams = [0.2,0.2,1]
    results = minimize(fun=lambda log_params, data: neg_loglikehood_normal_distribution_irresctricto(data,log_params),
        x0=initParams,args=(data,), method='Nelder-Mead')
    print(results.x)
    """
    print("\nEstimated Hessian:\n")
    print(se)
    result_data['alpha_original'].append(theta[0])
    result_data['alpha_gamma'].append(theta[1])
    result_data['sigma2'].append(theta[2])
    
    
    ll=-neg_loglikehood_normal_distribution_irresctricto(data,theta)
    result_data['LL'].append(ll)
    
    result_data['sd_original'].append(se[0])
    result_data['sd_gamma'].append(se[1])
    result_data['sd_sigma2'].append(se[2])
    
    
    N=len(dead)
    
    #f, p, NULL_HYPOTHESIS=f_test(se[0], se[1],N,alpha=0.05,alt="two_sided")
    f, p, NULL_HYPOTHESIS,params,str_err,LL =f_test_OLS(data)
    result_data['f'].append(f)
    result_data['p'].append(p)
    result_data['NULL_HYPOTHESIS'].append(NULL_HYPOTHESIS)
    result_data['alpha_original_ols'].append(params[0])
    result_data['alpha_gamma_ols'].append(params[1])
    result_data['sd_alpha_original_ols'].append(str_err[0])
    result_data['sd_alpha_gamma_ols'].append(str_err[1])
    result_data['LL_OLS'].append(LL)
    return result_data

def main_loglikehood_experiment_restringido(dead,icu):
    result_data = {'alpha':[],
                   'sigma2':[],
                   'LL':[], 
                   'sd_original':[],
                   'sd_sigma2':[],
                   'alpha_original_ols':[],
                   'sd_alpha_original_ols':[],
                   'LL_OLS':[]
                   }
    theta_start =[0.5,1]
    data=[dead,icu]
    res = minimize(fun=lambda log_params, data: neg_loglikehood_normal_distribution_restringido(data,log_params),
        x0=theta_start,args=(data,), method = 'BFGS',bounds=((0,0),(1,np.inf)),
    	       options={'disp': False})
    #se:=1/(hessiano)**(1/2)
    se = np.sqrt(np.diag(res.hess_inv))
    theta = res.x
    # Put Results in a DataFrame
    #results_ = pd.DataFrame({'Parameter':theta,'Std Err':se})
    print("Number of Function Iterations: ", res.nfev)
    print("Number of Function Iterations: ", res.nfev)
    print("Estimated parameters:\n")
    print(theta)
    
    
    initParams = [1, 1]
    results = minimize(fun=lambda log_params, data: neg_loglikehood_normal_distribution_restringido(data,log_params),
        x0=initParams,args=(data,), method='Nelder-Mead')
    print(results.x)
    
    print("\nEstimated Hessian:\n")
    print(se)
    result_data['alpha'].append(theta[0])
    result_data['sigma2'].append(theta[1])
    
    
    ll=-neg_loglikehood_normal_distribution_restringido(data,theta)
    result_data['LL'].append(ll)
    
    result_data['sd_original'].append(se[0])
    result_data['sd_sigma2'].append(se[1])
    
    N=len(dead)
    
    #f, p, NULL_HYPOTHESIS=f_test(se[0], se[1],N,alpha=0.05,alt="two_sided")
    data_smpl = pd.DataFrame(data, index=["y", "ICU"])
    data_smpl= data_smpl.T
    y_mdl = sm.OLS.from_formula("y ~ ICU-1 ", data = data_smpl)
    y_mdl_fit = y_mdl.fit()
    print(y_mdl_fit.summary().tables[1])
    LL=y_mdl_fit.llf
    print(f"\n Log likehhod {LL}")
    params=y_mdl_fit.params.values
    str_err=y_mdl_fit.bse.values
    result_data['alpha_original_ols'].append(params[0])
    result_data['sd_alpha_original_ols'].append(str_err[0])
    result_data['LL_OLS'].append(LL)
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
            
            
            result_data=main_loglikehood_experiment_irrestricto(dead,icu_original,icu_gamma)
            
        else:
            icu=icu_gamma+icu_original
            result_data=main_loglikehood_experiment_restringido(dead,icu)
        
        
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


def f_test_OLS(data):
    
    
    data_smpl = pd.DataFrame(data, index=["y", "ICU_orig", "ICU_gamma"])
    data_smpl= data_smpl.T
    y_mdl = sm.OLS.from_formula("y ~ ICU_orig + ICU_gamma -1 ", data = data_smpl)
    y_mdl_fit = y_mdl.fit()
    LL=y_mdl_fit.llf
    print(f"\n Log likehhod {LL}")
    print(y_mdl_fit.summary().tables[1])
    fres = y_mdl_fit.f_test("ICU_orig - ICU_gamma = 0")
    print(fres)
    params=y_mdl_fit.params.values
    str_err=y_mdl_fit.bse.values
    f_value=fres.fvalue[[0]]
    p_value=fres.pvalue
    print("if The  p-value > 0.05, so we do not reject the null hypothesis.")
    if p_value>0.05:
        null_hypothesis="not reject"
    else: 
        null_hypothesis="reject"
    return f_value,p_value,null_hypothesis,params,str_err,LL
    
    

#define F-test function
def f_test(sd_ori, sd_gamm,N,alpha=0.05,alt="two_sided"):
    """
    F-Test for Equality of Two Variances

    H0: σ12 = σ22 (the population variances are equal)
    H1: σ12 ≠ σ22 (the population variances are not equal)
    
	The hypothesis that the two standard deviations are equal is rejected if
    F > F(alpha,N1-1,N2-1)     for an upper one-tailed test
    
    F < F(1-alpha,N1-1,N2-1)     for a lower one-tailed test
    
    F < F(1-alpha/2,N1-1,N2-1)   for a two-tailed test
    or
    
    F > F(alpha/2,N1-1,N2-1)
    
    Note on F value: F value is inversely related to p value and higher F value 
    (greater than F critical value) indicates a significant p value.
    
    Parameters
    ----------
    x : TYPE
        The first group of data.
    y : TYPE
        The second group of data.
    alt : string
        The alternative hypothesis, one of "two_sided" (default), "greater" or "less"
    
    Returns: a tuple with the F statistic value and the p-value.
    -------
    f : TYPE
        DESCRIPTION.
    p : TYPE
        DESCRIPTION.
        
        
    links:
        1. https://www.statisticshowto.com/probability-and-statistics/hypothesis-testing/f-test/
        2. https://www.cuemath.com/data/f-test/
        3. http://atomic.phys.uni-sofia.bg/local/nist-e-handbook/e-handbook/eda/section3/eda359.htm
        4. https://www.itl.nist.gov/div898/handbook/eda/section3/eda359.htm 

    """
    f_hat = (sd_ori/sd_gamm)**2#calculate F test statistic 
    dfn = N-1 #define degrees of freedom numerator 
    dfd = N-1 #define degrees of freedom denominator 

    if alt == "two_sided":
        # two-sided by default
        p = 2*(1-f.cdf(f_hat, dfn, dfd)) #find p-value of F test statistic 
    
    if alt == "greater" or f_hat<=1:
        p = 1-f.cdf(f_hat, dfn, dfd) #find p-value of F test statistic 
    if alt == "less" or f_hat<=1:
        p = f.cdf(f_hat, dfn, dfd) #find p-value of F test statistic 
    print("""
                           F TEST
    NULL HYPOTHESIS UNDER TEST--SIGMA1 = SIGMA2
    ALTERNATIVE HYPOTHESIS UNDER TEST--SIGMA1 NOT EQUAL SIGMA2
    
          """)
        
    
    
    test="""
    TEST:
    F TEST STATISTIC VALUE      =   {}
    DEG. OF FREEDOM (NUMER.)    =    {}
    DEG. OF FREEDOM (DENOM.)    =    {}
    Significance level:  α = {}
    Critical values:  F(1-α/2,N1-1,N2-1) = {}
                      F(α/2,N1-1,N2-1) = {} 
    Rejection region:  Reject H0 if F < {} or F > {}
    F TEST STATISTIC CDF VALUE  =    {}
  
   NULL          NULL HYPOTHESIS        NULL HYPOTHESIS
   HYPOTHESIS    ACCEPTANCE INTERVAL    CONCLUSION
 SIGMA1 = SIGMA2    (0.000,0.950)         {}
    
    """.format(f_hat,dfn,dfd,alpha,1-f.cdf(1-alpha/2, dfn, dfd),1-f.cdf(alpha/2, dfn, dfd),
    1-f.cdf(1-alpha/2, dfn, dfd),1-f.cdf(alpha/2, dfn, dfd),p,None)
    print(test)
    
    NULL_HYPOTHESIS="fail to reject H0"
    if (1-f.cdf(1-alpha/2, dfn, dfd))<f_hat:
        NULL_HYPOTHESIS="reject H0"
    if (1-f.cdf(alpha/2, dfn, dfd))>f_hat:
        NULL_HYPOTHESIS="reject H0"
    return f_hat, p, NULL_HYPOTHESIS




def LR_test(df_irrestricto, df_restringido, prob = 0.95):
    list_order_group=['<=39','40-49', '50-59', '60-69', '>=70']
    # interpret test-statistic
    #prob = 0.95
    dof=2
    critical = chi2.ppf(prob, dof)
    
    for group in list_order_group: 
        result1=df_irrestricto[df_irrestricto['Grupo de edad']==group]['LL_OLS'].max()
        result2=df_restringido[df_restringido['Grupo de edad']==group]['LL_OLS'].max()
        LR=2*abs((np.min(result1)-np.min(result2)))
        print(f"For group: {group}")
        print('probability=%.3f, critical=%.3f, stat=%.3f' % (prob, critical, LR))
        if abs(LR) >= critical:
            print('Dependent (reject H0)')
        else:
            print('Independent (fail to reject H0)')
            
        p = 1-chi2.cdf(x=LR, df=dof)    
        print("calculate p value: {}".format(round(p,4)))


if __name__ == '__main__':
    data = read_data()
    uci,dead,dict_main_dead = call_icu_simulation_2021(data,params=None,
                        save_data=False,
                        update_data=False)
    df_irrestricto=main_fit_fatalitity_loglikehood(data,dict_main_dead,W=29,modelo_irrestricto=True)
    df_restringido=main_fit_fatalitity_loglikehood(data,dict_main_dead,W=29,modelo_irrestricto=False)
    LR_test(df_irrestricto, df_restringido, prob = 0.95)

