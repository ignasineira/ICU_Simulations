#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 14:08:24 2022

@author: ineira

orden de los grupos etarios:

    ["total"]
CONFIRMACION PCR desde INICIO DE LOS SINTOMAS': 

    [7]
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import poisson, geom, nbinom
from numpy.lib.stride_tricks import sliding_window_view
from datetime import  datetime
import json
import time


    
path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
path_ETL='Code/Argentina'

module_path=path_projet+path_ETL
if module_path not in sys.path:
    sys.path.append(module_path)
    
    

from data_processing_arg import read_data
from simulation_arg_v1 import *

def find_probability_icu(data, start_date, end_date,list_p_in_uci,list_n_in_uci):
    
    groupID_to_group = data['groupID_to_group']
    dateID_to_date = data['dateID_to_date']
    date_to_dateID = data["date_to_dateID"]
    #start_date='2020-10-01'  

    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    aux=np.array([list_n_in_uci,list_p_in_uci]).T
    #shif curve
    shif_curve=[8, 7, 8, 7, 4]
    shif_curve2=[10, 10, 12, 8, 11]
    inf = data['inf'].copy()
    """
    for i,item in enumerate(shif_curve):
        inf[i,0:-(item-1)]=data['inf'][i,item-1:]
    #for i,item in enumerate(shif_curve2):
        #inf[i,item-1:]=inf[i,0:-(item-1)]
    """
    factor=0
    L=np.mean(data['uci'][:,dateID_start_date:dateID_end_date+1], axis=1)
    W=np.array([int(nbinom(item[0],item[1]).mean()) for item in aux ])
    λ = np.mean(inf[:, dateID_start_date+factor:dateID_end_date+1+factor], axis=1)
    probability_not_vac = L/ (W * λ)
    
    return probability_not_vac

def genere_list_minsal_denis(prob_inicial,pt_factor=[0.16,0.67],minsal_factor=[0.43,0.89]):
    num_b=len(prob_inicial)
    aux_factor1=(1-minsal_factor[0])/(1-pt_factor[0])
    aux_factor2=(1-minsal_factor[1])/(1-pt_factor[1])
    factor_vac1=np.round(np.array([[1,1,aux_factor1,aux_factor1,aux_factor1,aux_factor1]]*num_b),4)
    factor_vac2=np.round(np.array([[aux_factor1,aux_factor1,aux_factor2,aux_factor2,aux_factor2,aux_factor2]]*num_b),4)
    
    vac_1 = np.round((prob_inicial*factor_vac1.T).T,4).tolist()
    vac_2 = np.round((prob_inicial*factor_vac2.T).T,4).tolist()
    return vac_1,vac_2




if __name__ == "__main__":
    
    data = read_data()
    increase_vac1=[[0.0, 0.0, 0.16, 0.16, 0.16, 0.16]]
    increase_vac2=[[0.16, 0.16, 0.636, 0.636, 0.636, 0.636]]
    
    probability_not_vac=[0.011]
    start_date='2020-07-01';end_date='2020-12-31'
    list_p_in_uci=[0.069];
    list_n_in_uci=[1.9495];
    
    probability_not_vac=find_probability_icu(data, start_date, end_date,list_p_in_uci,list_n_in_uci).tolist()
    
    prob_uci_vac1,prob_uci_vac2=genere_list_minsal_denis(np.array(probability_not_vac),pt_factor=[0.16,0.636],minsal_factor=[0.43,0.90])
    

    probability_not_vac_b117=probability_not_vac.copy()
    aplha=2.03
    #aplha=1
    probability_not_vac_b117=(np.array(probability_not_vac_b117)*aplha).tolist()
    prob_uci_vac1_b117,prob_uci_vac2_b117=genere_list_minsal_denis(np.array(probability_not_vac_b117),pt_factor=[0.16,0.636],minsal_factor=[0.43,0.90])
    
    probability_not_vac_new_variant=[0.011]
    prob_uci_vac1_new_variant,prob_uci_vac2_new_variant=genere_list_minsal_denis(np.array(probability_not_vac_new_variant),pt_factor=[0.16,0.636],minsal_factor=[0.43,0.90])
    
    shift=True
    day_change_time='2021-01-01'
    ID_day_change_time=data['date_to_dateID'][np.datetime64(day_change_time+"T00:00:00.000000000")]
    uci, dead,dict_main_dead=simulation_arg_v1(data,
                      increase_vac1,increase_vac2,
                      prob_uci_vac1,prob_uci_vac2,
                      prob_uci_vac1_b117,prob_uci_vac2_b117,
                      prob_uci_vac1_new_variant,prob_uci_vac2_new_variant,
                      p_shares_new_variant=[[0.        , 0.        , 0.01436602, 0.03082466, 0.05447681,
                             0.10934183, 0.15449835, 0.26977315, 0.32454422, 0.458103  ,
                             0.55592204, 0.5851679 , 0.64809384, 0.67343322]],
                      ID_list_day_new_variant=[279, 293, 314, 328, 342, 356, 370, 384, 398, 412, 426, 440, 454, 468],
                      p_shares_b117=[[0.        , 0.01226226, 0.        , 0.07686149, 0.        ,
                             0.07271762, 0.13565709, 0.19098712, 0.13575411, 0.14518475,
                             0.12053973, 0.1083494 , 0.08736559, 0.08860963]],
                      ID_list_day_b117=[279, 293, 314, 328, 342, 356, 370, 384, 398, 412, 426, 440, 454, 468],
                      window=29, 
                      probability_not_vac=probability_not_vac,
                      probability_not_vac_b117=probability_not_vac_b117,
                      probability_not_vac_new_variant=probability_not_vac_new_variant,
                      list_p_to_uci=[0.463],
                      list_n_to_uci= [8.8889],
                      list_p_to_uci_2021=[0.699],
                      list_n_to_uci_2021= [21.798],
                      ID_day_change_time=ID_day_change_time,
                      list_p_in_uci=[0.0699],
                      list_n_in_uci=[1.9495],
                      max_days_go_uci=30,
                      max_days_in_uci= 100, 
                      window_slide_to_uci=0,
                      shift=shift)
    
    plot_uci_pred_sns(data, uci, W=29, pond=1, start_date='2020-07-01', end_date='2021-01-01', infected=False)
    plot_uci_pred_sns(data, uci, W=29, pond=1, start_date='2020-07-01', end_date='2021-05-01', infected=False)
    plot_uci_pred_sns(data, uci, W=29, pond=1, start_date='2020-07-01', end_date='2021-12-01', infected=False)

    

