#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 14:22:58 2022

@author: ineira
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
path_ETL='Code/ETL'
path_icu_simulation='Code/icu_simulation'

module_path=path_projet+path_ETL
if module_path not in sys.path:
    sys.path.append(module_path)
  
module_path=path_projet+path_icu_simulation
if module_path not in sys.path:
    sys.path.append(module_path)

from data_processing import read_data,prepare_population
from simulation_v9 import split_infed_new_variant,porc_infected_by_vac,porc_days_to_uci_by_vac,porc_uci_by_vac,porc_days_in_uci_by_vac,porc_infected_and_uci_not_vac
import seaborn as sns ## These Three lines are necessary for Seaborn to work   
import matplotlib.pyplot as plt 
from matplotlib.dates import date2num
import matplotlib.dates as mdates

def simulation_dead_v9(data,
                  increase_vac1_Pfizer,increase_vac2_Pfizer,
                  increase_vac1_Sinovac,increase_vac2_Sinovac, 
                  prob_uci_vac1_Pfizer,prob_uci_vac2_Pfizer,
                  prob_uci_vac1_Sinovac,prob_uci_vac2_Sinovac,
                  prob_uci_vac1_b117_Pfizer,prob_uci_vac2_b117_Pfizer,
                  prob_uci_vac1_b117_Sinovac,prob_uci_vac2_b117_Sinovac,
                  prob_uci_vac1_new_variant_Pfizer,prob_uci_vac2_new_variant_Pfizer,
                  prob_uci_vac1_new_variant_Sinovac,prob_uci_vac2_new_variant_Sinovac,
                  p_shares_new_variant=[[0.0, 0.0, 0.269, 0.5, 0.739, 0.643],
                                        [0.0, 0.0, 0.357, 0.517, 0.522, 0.833],
                                        [0.0, 0.0, 0.462, 0.55, 0.818, 0.818],
                                        [0.0, 0.057, 0.418, 0.465, 0.636, 0.62],
                                        [0.0, 0.0, 0.333, 0.333, 0.571, 0.75]],
                  ID_list_day_new_variant=[260,275,289,320,348,379 ],
                  p_shares_b117=[[0.0, 0.062, 0.115, 0.096, 0.0, 0.0],
                                [0.0, 0.0, 0.0, 0.069, 0.0, 0.0],
                                [0.0, 0.0, 0.308, 0.1, 0.0, 0.0],
                                [0.0, 0.057, 0.114, 0.088, 0.029, 0.014],
                                [0.0, 0.0, 0.0, 0.167, 0.0, 0.0]],
                  ID_list_day_b117=[260,275,289,320,348,379 ],
                  window=29, 
                  probability_not_vac=[0.024, 0.0638, 0.15, 0.007125, 0.17],
                  probability_not_vac_b117=[0.024, 0.0638, 0.15, 0.007125, 0.17],
                  probability_not_vac_new_variant=[0.024, 0.0638, 0.15, 0.007125, 0.17],
                  list_p_to_uci=[0.463, 0.2139, 0.1494, 0.3976, 0.2466],
                  list_n_to_uci= [8.8889, 2.6667, 2.2222, 5.1111, 3.6667],
                  list_p_to_uci_2021=[0.699, 0.402, 0.365, 0.508, 0.7  ],
                  list_n_to_uci_2021= [21.798 ,  6.2727,  6.5657,  9.202 , 22.0909],
                  ID_day_change_time=275,
                  list_p_in_uci=[0.0699, 0.0619, 0.0515, 0.0739, 0.0412],
                  list_n_in_uci=[1.9495, 1.4646, 1.4444, 2.0101, 1.303],
                  max_days_go_uci=30,
                  max_days_in_uci= 100, 
                  window_slide_to_uci=0,
                  shift=False):
    start_time = time.time()
    
    lab_to_labID=data['lab_to_labID']
    groupID_to_group = data["groupID_to_group"]
    date_to_dateID = data["date_to_dateID"]
    
    L= len(lab_to_labID)
    D = 2
    G = len(groupID_to_group)
    T = len(date_to_dateID)
    
    #split case by variant
    split_infed_new_variant(data,p_shares_new_variant,ID_list_day_new_variant,p_shares_b117,ID_list_day_b117,shift=shift)
    
    #----------------- vacccine-----------
    ## estatic
    W=window
    increase_vac1_Pfizer=np.array(increase_vac1_Pfizer)
    increase_vac2_Pfizer=np.array(increase_vac2_Pfizer)
    
    increase_vac1_Sinovac=np.array(increase_vac1_Sinovac)
    increase_vac2_Sinovac=np.array(increase_vac2_Sinovac)
    
    ## (L,D,G,T-W+1,W)
    porc_infected = porc_infected_by_vac(data,L,D,G,T,
                                         increase_vac1_Pfizer,increase_vac2_Pfizer,
                                         increase_vac1_Sinovac,increase_vac2_Sinovac,
                                         window= W,
                                         new_variant=False)
    
    porc_infected_new_variant = porc_infected_by_vac(data,L,D,G,T,
                                         increase_vac1_Pfizer,increase_vac2_Pfizer,
                                         increase_vac1_Sinovac,increase_vac2_Sinovac,
                                         window= W,
                                         new_variant=True)
    porc_infected_b117 = porc_infected_by_vac(data,L,D,G,T,
                                         increase_vac1_Pfizer,increase_vac2_Pfizer,
                                         increase_vac1_Sinovac,increase_vac2_Sinovac,
                                         window= W,
                                         new_variant=True,b117=True)
    
    dateID_firt_dosis=data['dateID_firt_dosis']-W+1
    
    
    
    porc_uci1 = porc_uci_by_vac(porc_infected[0],D,G,T,W, 
                               prob_uci_vac1=prob_uci_vac1_Pfizer,
                               prob_uci_vac2=prob_uci_vac2_Pfizer)
    porc_uci2 = porc_uci_by_vac(porc_infected[1],D,G,T,W, 
                               prob_uci_vac1=prob_uci_vac1_Sinovac,
                               prob_uci_vac2=prob_uci_vac2_Sinovac)
    
    porc_uci1_new_variant = porc_uci_by_vac(porc_infected_new_variant[0],D,G,T,W, 
                               prob_uci_vac1=prob_uci_vac1_new_variant_Pfizer,
                               prob_uci_vac2=prob_uci_vac2_new_variant_Pfizer)
    porc_uci2_new_variant = porc_uci_by_vac(porc_infected_new_variant[1],D,G,T,W, 
                               prob_uci_vac1=prob_uci_vac1_new_variant_Sinovac,
                               prob_uci_vac2=prob_uci_vac2_new_variant_Sinovac)
    
    porc_uci1_b117  = porc_uci_by_vac(porc_infected_b117[0],D,G,T,W, 
                               prob_uci_vac1=prob_uci_vac1_b117_Pfizer,
                               prob_uci_vac2=prob_uci_vac2_b117_Pfizer)
    porc_uci2_b117  = porc_uci_by_vac(porc_infected_b117[1],D,G,T,W, 
                               prob_uci_vac1=prob_uci_vac1_b117_Sinovac,
                               prob_uci_vac2=prob_uci_vac2_b117_Sinovac)
    
    
    porc_uci_final=porc_uci1+porc_uci2+porc_uci1_new_variant+porc_uci2_new_variant+porc_uci1_b117+porc_uci2_b117
    
    dateID_day_change_time=ID_day_change_time-W+1
    #depends assigned probability
    porc_days_to_uci_2020=porc_days_to_uci_by_vac(porc_uci_final[:,:,:dateID_day_change_time,:],
                                             D,G,W,
                                             list_p_to_uci=list_p_to_uci,
                                             list_n_to_uci=list_n_to_uci,
                                             max_days_go_uci= max_days_go_uci,
                                             window_slide=window_slide_to_uci)
    
    porc_days_to_uci_2021=porc_days_to_uci_by_vac(porc_uci_final[:,:,dateID_day_change_time:,:],
                                             D,G,W,
                                             list_p_to_uci=list_p_to_uci_2021,
                                             list_n_to_uci=list_n_to_uci_2021,
                                             max_days_go_uci= max_days_go_uci,
                                             window_slide=window_slide_to_uci)
    
    
    
    porc_days_to_uci=np.concatenate((porc_days_to_uci_2020,porc_days_to_uci_2021), axis=-2)
    
    porc_days_in_uci=porc_days_in_uci_by_vac(porc_days_to_uci,G,
                                             max_days_in_uci= max_days_in_uci,
                                             list_p_in_uci=list_p_in_uci,
                                             list_n_in_uci=list_n_in_uci)
     #----------------- not vaccine -----------
    
    porc_uci_not=porc_infected_and_uci_not_vac(data,D,G,T,W, 
                                               probability=probability_not_vac)
    porc_uci_not_new_variant=porc_infected_and_uci_not_vac(data,D,G,T,W, 
                                               probability=probability_not_vac_new_variant,
                                               new_variant=True)
    porc_uci_not_b117=porc_infected_and_uci_not_vac(data,D,G,T,W, 
                                               probability=probability_not_vac_b117,
                                               new_variant=True,b117=True)
    data['porc_uci_not']=porc_uci_not.squeeze()
    data['porc_uci_not_new_variant']=porc_uci_not_new_variant.squeeze()
    data['porc_uci_not_b117']=porc_uci_not_b117.squeeze()
    porc_uci_not_final=porc_uci_not+porc_uci_not_new_variant+porc_uci_not_b117
    
    
    
     
    porc_days_to_uci_not_vac_2020=porc_days_to_uci_by_vac(porc_uci_not_final[:,:,:dateID_day_change_time,:],D,G,W,
                                                     list_p_to_uci=list_p_to_uci,
                                                     list_n_to_uci=list_n_to_uci,
                                                     max_days_go_uci= max_days_go_uci,
                                                     window_slide=window_slide_to_uci)
    
    porc_days_to_uci_not_vac_2021=porc_days_to_uci_by_vac(porc_uci_not_final[:,:,dateID_day_change_time:,:],D,G,W,
                                                     list_p_to_uci=list_p_to_uci_2021,
                                                     list_n_to_uci=list_n_to_uci_2021,
                                                     max_days_go_uci= max_days_go_uci,
                                                     window_slide=window_slide_to_uci)
    porc_days_to_uci_not_vac=np.concatenate((porc_days_to_uci_not_vac_2020,porc_days_to_uci_not_vac_2021), axis=-2)
    
    porc_days_in_uci_not_vac=porc_days_in_uci_by_vac(porc_days_to_uci_not_vac,G,
                                                     max_days_in_uci= max_days_in_uci,
                                                     list_p_in_uci=list_p_in_uci,
                                                     list_n_in_uci=list_n_in_uci)

    
    
    #np.sum(porc_days_in_uci,axis=(2,-1))
    K,H,D,G,T_W_1,W = porc_days_in_uci.shape
    T_W_1=porc_days_in_uci_not_vac.shape[4]
    print(T_W_1)
    print(porc_days_in_uci.shape)
    aux_vac=np.sum(porc_days_in_uci,axis=(2,-1))
    aux_not_vac=np.sum(porc_days_in_uci_not_vac,axis=(2,-1))
    
    
    aux=np.add(aux_vac,aux_not_vac)
    
    uci = np.zeros((G,T_W_1+K+H))
    for t in range(T_W_1):
        for h in range(H):
            for k in range(1,K):
                uci[:,t+h:t+h+k]+=(np.expand_dims(aux[k,h,:,t],axis=(-1)))
                
                
    dead=np.zeros((G,T_W_1+K+H))
    for t in range(T_W_1):
        for h in range(H):
            for k in range(1,K):
                dead[:,t+h+k]+=aux[k,h,:,t]
                
                
    print(np.isnan(uci).any())
    #uci_share_new_variant=uci_share_new_variant/uci
    end_time = round(time.time()-start_time,4)
    #print("Producto 9 is ready")
    print("Execution time:" + str(end_time) +" s.")
    print("Simulation is  ready")
    print("="*40)
    return uci,dead#,uci_share_new_variant