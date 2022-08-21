"""
Created on Mon Mar 29 18:16:33 2021

@author: ignas


Nota. 
Ahora los tiempo es uci distribuyen geometrico
y pueden variar en el tiempo




['40-49', '50-59', '60-69', '<=39', '>=70']

[11.0, 10.0, 13.0, 8.0, 11.0]#
[8, 7, 8, 7, 4]#'T. CONFIRMACION PCR desde INICIO DE LOS SINTOMAS'
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
from tqdm import tqdm



path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
path_ETL='Code/ETL'

module_path=path_projet+path_ETL
if module_path not in sys.path:
    sys.path.append(module_path)
from data_processing import read_data,prepare_population

from fit_probability_to_icu import find_probability_icu



import seaborn as sns ## These Three lines are necessary for Seaborn to work   
import matplotlib.pyplot as plt 
from matplotlib.dates import date2num
import matplotlib.dates as mdates

plt.figure(dpi=1200)
#data = read_data()


def simulation_v9(data,
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
    
    L= len(lab_to_labID) #Nª of labs
    D = 2 # Nª dosis
    G = len(groupID_to_group) #Nª of age group
    T = len(date_to_dateID) # Nª of days 
    
    #split case by variant , the output is in data
    split_infed_new_variant(data,p_shares_new_variant,ID_list_day_new_variant,p_shares_b117,ID_list_day_b117,shift=shift)
    
    #----------------- vacccine-----------
    ## estatic
    W=window #is the number of date beteewn firts dosis and second dosis
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
    
    
    
    """
    New problem: we want to identify those leaving the ICU by variant.

    remember that our model assumes that all patients in the ICU behave the same way. 
    """
    porc_uci_not_variant=porc_uci1+porc_uci2
    porc_uci_new_variant=porc_uci1_new_variant+porc_uci2_new_variant
    porc_uci_b117=porc_uci1_b117+porc_uci2_b117
    
    dict_porc_uci_vacc={
        'Not variant': porc_uci_not_variant,
        'variant':porc_uci_new_variant,
        'b117':porc_uci_b117
        }
    
    print("Not vacc")
    dict_porc_in_icu_vacc={}
    for key, value in tqdm(dict_porc_uci_vacc.items(),desc="Progress"):
        print(key)
        porc_days_to_uci_2020_key=porc_days_to_uci_by_vac(value[:,:,:dateID_day_change_time,:],
                                                 D,G,W,
                                                 list_p_to_uci=list_p_to_uci,
                                                 list_n_to_uci=list_n_to_uci,
                                                 max_days_go_uci= max_days_go_uci,
                                                 window_slide=window_slide_to_uci)
        
        porc_days_to_uci_2021_key=porc_days_to_uci_by_vac(value[:,:,dateID_day_change_time:,:],
                                                 D,G,W,
                                                 list_p_to_uci=list_p_to_uci_2021,
                                                 list_n_to_uci=list_n_to_uci_2021,
                                                 max_days_go_uci= max_days_go_uci,
                                                 window_slide=window_slide_to_uci)
        porc_days_to_uci_key=np.concatenate((porc_days_to_uci_2020_key,porc_days_to_uci_2021_key), axis=-2)
        
        porc_days_in_uci_key=porc_days_in_uci_by_vac(porc_days_to_uci_key,G,
                                                 max_days_in_uci= max_days_in_uci,
                                                 list_p_in_uci=list_p_in_uci,
                                                 list_n_in_uci=list_n_in_uci)
        dict_porc_in_icu_vacc[key]=porc_days_in_uci_key
        
    dict_porc_uci_not_vacc={
        'Not variant': porc_uci_not,
        'variant':porc_uci_not_new_variant,
        'b117':porc_uci_not_b117,
        }
    
    dict_porc_in_icu_not_vacc={}
    print("Not vacc")
    for key, value in tqdm(dict_porc_uci_not_vacc.items(),desc="Progress"):
        print(key)
        porc_days_to_uci_not_vac_2020_key=porc_days_to_uci_by_vac(value[:,:,:dateID_day_change_time,:],D,G,W,
                                                         list_p_to_uci=list_p_to_uci,
                                                         list_n_to_uci=list_n_to_uci,
                                                         max_days_go_uci= max_days_go_uci,
                                                         window_slide=window_slide_to_uci)
        
        porc_days_to_uci_not_vac_2021_key=porc_days_to_uci_by_vac(value[:,:,dateID_day_change_time:,:],D,G,W,
                                                         list_p_to_uci=list_p_to_uci_2021,
                                                         list_n_to_uci=list_n_to_uci_2021,
                                                         max_days_go_uci= max_days_go_uci,
                                                         window_slide=window_slide_to_uci)
        porc_days_to_uci_not_vac_key=np.concatenate((porc_days_to_uci_not_vac_2020_key,
                                                 porc_days_to_uci_not_vac_2021_key), axis=-2)
        
        porc_days_in_uci_not_vac_key=porc_days_in_uci_by_vac(porc_days_to_uci_not_vac_key,G,
                                                         max_days_in_uci= max_days_in_uci,
                                                         list_p_in_uci=list_p_in_uci,
                                                         list_n_in_uci=list_n_in_uci)
        dict_porc_in_icu_not_vacc[key]=porc_days_in_uci_not_vac_key
    
    
    
    #np.sum(porc_days_in_uci,axis=(2,-1))
    
    K,H,D,G,T_W_1,W = porc_days_in_uci.shape
    T_W_1=porc_days_in_uci_not_vac.shape[4]
    print(T_W_1)
    print(porc_days_in_uci.shape)
    aux_vac=np.sum(porc_days_in_uci,axis=(2,-1))
    aux_not_vac=np.sum(porc_days_in_uci_not_vac,axis=(2,-1))
    
    
    aux=np.add(aux_vac,aux_not_vac)
    
    uci = np.zeros((G,T_W_1+K+H))
    for t in tqdm(range(T_W_1)):
        for h in range(H):
            for k in range(1,K):
                uci[:,t+h:t+h+k]+=(np.expand_dims(aux[k,h,:,t],axis=(-1)))
    dead=np.zeros((G,T_W_1+K+H))
    for t in range(T_W_1):
        for h in range(H):
            for k in range(1,K):
                dead[:,t+h+k]+=aux[k,h,:,t]
                
    #print(np.isnan(uci).any())
    #uci_share_new_variant=uci_share_new_variant/uci
    print("="*40)
    print("Generete data for fatality by variant")
    dict_icu_variant={
        'total':uci
        }
    dict_dead_variant={
        'total':dead
        }
    for key, value in dict_porc_in_icu_not_vacc.items():
        aux_vac=np.sum(dict_porc_in_icu_vacc[key],axis=(2,-1))
        
        aux_not_vac=np.sum(value,axis=(2,-1))
        aux_key=np.add(aux_vac,aux_not_vac)
        uci_key = np.zeros((G,T_W_1+K+H))
        for t in tqdm(range(T_W_1)):
            for h in range(H):
                for k in range(1,K):
                    uci_key[:,t+h:t+h+k]+=(np.expand_dims(aux_key[k,h,:,t],axis=(-1)))
        
        dead_key=np.zeros((G,T_W_1+K+H))
        for t in range(T_W_1):
            for h in range(H):
                for k in range(1,K):
                    dead_key[:,t+h+k]+=aux_key[k,h,:,t]
        
        dict_icu_variant[key]=uci_key
        dict_dead_variant[key]=dead_key
        
    dict_main_dead={
        'dict_icu_variant':dict_icu_variant,
        'dict_dead_variant':dict_dead_variant
        }
    
    end_time = round(time.time()-start_time,4)
    
    print("  ")
    print("Execution time:" + str(end_time) +" s.")
    print("Simulation is  ready")
    print("="*40)
    return uci, dead,dict_main_dead#,uci_share_new_variant


def simulation_v9_fase_1(data,
                  increase_vac1_Pfizer,increase_vac2_Pfizer,
                  increase_vac1_Sinovac,increase_vac2_Sinovac,
                  prob_uci_vac1_Pfizer,prob_uci_vac2_Pfizer,
                  prob_uci_vac1_Sinovac,prob_uci_vac2_Sinovac,
                  window=29, 
                  probability_not_vac=[0.024, 0.0638, 0.15, 0.007125, 0.17],
                  list_p_to_uci=[0.463, 0.2139, 0.1494, 0.3976, 0.2466],
                  list_n_to_uci= [8.8889, 2.6667, 2.2222, 5.1111, 3.6667],
                  list_p_to_uci_2021=[0.699, 0.402, 0.365, 0.508, 0.7  ],
                  list_n_to_uci_2021= [21.798 ,  6.2727,  6.5657,  9.202 , 22.0909],
                  ID_day_change_time=275,
                  list_p_in_uci=[0.0699, 0.0619, 0.0515, 0.0739, 0.0412],
                  list_n_in_uci=[1.9495, 1.4646, 1.4444, 2.0101, 1.303],
                  max_days_go_uci=30,
                  max_days_in_uci= 100,
                  window_slide_to_uci=0,shift=False):
    start_time = time.time()
    probability_new_variant=[[0,0,0]]*5
    ID_list_day_new_variant=[247,275,337]
    lab_to_labID=data['lab_to_labID']
    groupID_to_group = data["groupID_to_group"]
    date_to_dateID = data["date_to_dateID"]
    
    L= len(lab_to_labID)
    D = 2
    G = len(groupID_to_group)
    T = len(date_to_dateID)
    
    #split case by variant
    split_infed_new_variant(data,probability_new_variant,ID_list_day_new_variant=ID_list_day_new_variant,probability_b117=probability_new_variant,ID_list_day_b117=ID_list_day_new_variant,shift=shift)
    
    #----------------- vac not variant-----------
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
 
    
    porc_uci1 = porc_uci_by_vac(porc_infected[0],D,G,T,W, 
                               prob_uci_vac1=prob_uci_vac1_Pfizer,
                               prob_uci_vac2=prob_uci_vac2_Pfizer)
    porc_uci2 = porc_uci_by_vac(porc_infected[1],D,G,T,W, 
                               prob_uci_vac1=prob_uci_vac1_Sinovac,
                               prob_uci_vac2=prob_uci_vac2_Sinovac)
    porc_uci_final=np.add(porc_uci1,porc_uci2)
    #depends assigned probability
    
    dateID_day_change_time=ID_day_change_time-W+1
    
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
    
    
    porc_days_to_uci=porc_days_to_uci_by_vac(porc_uci_final,D,G,W,
                                             list_p_to_uci=list_p_to_uci,
                                             list_n_to_uci=list_n_to_uci,
                                             max_days_go_uci= max_days_go_uci,
                                             window_slide=window_slide_to_uci)
    
    porc_days_in_uci=porc_days_in_uci_by_vac(porc_days_to_uci,G,
                                             max_days_in_uci= max_days_in_uci,
                                             list_p_in_uci=list_p_in_uci,
                                             list_n_in_uci=list_n_in_uci)
    
    
    #----------------- not vac  not variant -----------
    
    porc_uci_not=porc_infected_and_uci_not_vac(data,D,G,T,W, 
                                               probability=probability_not_vac)
    
    
    porc_days_to_uci_not_vac_2020=porc_days_to_uci_by_vac(porc_uci_not[:,:,:dateID_day_change_time,:],D,G,W,
                                                     list_p_to_uci=list_p_to_uci,
                                                     list_n_to_uci=list_n_to_uci,
                                                     max_days_go_uci= max_days_go_uci,
                                                     window_slide=window_slide_to_uci)
    
    porc_days_to_uci_not_vac_2021=porc_days_to_uci_by_vac(porc_uci_not[:,:,dateID_day_change_time:,:],D,G,W,
                                                     list_p_to_uci=list_p_to_uci_2021,
                                                     list_n_to_uci=list_n_to_uci_2021,
                                                     max_days_go_uci= max_days_go_uci,
                                                     window_slide=window_slide_to_uci)
    porc_days_to_uci_not_vac=np.concatenate((porc_days_to_uci_not_vac_2020,porc_days_to_uci_not_vac_2021), axis=-2)
    
    porc_days_to_uci_not_vac=porc_days_to_uci_by_vac(porc_uci_not,D,G,W,
                                                     list_p_to_uci=list_p_to_uci,
                                                     list_n_to_uci=list_n_to_uci,
                                                     max_days_go_uci= max_days_go_uci,
                                                     window_slide=window_slide_to_uci)
    porc_days_in_uci_not_vac=porc_days_in_uci_by_vac(porc_days_to_uci_not_vac,G,
                                                     max_days_in_uci= max_days_in_uci,
                                                     list_p_in_uci=list_p_in_uci,
                                                     list_n_in_uci=list_n_in_uci)
    
    
    
    #np.sum(porc_days_in_uci,axis=(2,-1))
    K,H,D,G,T_W_1,W = porc_days_in_uci.shape
    T_W_1=porc_days_in_uci_not_vac.shape[4]
    print(T_W_1)
    aux_vac=np.sum(porc_days_in_uci,axis=(2,-1))
    aux_not_vac=np.sum(porc_days_in_uci_not_vac,axis=(2,-1))
    aux=np.add(aux_vac,aux_not_vac)
    
    uci = np.zeros((G,T_W_1+K+H))
    for t in range(T_W_1):
        for h in range(H):
            for k in range(1,K):
            #for t in range(porc_days_in_uci.shape[0]):
                uci[:,t+h:t+h+k]+=(np.expand_dims(aux[k,h,:,t],axis=(-1)))
                
    end_time = round(time.time()-start_time,4)
    #print("Producto 9 is ready")
    print("Execution time:" + str(end_time) +" s.")
    print("Simulation is  ready")
    print("="*40)
    return uci


def porc_infected_by_vac(data,L,D,G,T,
                         increase_vac1_Pfizer,increase_vac2_Pfizer,
                         increase_vac1_Sinovac,increase_vac2_Sinovac, 
                         window=100, new_variant=False,b117=False):
    """
    Porbabilidad que el sintomatico de ese dia se vacuno hace w dias hacia atras
    
    
    data['inf'] : (G,T)-> (G,T-(W-1),1)
    
    sliding_window data['vac'] : (L,D,G,T)-> (D,G,T-(W-1),W)
    data['acc_vac'] : (D,G,T)
    -----------
    return
        (l,D,G,T-(W-1),W)
    """

    sl_window = sliding_window_view(data['vac'],window, axis=-1).copy()
    #suppose the 2do dosis accumalte all
    sl_window[:,1,:,:,0]=data['vac_acc'][:,1,:,:T-(window-1)]
    #chage the las value t-w for acc vac in t-w
    """
    sl_window[:,:,:,:,0]=data['vac_acc'][:,:,:,:T-(window-1)]
    
    #chage number of 1 dosis depending on the 2do dosis
      
    dateID_second_dosis = data['dateID_second_dosis']
    for lab in range(L):
        for group in range(G):
            for t in range(dateID_second_dosis,T):
                replace_value_for_1_vacc(data['vac_acc'][lab,1,group,t], sl_window[lab,0,group,t-(window-1),:], i=0)
                
    """
    window_porc_vac_pop = np.multiply(sl_window,1/np.expand_dims(data['pop'],axis=(0,1,-2,-1)))
    
    
    
    ###---------------------------ponderar infectado------
    day_step=np.array([7,7,7,7,window-28])

    
    #GRILLA ESCALONADA
    grilla_vac1_Pfizer=[
            np.concatenate([
                np.linspace(increase_vac1_Pfizer[j,i],increase_vac1_Pfizer[j,i],day_step[i],endpoint=False) for i in range(increase_vac1_Pfizer.shape[1]-1)],
                axis=None).tolist()[::-1] for j in range(increase_vac1_Pfizer.shape[0])]
    grilla_vac2_Pfizer=[
            np.concatenate([
                np.linspace(increase_vac2_Pfizer[j,i],increase_vac2_Pfizer[j,i],day_step[i],endpoint=False) for i in range(increase_vac2_Pfizer.shape[1]-1)],
                axis=None).tolist()[::-1] for j in range(increase_vac2_Pfizer.shape[0])]
    
    grilla_vac1_Sinovac=[
            np.concatenate([
                np.linspace(increase_vac1_Sinovac[j,i],increase_vac1_Sinovac[j,i],day_step[i],endpoint=False) for i in range(increase_vac1_Sinovac.shape[1]-1)],
                axis=None).tolist()[::-1] for j in range(increase_vac1_Sinovac.shape[0])]
    grilla_vac2_Sinovac=[
            np.concatenate([
                np.linspace(increase_vac2_Sinovac[j,i],increase_vac2_Sinovac[j,i],day_step[i],endpoint=False) for i in range(increase_vac2_Sinovac.shape[1]-1)],
                axis=None).tolist()[::-1] for j in range(increase_vac2_Sinovac.shape[0])]
    
    #dado que W=0 es el t-W se deven dar vuelta los valores
    grilla = np.array([[grilla_vac1_Pfizer,grilla_vac2_Pfizer],[grilla_vac1_Sinovac,grilla_vac2_Sinovac]]) #(L,D,G,W)
    
    numerador= np.multiply(window_porc_vac_pop,np.expand_dims(1-grilla,axis=(-2)))#(L,D,G,T-(W-1),W)
    #sum over lab,dosis and W
    denominador= np.sum(numerador,axis=(0,1,-1))#(G,T-(W-1))
    
    #call porc not vac
    porc_not_vac_pop = np.multiply(data['not_vac'],1/np.expand_dims(data['pop'],axis=(-1)))[:,window-1:]#(G,T-(W-1))
    denominador = np.add(denominador, porc_not_vac_pop)#(G,T-(W-1))
    
    
    ##(L,D,G,T-(W-1),W) + (G,T-(W-1),1)-->(L,D,G,T-(W-1),W)
    contagiados_sintomaticos = np.multiply(numerador, np.expand_dims(1/denominador,axis=(-1)))
    
    if new_variant:
        
        if b117:
            #save the values in the dict
            data['denominador_b117']= denominador
            
            
            return np.multiply(
                np.expand_dims(data['inf_b117'][:,window-1:],axis=(-1)),
                contagiados_sintomaticos)
            
        else:
             #save the values in the dict
            data['denominador_new_variant']= denominador
            
            
            return np.multiply(
                np.expand_dims(data['inf_new_variant'][:,window-1:],axis=(-1)),
                contagiados_sintomaticos)
    
    
    else:
        #save the values in the dict
        data['denominador_not_variant']= denominador
        
        
        return np.multiply(
            np.expand_dims(data['inf_no_variant'][:,window-1:],axis=(-1)),
            contagiados_sintomaticos)




def replace_value_for_1_vacc(vac_acc_2jt, split_array, i=0):
    
    """
    
    
    """
    diff = split_array[i]-vac_acc_2jt
    if diff >=0:
        split_array[i]=diff
        pass
    
    else:
        split_array[i]=0
        replace_value_for_1_vacc(abs(diff), split_array, i=i+1)
        
        
def porc_infected_and_uci_not_vac(data,D,G,T, window=100, probability=[0.05,0.05,0.05,0.05,0.05], for_train=False, matrix_probability=np.nan,end_time=-1, new_variant=False,b117=False):
    """
    Caso para un 1-D list of probability 
        Porbabilidad que el sintomatico de ese dia no se vacuno y vaya a uci
        data['inf'] : (G,T)
        data['not_vac'] : (G,T)
        probability : (G,)
        -----------
        return
            (G,T-(W-1))---->(1,,G,T-(W-1),1)
            
    Caso para un 2-D list of probability 
        Porbabilidad que el sintomatico de ese dia no se vacuno y vaya a uci
        data['inf'] : (G,T)---w-1: firt day of vacc
        data['not_vac'] : (G,T)
        probability : (X,G)  
        -----------
        return
            (G,T-(W-1))---->(X,1,G,T*,1)
    """
    if for_train:
        if new_variant:
            pass
        
        else:
            porc_not_vac_pop = np.multiply(data['not_vac'],1/np.expand_dims(data['pop'],axis=(-1)))
            
        
            return np.expand_dims(np.multiply(
                np.multiply(
                data['inf_no_variant'],
                porc_not_vac_pop)[:,window-1:end_time],
                np.expand_dims(matrix_probability,axis=(-1))),axis=(1,-1))

    else:
        if new_variant:
            
            if b117:
                print("UK check!")
                print(probability)
                denominador=data['denominador_b117']
                porc_not_vac_pop = np.multiply(data['not_vac'],1/np.expand_dims(data['pop'],axis=(-1)))
                
            
                return np.expand_dims(
                    np.multiply(
                        np.multiply(
                            np.multiply(data['inf_b117'],porc_not_vac_pop)[:,window-1:],
                            1/denominador),
                        np.expand_dims(probability,axis=(-1))),axis=(0,-1))
        
            else:
                print("P1 check!")
                print(probability)
                denominador=data['denominador_new_variant']
                porc_not_vac_pop = np.multiply(data['not_vac'],1/np.expand_dims(data['pop'],axis=(-1)))
                
            
                return np.expand_dims(
                    np.multiply(
                        np.multiply(
                            np.multiply(data['inf_new_variant'],porc_not_vac_pop)[:,window-1:],
                            1/denominador),
                        np.expand_dims(probability,axis=(-1))),axis=(0,-1))

            
        
        else:
            print("Not variant check!")
            print(probability)
            denominador=data['denominador_not_variant']
            porc_not_vac_pop = np.multiply(data['not_vac'],1/np.expand_dims(data['pop'],axis=(-1)))
            
        
            return np.expand_dims(
                np.multiply(
                    np.multiply(
                        np.multiply(data['inf_no_variant'],porc_not_vac_pop)[:,window-1:],
                        1/denominador),
                    np.expand_dims(probability,axis=(-1))),axis=(0,-1))




def porc_uci_by_vac(porc_infected,D,G,T,W,prob_uci_vac1=np.nan, prob_uci_vac2=np.nan,for_train_vac=False,matrix_prob_uci_vac1=np.nan,matrix_prob_uci_vac2=np.nan):
    """
    Caso para un 2-D list of probability 
        porc_infected : (D,G,T-(W-1),W)
        porc uci: (D,G,W)->(D,G,1,W)
        -----------
        return
            (D,G,T-(W-1),W)
            
    caso para un 3 D probabolity
            porc_infected : (D,G,T-(W-1),W)
            porc uci: (D,Q,G,W)->(Q,D,G,1,W)
            
            return 
                (Q,D,G,T-(W-1),W)
    """
    if for_train_vac:
        """ 
        prob_uci_vac1, prob_uci_vac2 convert np.array y have the same dimensions
        """
        axis_0 =matrix_prob_uci_vac1.shape[0]
        axis_1 =matrix_prob_uci_vac1.shape[1]
        axis_2 =matrix_prob_uci_vac1.shape[2]
        day_step=np.array([7,7,7,7,W-28])
        day_step=np.tile(day_step,(G,1))
        
        grilla_vac1=[[
                np.concatenate([
                    np.linspace(matrix_prob_uci_vac1[k,j,i],matrix_prob_uci_vac1[k,j,i+1],day_step[j,i],endpoint=False) for i in range(axis_2-1)],
                    axis=None).tolist() for j in range(axis_1)] for k in range(axis_0)]
        grilla_vac2=[[
                np.concatenate([
                    np.linspace(matrix_prob_uci_vac2[k,j,i],matrix_prob_uci_vac2[k,j,i+1],day_step[j,i],endpoint=False) for i in range(axis_2-1)],
                    axis=None).tolist() for j in range(axis_1)] for k in range(axis_0)]
        
        grilla = np.array([grilla_vac1,grilla_vac1]).transpose([1,0,2,3])
        grilla = np.expand_dims(grilla,axis=(-2))
        
        return np.multiply(porc_infected,grilla)
    
    else:
        
        day_step=np.array([7,7,7,7,W-28])
        day_step=np.tile(day_step,(G,1))
        prob_uci_vac1= np.array(prob_uci_vac1)
        prob_uci_vac2= np.array(prob_uci_vac2)
        "dar vuelta la lista de probabilidades, ya que W esta escrito al reves"
        grilla_vac1=[
                np.concatenate([
                    np.linspace(prob_uci_vac1[j,i],prob_uci_vac1[j,i],day_step[j,i],endpoint=False) for i in range(prob_uci_vac1.shape[1]-1)],
                    axis=None).tolist()[::-1] for j in range(prob_uci_vac1.shape[0])]
        grilla_vac2=[
                np.concatenate([
                    np.linspace(prob_uci_vac2[j,i],prob_uci_vac2[j,i],day_step[j,i],endpoint=False) for i in range(prob_uci_vac2.shape[1]-1)],
                    axis=None).tolist()[::-1] for j in range(prob_uci_vac2.shape[0])]
        
        #dado que W=0 es el t-W se deven dar vuelta los valores
        grilla = np.array([grilla_vac1,grilla_vac2])
        grilla = np.expand_dims(grilla,axis=(-2))
        
        
        return np.multiply(porc_infected,grilla)
    

def porc_days_to_uci_by_vac(porc_uci,D,G,W,
                            max_days_go_uci= 23,
                            list_p_to_uci=[0.4327, 0.2067, 0.1125, 0.3757, 0.2541],
                            list_n_to_uci=[8.0918, 2.638, 1.6357, 4.8271, 3.8911],
                            custom_distribution=False, 
                            for_train_not_vac=False,
                            for_train_vac=False, 
                            matrix_probability=np.nan,
                            window_slide=3):
    """
    [40-49,50-59,60-69.<=39,>=70]
    
    Caso para un 1-D list of probability 
        max_days_go_uci: maximum number of days that a person can go to the ICU
        mu: poisson takes mu  as shape parameter.
        -------
        porc_uci : (D,G,T-(W-1),W)
        porc time to uci: (G,H) ---> (H, 1, G, 1, 1)
        **H: days in future
        -----------
        return
            (H,D,G,T-(W-1),W)
            
    Caso para un 2-D list of probability not vac
        max_days_go_uci: maximum number of days that a person can go to the ICU
        mu: poisson takes mu  as shape parameter.
        -------
        porc_uci : (X,D,G,T-(W-1),W)
        porc time to uci: (Y,G,H) ---> (Y,H,1, 1, G, 1, 1)
        **H: days in future
        -----------
        return
            (Y,H,X,D,G,T-(W-1),W)
            
    Caso train vac
        max_days_go_uci: maximum number of days that a person can go to the ICU
        mu: poisson takes mu  as shape parameter.
        -------
        porc_uci :(Q,D,G,T-(W-1),W)
        porc time to uci: (G,H) ---> (H,1, 1, G, 1, 1)
        **H: days in future
        -----------
        return
            (H,Q,D,G,T-(W-1),W)
        
    """
    

        
    if custom_distribution:
        x = np.arange(7) #time
        p = (0.1, 0.2, 0.3, 0.1, 0.1, 0.0, 0.2)
        #custm = stats.rv_discrete(name='custm', values=(xk, pk))
        #custm.pmf(xk)
    else:
        x = np.arange(max_days_go_uci)
        aux=np.array([list_n_to_uci,list_p_to_uci]).T
        rvs_nb=[nbinom(item[0],item[1] )for item in aux ]
        
        acc_nb=np.expand_dims(np.array([rv.cdf(max_days_go_uci-1) for rv in rvs_nb]),axis=(-1))

        
        p = np.array([rv.pmf(x) for rv in rvs_nb])
        
    
    p_aux = np.zeros((p.shape[0],p.shape[1]+window_slide))
    p_aux[:,window_slide:] = p[:,:]
    grilla= np.expand_dims((p/acc_nb).transpose([1,0]),axis=(1,-2,-1))
    #grilla= np.expand_dims(p.transpose([1,0]),axis=(1,-2,-1))
    #grilla.shape -->(max_days_go_uci, 1, G, 1, 1)
    
    return np.multiply(porc_uci,grilla)



def porc_days_in_uci_by_vac(porc_days_to_uci,G,
                            max_days_in_uci=28,
                            list_p_in_uci=[0.0511, 0.0702, 0.0513, 0.0496, 0.0424],
                            list_n_in_uci=[1.408, 1.9843, 1.4427, 1.3219, 1.3486],
                            custom_distribution=False, 
                            for_train_not_vac=False,
                            for_train_vac=False,
                            matrix_probability=np.nan,
                            window_slide=2):
    """
    
    [40.49,50-59,60-69.<=39,>=70]
    max_days_in_uci: maximum number of days that a person can stay in the ICU
    mu: poisson takes mu  as shape parameter.
    -------
     porc_to_uci: (H,D,G,T-(W-1),W)
    porc time in uci: (G,K) ---> (K,1, 1, G, 1, 1)
    **K: days in future
    -----------
    return
        (K,H,D,G,T-(W-1),W)
        
    Caso para un 2-D list of probability
        max_days_in_uci: maximum number of days that a person can stay in the ICU
    mu: poisson takes mu  as shape parameter.
        -------
        (Y,H,X,D,G,T-(W-1),W) .(Y,H,X,D,G,T-(W-1),W)
        porc time in uci: (Z,G,K) ---> (Z,K,1,1,1, 1, G, 1, 1)
        **K: days in future
    -----------
    return
        (Z,K,Y,H,X,D,G,T-(W-1),W)
        
        
    -----------
     Caso train vac
        -------
        porc_to_uci : (H,Q,D,G,T-(W-1),W)
        porc time in uci: (G,K) ---> (K,1,1, 1, G, 1, 1)
        **H: days in future
        -----------
        return
            (K,H,Q,D,G,T-(W-1),W)
    """
    
    
    if for_train_not_vac:
        axis_0 =matrix_probability.shape[0]
        axis_1 =matrix_probability.shape[1]
        x = np.arange(max_days_in_uci) #time
        
        rvs =[ [poisson((1/matrix_probability[i,j])) for j in range(axis_1)] for i in range(axis_0)]
        p = np.array([[rv.pmf(x) for rv in rvs[i]] for i in range(axis_0)])
        acc=np.expand_dims(np.array([[rv.cdf(max_days_in_uci-1) for rv in rvs[i]]for i in range(axis_0)]),axis=(-1))
        
        
        grilla= np.expand_dims(p.transpose([0,2,1]),axis=(2,3,4,5,-2,-1))
        return np.multiply(porc_days_to_uci,grilla)
    
    if for_train_vac:
        pass
    
    else:
        if custom_distribution:
            pass

        else:
            x = np.arange(max_days_in_uci) #time
            aux=np.array([list_n_in_uci,list_p_in_uci]).T
            rvs_nb=[nbinom(item[0],item[1] )for item in aux ]
            acc=np.expand_dims(np.array([rv.cdf(max_days_in_uci-1) for rv in rvs_nb]),axis=(-1))
            p = np.array([rv.pmf(x) for rv in rvs_nb])
        
        grilla= np.expand_dims((p/acc).transpose([1,0]),axis=(1,2,-2,-1))
        return np.multiply(porc_days_to_uci,grilla)



def plot_rv_discrete(x,rv):
    fig, ax = plt.subplots(1, 1)
    ax.plot(x, rv.pmf(x), 'ro', ms=12, mec='r')
    ax.vlines(x, 0, rv.pmf(x), colors='r', lw=4)
    plt.show()
    
    fig, ax = plt.subplots(1, 1)
    ax.vlines(x, 0, rv.pmf(x), colors='k', linestyles='-', lw=1,
            label='frozen pmf')
    ax.legend(loc='best', frameon=False)
    plt.show()
    
    
    
def plot_uci_pred_v0(data, uci, W=100, pond=1, start_date=225):
    groupID_to_group = data['groupID_to_group']
    dateID_to_date = data['dateID_to_date']
    cols =['Grupo de edad', 'start_date','inf', 'uci_real', 'uci_pred']
    lst = []
    for g in range(len(groupID_to_group)):
        for date in range(start_date,data['uci'].shape[1]): 
            info = [groupID_to_group[g],dateID_to_date[date],data['inf'][g,date]*pond,data['uci'][g,date]]
            info.append(uci[g,date-(W-1)])
            lst.append(info)
    df_res = pd.DataFrame(lst, columns=cols)
    for i, (group_name, group_df) in enumerate(df_res.groupby(["Grupo de edad"])):
    #for i in range(2):
    
        
        
        #ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred','inf'])
        ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred'])
        # set the title
        ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
        ax.grid()
        
       # ax.xaxis_date()
        ##tks = 35
        #locator = mdates.AutoDateLocator(minticks=tks, maxticks=tks)
        #formatter = mdates.ConciseDateFormatter(locator)
        #ax.xaxis.set_major_locator(locator)
        #ax.xaxis.set_major_formatter(formatter)
    
        ax.axvspan(date2num(datetime(2021,2,3)), date2num(datetime(2021,2,15)), 
               label="older adults",color="green", alpha=0.3)
        ax.axvspan(date2num(datetime(2021,2,22)), date2num(datetime(2021,2,23)), 
               label="older adults",color="red", alpha=0.3)
        ax.axvspan(date2num(datetime(2021,3,3)), date2num(datetime(2021,3,15)), 
               label="2 vaccine older adults ",color="blue", alpha=0.3)
        ax.axes.set_title("Timeseries UCI for "+group_name ,fontsize=15)
        
        ax.axvspan(date2num(datetime(2021,1,15)), date2num(datetime(2021,1,16)), 
               label="older adults",color="red", alpha=0.3)
        
        ax.axvspan(date2num(datetime(2021,1,6)), date2num(datetime(2021,1,7)), 
               label="older adults",color="red", alpha=0.3)
        
        try:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name) +'.png')
        except:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name)[2:] +'.png')
        plt.tight_layout()
        plt.show()
        


def plot_uci_pred(data, uci,probability_not_vac,df_77,df_pop=prepare_population(), W=29, pond=1, start_date='2020-07-01',end_date='2021-05-10', infected=False, case=0,probability_not_vac_new_variant=None,line_porc_vacc=True):
    
    
    case_list=[
        "Base case",
        "Base case + vaccination campaign",
        "Base case + vaccination campaign + new variant without vaccination ",
        "Base case + vaccination campaign + new variant with vaccination"
        ]
    #mean_to_uci=[10.327, 9.127, 11.543, 7.,477, 10.866]
    #mean_in_uci=[23.307, 23.782, 23.831, 20.489, 27.321]
    mean_to_uci=[10.327, 9.127, 11.543, 7.477, 10.866]
    mean_in_uci=[26.169,  22.294, 26.657,25.319 , 30.453]
    
    
    porc=0.2
    var_row = 'Region'
    list_dosis = df_77["Dosis"].unique().tolist()
    list_reg = df_77["Region"].unique().tolist()
    data_porc=[]
    for i, (group_name, group_df) in enumerate(df_77.groupby(["Grupo de edad"])):
            group_df['accumulated_vaccinated']=group_df['accumulated_vaccinated']/df_pop[df_pop["Grupo de edad"]==group_name]['Personas'].values[0]
            value=group_df[(df_77[var_row]=='Total')&(group_df['Dosis']==list_dosis[0])&(group_df['accumulated_vaccinated']>=porc)]['start_date'].values[0]
            print(value)
            data_porc.append(value)
    groupID_to_group = data['groupID_to_group']
    dateID_to_date = data['dateID_to_date']
    date_to_dateID = data["date_to_dateID"]
    #start_date='2020-10-01'  

    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    cols =['Grupo de edad', 'start_date','inf', 'uci_real', 'uci_pred']
    lst = []
    for g in range(len(groupID_to_group)):
        for date in range(dateID_start_date,dateID_end_date+1): 
            info = [groupID_to_group[g],dateID_to_date[date],data['inf'][g,date]*pond,data['uci'][g,date]]
            info.append(uci[g,date-(W-1)])
            lst.append(info)
    df_res = pd.DataFrame(lst, columns=cols)
    
    group_name_position={
        '40-49':(0,1),
        '50-59':(1,0),
        '60-69':(1,1), 
        '<=39':(0,0)
        }
    figs, axs = plt.subplots(2,2,figsize=(30*2, 13*2),sharex=True, sharey=False)
    for i, (group_name, group_df) in enumerate(df_res.groupby(["Grupo de edad"])):
    #for i in range(2):
    
        #plt.subplots(figsize=(15, 9))
        # plot the group on these axes
        
        
        if group_name=='>=70':
            fig, ax = plt.subplots(figsize=(30, 13))
        else:
            index_group_name=group_name_position[group_name]
            ax=axs[index_group_name]
        # plot the group on these axes
        if group_name=='>=70':
            group_df.plot( kind='line', x="start_date",
                          y=['uci_real', 'uci_pred','inf'], ax=ax, 
                          #title="",
                          lw=6,
                          label=['UCI beds today (Producto 9)', 'Predicted UCI beds today (Flow model)', 'Symptomatic cases today'])
            
        else:
            if infected:
                group_df.plot( kind='line', x="start_date",
                              y=['uci_real', 'uci_pred','inf'], ax=ax, 
                              lw=6,
                              label=['UCI beds today (Producto 9)', 'Predicted UCI beds today (Flow model)', 'Symptomatic cases today'])
            else:
                group_df.plot( kind='line', x="start_date",
                              y=['uci_real', 'uci_pred'],
                              ax=ax,
                              lw=6,
                              label=['Observed ICU occupancy', 'Expected ICU occupancy'])
        
        # Provide tick lines across the plot to help your viewers trace along    
        # the axis ticks. Make sure that the lines are light and small so they    
        # don't obscure the primary data lines.    
        
        # Define the upper limit, lower limit, interval of Y axis and colors
        y_LL = 100
        y_UL = int(max(group_df['uci_pred'].max(),group_df['uci_real'].max())*1.1)
        y_interval = 100
        # Draw Tick lines  
        
        if min(group_df['uci_pred'].min(),group_df['uci_real'].min())>200:
            y_LL=200
        for y in range(y_LL, y_UL, y_interval):    
            ax.hlines(y, xmin=group_df["start_date"].min()+pd.DateOffset(-2), xmax=group_df["start_date"].max()+pd.DateOffset(2), colors='black', alpha=0.4, linestyles="--", lw=0.55)
        
        
        # set the title
        ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
        ax.grid()
        
        ax.xaxis_date()
        """
        tks = 30
        #locator = mdates.AutoDateLocator(minticks=tks, maxticks=tks)
        locator = mdates.DateFormatter('%Y-%m-%d')
        formatter = mdates.ConciseDateFormatter(locator)
        #ax.xaxis.set_major_locator(locator)
        #ax.xaxis.set_major_formatter(formatter)
        """
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        """
        ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,3)), 
               label="older adults",color="red", alpha=0.3)
        ax.axvspan(date2num(datetime(2021,2,3)), date2num(datetime(2021,2,15)), 
               label="older adults",color="green", alpha=0.3)
        ax.axvspan(date2num(datetime(2021,2,22)), date2num(datetime(2021,2,23)), 
               label="older adults",color="red", alpha=0.3)
        ax.axvspan(date2num(datetime(2021,3,3)), date2num(datetime(2021,3,15)), 
               label="2 vaccine older adults ",color="blue", alpha=0.3)
        ax.axes.set_title("Timeseries UCI for "+group_name ,fontsize=15)
        
        ax.axvspan(date2num(datetime(2021,1,15)), date2num(datetime(2021,1,16)), 
               label="older adults",color="red", alpha=0.3)
        
        ax.axvspan(date2num(datetime(2021,1,6)), date2num(datetime(2021,1,7)), 
               label="older adults",color="red", alpha=0.3)
        """
        
        ax.axes.set_title("Timeseries UCI for "+group_name +""
                          "\n"+case_list[case],fontsize= 25)
        
        ax.set_ylabel('ICU occupancy', fontsize=27)
        ax.set_xlabel('Date \n', fontsize=20)
        
        ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,2)), 
               label="older adults",color="green", alpha=0.3)
        ax.axvspan(date2num(data_porc[i]), date2num(data_porc[i])+1, 
               label="older adults",color="blue", alpha=0.3)

        add_days=21
        ax.axvspan(date2num(data_porc[i])+add_days, date2num(data_porc[i])+1+add_days, 
               label="older adults",color="red", alpha=0.3)
        """
        
        bbox_props = dict(boxstyle="round4, pad=0.6", fc="cyan", ec="b", lw=.5)
        ax.annotate('Date = {}'.format( pd.to_datetime(data_porc[0]).strftime('%a, %Y-%m-%d')),
            fontsize=9,
            fontweight='bold',
            #xy=(date2num(data_porc[i])+add_days,date2num(data_porc[i])+add_days+1), 
            xy=(0.5, 0.5),
            xycoords='axes fraction',
            xytext=(0.5, 0.9),      
            textcoords='offset points',
            arrowprops=dict(arrowstyle="->"), bbox=bbox_props)   
        """
        #ax.label_outer()
        ax.xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=16)
        ax.fill_between( x=group_df["start_date"].values,y1=group_df['uci_pred']+20,y2=group_df['uci_pred']-20, facecolor='orange', alpha=0.2)
        if case ==2 |case ==3:
            ax.annotate(""
                    "\n* Data source: https://github.com/MinCiencia/Datos-COVID19" 
                    "\n** The flow model was fitted with a probability of going to the ICU:  "+str(round(probability_not_vac[i],3))+""
                    "\n** The flow model was fitted with a probability of going to the ICU for new variant:  "+str(round(probability_not_vac_new_variant[i],3))+""
                    "\n*** The time to ICU is modeled as a negative binomial distribution with mean:  "+str(round(mean_to_uci[i],1))+""
                    "\n*** The time in ICU is modeled as a negative binomial distribution with mean:  "+str(round(mean_in_uci[i],1))
                    , xy=(0.0, -0.3), xycoords="axes fraction",fontsize = 17)#color = '#f0f0f0', backgroundcolor = 'grey'
        else:
            ax.annotate(""
                        "\n* Data source: https://github.com/MinCiencia/Datos-COVID19" 
                        "\n** The flow model was fitted with a probability of going to the ICU:  "+str(round(probability_not_vac[i],3))+""
                        "\n*** The time to ICU is modeled as a negative binomial distribution with mean:  "+str(round(mean_to_uci[i],1))+""
                        "\n*** The time in ICU is modeled as a negative binomial distribution with mean:  "+str(round(mean_in_uci[i],1))
                        , xy=(0.0, -0.25), xycoords="axes fraction",fontsize = 17)#color = '#f0f0f0', backgroundcolor = 'grey'
        # plot the group on these axes
        plt.tight_layout()
    """
        try:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name) +'.png')
        except:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name)[2:] +'.png')
    """
        #plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
    #plt.setp(axs[0,0].get_xticklabels(), visible=False)
    axs[0,1].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=14)
    plt.gcf().autofmt_xdate()
    plt.show()
        
        
def plot_uci_pred_sns(data, uci, W=29, pond=1, start_date='2020-07-01', end_date='2021-05-15', infected=False):
    groupID_to_group = data['groupID_to_group']
    dateID_to_date = data['dateID_to_date']
    date_to_dateID = data["date_to_dateID"]
    #start_date='2020-10-01'

    dateID_start_date = data['date_to_dateID'][np.datetime64(
        start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(
        end_date+"T00:00:00.000000000")]
    cols = ['Grupo de edad', 'start_date', 'inf', 'uci', 'type']
    lst = []
    for g in range(len(groupID_to_group)):
        for date in range(dateID_start_date, dateID_end_date+1):
            info = [groupID_to_group[g], dateID_to_date[date],
                    data['inf'][g, date]*pond]
            info1 = info.copy()
            info2 = info.copy()
            info1.extend([data['uci'][g, date], 'real'])
            info2.extend([uci[g, date-(W-1)], 'predicted'])
            lst.append(info1)
            lst.append(info2)
    df_res = pd.DataFrame(lst, columns=cols)

    #palette = sns.cubehelix_palette(light=.7, n_colors=6)
    for i, (group_name, group_df) in enumerate(df_res.groupby(["Grupo de edad"])):
        #for i in range(2):

        #plt.subplots(figsize=(15, 9))
        # plot the group on these axes

        #fig, ax = plt.subplots()

        g = sns.relplot(x="start_date", y="uci",
                        hue="type",
                        #palette=palette,
                        kind="line", data=group_df, sizes=((15, 9)))

        #ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred'])
        # set the title
        g.set(xlim=[group_df["start_date"].min()+pd.DateOffset(-2),
              group_df["start_date"].max()+pd.DateOffset(2)])
        g.ax.grid()

        g.ax.xaxis_date()
        """
        tks = 30
        #locator = mdates.AutoDateLocator(minticks=tks, maxticks=tks)
        locator = mdates.DateFormatter('%Y-%m-%d')
        formatter = mdates.ConciseDateFormatter(locator)
        #ax.xaxis.set_major_locator(locator)
        #ax.xaxis.set_major_formatter(formatter)
        """
        g.ax.xaxis.set_major_locator(mdates.MonthLocator())
        g.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

        g.ax.axvspan(date2num(datetime(2021, 1, 1)), date2num(datetime(2021, 1, 3)),
                     label="older adults", color="red", alpha=0.3)

        """
        ax.axvspan(date2num(datetime(2021,2,3)), date2num(datetime(2021,2,15)), 
               label="older adults",color="green", alpha=0.3)
        ax.axvspan(date2num(datetime(2021,2,22)), date2num(datetime(2021,2,23)), 
               label="older adults",color="red", alpha=0.3)
        ax.axvspan(date2num(datetime(2021,3,3)), date2num(datetime(2021,3,15)), 
               label="2 vaccine older adults ",color="blue", alpha=0.3)
        ax.axes.set_title("Timeseries UCI for "+group_name ,fontsize=15)
        
        ax.axvspan(date2num(datetime(2021,1,15)), date2num(datetime(2021,1,16)), 
               label="older adults",color="red", alpha=0.3)
        
        ax.axvspan(date2num(datetime(2021,1,6)), date2num(datetime(2021,1,7)), 
               label="older adults",color="red", alpha=0.3)
        """

        g.ax.axes.set_title("Timeseries UCI for "+group_name, fontsize=15)
        g.ax.set_ylabel('N° beds', fontsize=10)
        g.ax.set_xlabel('Date', fontsize=10)

        g.ax.legend(bbox_to_anchor=(1.01, 1.02), loc='upper left')
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        """
        try:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name) +'.png')
        except:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name)[2:] +'.png')
        """
        #plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
        plt.show()


def plot_dead_pred_sns(data, dead, W=29, pond=0.22, start_date='2020-07-01',end_date='2021-05-15', infected=False):
    groupID_to_group = data['groupID_to_group']
    dateID_to_date = data['dateID_to_date']
    date_to_dateID = data["date_to_dateID"]
    #start_date='2020-10-01'  

    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    cols =['Grupo de edad', 'start_date','inf', 'dead', 'type']
    lst = []
    for g in range(len(groupID_to_group)):
        for date in range(dateID_start_date,dateID_end_date+1): 
            info = [groupID_to_group[g],dateID_to_date[date],data['inf'][g,date]*pond]
            info1=info.copy()
            info2=info.copy()
            info1.extend([data['dead'][g,date], 'real'])
            info2.extend([dead[g,date-(W-1)]*pond, 'predicted'])
            lst.append(info1)
            lst.append(info2)
    df_res = pd.DataFrame(lst, columns=cols)
    
    #palette = sns.cubehelix_palette(light=.7, n_colors=6)
    for i, (group_name, group_df) in enumerate(df_res.groupby(["Grupo de edad"])):
    #for i in range(2):
    
        #plt.subplots(figsize=(15, 9))
        # plot the group on these axes
            
        #fig, ax = plt.subplots()

        g=sns.relplot(x="start_date", y="dead",
                                       hue="type",
                                       #palette=palette,
                                       kind="line", data=group_df,sizes=((15, 9)))
        
        #ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred'])
        # set the title
        g.set(xlim=[group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
        g.ax.grid()
        
        g.ax.xaxis_date()
        """
        tks = 30
        #locator = mdates.AutoDateLocator(minticks=tks, maxticks=tks)
        locator = mdates.DateFormatter('%Y-%m-%d')
        formatter = mdates.ConciseDateFormatter(locator)
        #ax.xaxis.set_major_locator(locator)
        #ax.xaxis.set_major_formatter(formatter)
        """
        g.ax.xaxis.set_major_locator(mdates.MonthLocator())
        g.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        
        g.ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,3)), 
               label="older adults",color="red", alpha=0.3)
        
        """
        ax.axvspan(date2num(datetime(2021,2,3)), date2num(datetime(2021,2,15)), 
               label="older adults",color="green", alpha=0.3)
        ax.axvspan(date2num(datetime(2021,2,22)), date2num(datetime(2021,2,23)), 
               label="older adults",color="red", alpha=0.3)
        ax.axvspan(date2num(datetime(2021,3,3)), date2num(datetime(2021,3,15)), 
               label="2 vaccine older adults ",color="blue", alpha=0.3)
        ax.axes.set_title("Timeseries UCI for "+group_name ,fontsize=15)
        
        ax.axvspan(date2num(datetime(2021,1,15)), date2num(datetime(2021,1,16)), 
               label="older adults",color="red", alpha=0.3)
        
        ax.axvspan(date2num(datetime(2021,1,6)), date2num(datetime(2021,1,7)), 
               label="older adults",color="red", alpha=0.3)
        """
        
        g.ax.axes.set_title("Timeseries dead for "+group_name ,fontsize=15)
        g.ax.set_ylabel('N° beds', fontsize=10)
        g.ax.set_xlabel('Date', fontsize=10)
        
        #g.ax.legend(bbox_to_anchor=(1.01, 1.02), loc='upper left')
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        """
        try:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name) +'.png')
        except:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name)[2:] +'.png')
        """
        #plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
        plt.show()
        
def plot_market_share_pred_2020(data,uci,uci_share_not_variant, W=29, pond=1, start_date='2021-01-01',end_date='2021-04-23', infected=False):
    groupID_to_group = data['groupID_to_group']
    dateID_to_date = data['dateID_to_date']
    date_to_dateID = data["date_to_dateID"]
    #start_date='2020-10-01'  

    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    cols =['Grupo de edad', 'start_date','uci_share_not_variant', 'uci_share_new_variant','uci_pred']
    lst = []
    for g in range(len(groupID_to_group)):
        for date in range(dateID_start_date,dateID_end_date): 
            info = [groupID_to_group[g],dateID_to_date[date]]
            info.append(uci_share_not_variant[g,date-(W-1)])
            info.append(1-uci_share_not_variant[g,date-(W-1)])
            info.append(uci[g,date-(W-1)])
            lst.append(info)
    df_res = pd.DataFrame(lst, columns=cols)
    
    aux_values=[]
    for i, (group_name, group_df) in enumerate(df_res.groupby(["Grupo de edad"])):
    #for i in range(2):
        aux_values.append(group_df[['uci_share_new_variant','uci_pred']].tail(1).values)
        #plt.subplots(figsize=(15, 9))
        # plot the group on these axes
        if group_name=='>=70':
            ax=group_df.plot( kind='line', x="start_date", y=['uci_share_not_variant', 'uci_share_new_variant'])
            
        else:
            if infected:
                ax=group_df.plot( kind='line', x="start_date", y=['uci_share_not_variant', 'uci_share_new_variant'])
            else:
                ax=group_df.plot( kind='line', x="start_date", y=['uci_share_not_variant', 'uci_share_new_variant'])
        # set the title
        ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
        ax.grid()
        
        ax.xaxis_date()
        """
        tks = 30
        #locator = mdates.AutoDateLocator(minticks=tks, maxticks=tks)
        locator = mdates.DateFormatter('%Y-%m-%d')
        formatter = mdates.ConciseDateFormatter(locator)
        #ax.xaxis.set_major_locator(locator)
        #ax.xaxis.set_major_formatter(formatter)
        """
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        
        #ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,3)), 
               #label="older adults",color="red", alpha=0.3)
        
        """
        ax.axvspan(date2num(datetime(2021,2,3)), date2num(datetime(2021,2,15)), 
               label="older adults",color="green", alpha=0.3)
        ax.axvspan(date2num(datetime(2021,2,22)), date2num(datetime(2021,2,23)), 
               label="older adults",color="red", alpha=0.3)
        ax.axvspan(date2num(datetime(2021,3,3)), date2num(datetime(2021,3,15)), 
               label="2 vaccine older adults ",color="blue", alpha=0.3)
        ax.axes.set_title("Timeseries UCI for "+group_name ,fontsize=15)
        
        ax.axvspan(date2num(datetime(2021,1,15)), date2num(datetime(2021,1,16)), 
               label="older adults",color="red", alpha=0.3)
        
        ax.axvspan(date2num(datetime(2021,1,6)), date2num(datetime(2021,1,7)), 
               label="older adults",color="red", alpha=0.3)
        """
        
        ax.axes.set_title("Timeseries UCI for "+group_name ,fontsize=15)
        ax.set_ylabel('% N° beds', fontsize=10)
        ax.set_xlabel('Date', fontsize=10)
        x_coordinates = [dateID_to_date[324],dateID_to_date[335] ,
                         dateID_to_date[342],dateID_to_date[348],
                         dateID_to_date[355]]
        y_coordinates = [0.333, 0.2, 0.53,0.5,0.813]

        plt.scatter(x_coordinates, y_coordinates)
        try:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name) +'.png')
        except:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name)[2:] +'.png')
        
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        #plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
        plt.show()
        
    print(aux_values)

    
def MSE():
    #mean_inial = np.mean((uci_inicial[:,dateID_start_date-(window-1):dateID_firt_dosis-(window-1)]-uci)**2,axis=-1)
    #mean_aux = np.mean((aux-uci)**2,axis=-1)
    #[np.min(mean_aux[:,:,:,i]) for i  in range(5)]
    #[np.max(mean_aux[:,:,:,i]) for i  in range(5)]
    
    #mean_aux=np.mean(((uci_inicial_not-uci)**2), axis=-1)
    pass


def split_infed_new_variant(data,probability_new_variant,ID_list_day_new_variant,probability_b117,ID_list_day_b117,shift=False):
    """
    

    Parameters
    ----------
    data : dictionary with very useful information
        type  dictionary
    probability_new_variant : probability of catching one variant per month
        type list 
        
    ID_list_day_new_variant :id days that the probability of getting infected changes
        DESCRIPTION.

    Returns
    -------
    

    """
    #shif curve
    shif_curve=[8, 7, 8, 7, 4]
    #shif_curve=[0,0,0,0,0]
    inf = data['inf'].copy()
    
    if shift:
    
        for i,item in enumerate(shif_curve):
            inf[i,0:-(item-1)]=data['inf'][i,item-1:]
    
    ID_list_day_new_variant.append(inf.shape[1]) #add the last day of inf
    ID_list_day_new_variant.insert(0,0)
    
    for item in probability_new_variant: #by age extender la serie dia 1 hasta el ultimo
        item.append(item[-1]) #replicar el ultimo valor
        item.insert(0,0)#replicar el primer valor
    ID_list_day_new_variant = np.array(ID_list_day_new_variant)
    probability_new_variant = np.array(probability_new_variant)
    ID_firts_day_new_variant=ID_list_day_new_variant[0]
    day_step=np.diff(ID_list_day_new_variant)
    
    
    ID_list_day_b117.append(inf.shape[1]) #add the last day of inf
    ID_list_day_b117.insert(0,0)
    for item in probability_b117:
        item.append(item[-1])
        item.insert(0,0)
    ID_list_day_b117 = np.array(ID_list_day_b117)
    probability_b117 = np.array(probability_b117)
    ID_firts_day_b117=ID_list_day_b117[0]
    day_step_b117=np.diff(ID_list_day_b117)
    
    #generate the market share for all time series
    grilla_new_variant=np.array([np.concatenate(
        [np.linspace(probability_new_variant[j,i],probability_new_variant[j,i+1],day_step[i],endpoint=False) for i in range(day_step.shape[0])])
        for j in range(5)])
    grilla_b117=np.array([np.concatenate(
        [np.linspace(probability_b117[j,i],probability_b117[j,i+1],day_step_b117[i],endpoint=False) for i in range(day_step_b117.shape[0])])
        for j in range(5)])
    data['grilla_new_variant']=grilla_new_variant
    data['grilla_b117']=grilla_b117
    #data['inf_no_variant']=data['inf'].copy()
    data['inf_no_variant']=inf.copy()
    data['inf_no_variant']= inf.copy()*(1-grilla_new_variant-grilla_b117)
    data['inf_new_variant']= inf.copy()*(grilla_new_variant)
    data['inf_b117']=inf.copy()*(grilla_b117)
    
    
def generate_list(probability_not_vac_new_variant):
    prob_uci_vac1_new_variant=[]
    for i in range(5):
        prob_uci_vac1_new_variant.append([probability_not_vac_new_variant[i] for j in range(6)])
        
    return prob_uci_vac1_new_variant,prob_uci_vac1_new_variant

def genere_list_minsal(prob_inicial, proc_final_v1,proc_final_v2):
    prob_final_vac1= prob_inicial*proc_final_v1
    prob_final_vac2= prob_inicial*proc_final_v2
    vac_1 = np.around(np.linspace(prob_inicial, prob_final_vac1, num=6),4).T.tolist()
    vac_2 = np.around(np.linspace(prob_final_vac1, prob_final_vac2, num=6),4).T.tolist()
    #vac_2.extend(np.around(np.linspace(prob_final_vac1, prob_final_vac2, num=2),4).tolist())
    #vac_2.extend(np.around(np.linspace(prob_final_vac2, prob_final_vac2, num=3),4).T.tolist())
    return vac_1,vac_2




def genere_list_minsal_denis(prob_inicial,pt_factor=[0.16,0.67],minsal_factor=[0.43,0.89]):
    num_b=len(prob_inicial)
    aux_factor1=(1-minsal_factor[0])/(1-pt_factor[0])
    aux_factor2=(1-minsal_factor[1])/(1-pt_factor[1])
    factor_vac1=np.round(np.array([[1,1,aux_factor1,aux_factor1,aux_factor1,aux_factor1]]*num_b),4)
    factor_vac2=np.round(np.array([[aux_factor1,aux_factor1,aux_factor2,aux_factor2,aux_factor2,aux_factor2]]*num_b),4)
    
    vac_1 = np.round((prob_inicial*factor_vac1.T).T,4).tolist()
    vac_2 = np.round((prob_inicial*factor_vac2.T).T,4).tolist()
    return vac_1,vac_2

#aux(data,probability_not_vac,probability_not_vac_new_variant,prob_uci_vac1_Sinovac,prob_uci_vac1_new_variant_Sinovac,prob_uci_vac2_Sinovac,prob_uci_vac2_new_variant_Sinovac)
def aux(data,probability_not_vac,probability_not_vac_new_variant,prob_uci_vac1,prob_uci_vac1_new_variant,prob_uci_vac2,prob_uci_vac2_new_variant):
    aux=pd.DataFrame(index=data['groupID_to_group'].values())
    aux['No vaccine']=np.round(probability_not_vac,4)
    aux['No vaccine new variant']=probability_not_vac_new_variant
    aux['First dose 14 days']=np.array(prob_uci_vac1)[:,2]
    aux['First dose 14 days new variant']=np.array(prob_uci_vac1_new_variant)[:,2]
    aux['First dose 28 days']=np.array(prob_uci_vac1)[:,5]
    aux['First dose 28 days new variant']=np.array(prob_uci_vac1_new_variant)[:,5]
    aux['Second dose 14 days']=np.array(prob_uci_vac2)[:,2]
    aux['Second dose 14 days new variant']=np.array(prob_uci_vac2_new_variant)[:,2]
    aux['Second dose 28 days']=np.array(prob_uci_vac2)[:,5]
    aux['Second dose 28 days new variant']=np.array(prob_uci_vac2_new_variant)[:,5]
    aux.sort_values(by='No vaccine',inplace=True)
    f, ax = plt.subplots(figsize=(14, 13))
    sns.heatmap(aux, annot=True, linewidths=1.5, ax=ax, cmap='RdYlGn_r', annot_kws={"size": 20},fmt='g')
    ax.axes.set_title("Probability of ICU admission", fontsize=24, y=1.01)
    ax.set_yticklabels(aux.index, rotation=0,fontsize=20)
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='left')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()
    
    
    aux=pd.DataFrame(index=data['groupID_to_group'].values())
    aux['No vaccine']=np.round(probability_not_vac,4)
    aux['No vaccine new variant']=probability_not_vac_new_variant
    aux['First dose 14 days']=np.array(prob_uci_vac1)[:,2]
    aux['First dose 14 days new variant']=np.array(prob_uci_vac1_new_variant)[:,2]
    aux['First dose 28 days']=np.array(prob_uci_vac1)[:,5]
    aux['First dose 28 days new variant']=np.array(prob_uci_vac1_new_variant)[:,5]
    aux['Second dose 14 days']=np.array(prob_uci_vac2)[:,2]
    aux['Second dose 14 days new variant']=np.array(prob_uci_vac2_new_variant)[:,2]
    aux['Second dose 28 days']=np.array(prob_uci_vac2)[:,5]
    aux['Second dose 28 days new variant']=np.array(prob_uci_vac2_new_variant)[:,5]
    
    aux.sort_values(by='No vaccine',inplace=True)
    firts_column=aux['No vaccine'].values.copy()
    for column in aux.columns:
        values=aux[column].values/firts_column
        aux[column]=np.round(values,3)
    
    f, ax = plt.subplots(figsize=(14, 13))
    sns.heatmap(aux, annot=True, linewidths=1.5, ax=ax, cmap='RdYlGn_r', annot_kws={"size": 20},fmt='g')
    ax.axes.set_title("Weights for probability of ICU admission", fontsize=24, y=1.01)
    ax.set_yticklabels(aux.index, rotation=0,fontsize=20)
    plt.setp(ax.get_xticklabels(), rotation=45, horizontalalignment='left',fontsize=20)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()
    
    pass
