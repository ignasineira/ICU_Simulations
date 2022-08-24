#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 21 19:00:08 2022

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

path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]
if path_projet[-1]!='/':
    path_projet+='/'
path_projet+='ICU_Simulations/'
path_ETL='Code/ETL'

module_path=path_projet+path_ETL
if module_path not in sys.path:
    sys.path.append(module_path)

from data_processing import read_data,prepare_population

from fit_probability_to_icu import find_probability_icu
from simulation_v9 import genere_list_minsal_denis,simulation_v9_fase_1,plot_uci_pred_sns
from simulation_MonteCarlos_v9 import Camas_UCI_new_variant

import seaborn as sns ## These Three lines are necessary for Seaborn to work   
import matplotlib.pyplot as plt 
from matplotlib.dates import date2num
import matplotlib.dates as mdates


def call_icu_simulation_2020(data,params,
                        save_data=True,
                        update_data=True):
    """
    Main idea:
    """

    if update_data:
        """
        uci,dead,dict_main_dead = simulation_v9(data,
                        increase_vac1_Pfizer,increase_vac2_Pfizer,
                        increase_vac1_Sinovac,increase_vac2_Sinovac, 
                        prob_uci_vac1_Pfizer,prob_uci_vac2_Pfizer,
                        prob_uci_vac1_Sinovac,prob_uci_vac2_Sinovac,
                        prob_uci_vac1_b117_Pfizer,prob_uci_vac2_b117_Pfizer,
                        prob_uci_vac1_b117_Sinovac,prob_uci_vac2_b117_Sinovac,
                        prob_uci_vac1_new_variant_Pfizer,prob_uci_vac2_new_variant_Pfizer,
                        prob_uci_vac1_new_variant_Sinovac,prob_uci_vac2_new_variant_Sinovac,
                        p_shares_new_variant=p_shares_new_variant,
                        ID_list_day_new_variant=ID_list_day_new_variant,
                        p_shares_b117=p_shares_b117,
                        ID_list_day_b117=ID_list_day_b117,
                        window=window, 
                        probability_not_vac = probability_not_vac,
                        probability_not_vac_b117 = probability_not_vac_b117,
                        probability_not_vac_new_variant = probability_not_vac_new_variant,
                        list_p_to_uci=list_p_to_uci,
                        list_n_to_uci= list_n_to_uci,
                        list_p_to_uci_2021=list_p_to_uci_2021,
                        list_n_to_uci_2021= list_n_to_uci_2021,
                        ID_day_change_time=ID_day_change_time,
                        list_p_in_uci=list_p_in_uci,
                        list_n_in_uci=list_n_in_uci,
                        max_days_go_uci=max_days_go_uci,
                        max_days_in_uci= max_days_in_uci, 
                        window_slide_to_uci=0,
                        shift=shift)
        """
        
        increase_vac1_Pfizer=[[0.0, 0.0, 0.524, 0.524, 0.524, 0.524],
            [0.0, 0.0, 0.524, 0.524, 0.524, 0.524],
            [0.0, 0.0, 0.524, 0.524, 0.524, 0.524],
            [0.0, 0.0, 0.524, 0.524, 0.524, 0.524],
            [0.0, 0.0, 0.524, 0.524, 0.524, 0.524]]
        increase_vac2_Pfizer=[[0.524, 0.524, 0.9, 0.9, 0.9, 0.9],
            [0.524, 0.524, 0.9, 0.9, 0.9, 0.9],
            [0.524, 0.524, 0.9, 0.9, 0.9, 0.9],
            [0.524, 0.524, 0.9, 0.9, 0.9, 0.9],
            [0.524, 0.524, 0.9, 0.9, 0.9, 0.9]]
        
        increase_vac1_Sinovac=[[0.0, 0.0, 0.16, 0.16, 0.16, 0.16],
            [0.0, 0.0, 0.16, 0.16, 0.16, 0.16],
            [0.0, 0.0, 0.16, 0.16, 0.16, 0.16],
            [0.0, 0.0, 0.16, 0.16, 0.16, 0.16],
            [0.0, 0.0, 0.16, 0.16, 0.16, 0.16]]
        increase_vac2_Sinovac=[[0.16, 0.16, 0.636, 0.636, 0.636, 0.636],
            [0.16, 0.16, 0.636, 0.636, 0.636, 0.636],
            [0.16, 0.16, 0.636, 0.636, 0.636, 0.636],
            [0.16, 0.16, 0.636, 0.636, 0.636, 0.636],
            [0.16, 0.16, 0.636, 0.636, 0.636, 0.636]]
        
        probability_not_vac=[0.011, 0.029, 0.081, 0.0032, 0.083] #strart parameters
        start_date='2020-05-01';end_date='2020-12-31'
        list_p_in_uci=[0.0699, 0.0619, 0.0515, 0.0739, 0.0412];
        list_n_in_uci=[1.9495, 1.4646, 1.4444, 2.0101, 1.303];
        
        probability_not_vac[:-1]=find_probability_icu(data, start_date, end_date,list_p_in_uci,list_n_in_uci)[:-1].tolist()
        
        prob_uci_vac1_Pfizer,prob_uci_vac2_Pfizer=genere_list_minsal_denis(np.array(probability_not_vac),pt_factor=[0.524,0.909],minsal_factor=[0.99,0.984])
        prob_uci_vac1_Sinovac,prob_uci_vac2_Sinovac=genere_list_minsal_denis(np.array(probability_not_vac),pt_factor=[0.16,0.636],minsal_factor=[0.43,0.90])
        
        
        probability_not_vac_b117=probability_not_vac.copy()
        aplha=2.03
        probability_not_vac_b117[:-1]=(np.array(probability_not_vac_b117[:-1])*aplha).tolist()
        prob_uci_vac1_b117_Pfizer,prob_uci_vac2_b117_Pfizer=genere_list_minsal_denis(np.array(probability_not_vac_b117),pt_factor=[0.524,0.909],minsal_factor=[0.99,0.984])
        prob_uci_vac1_b117_Sinovac,prob_uci_vac2_b117_Sinovac=genere_list_minsal_denis(np.array(probability_not_vac_b117),pt_factor=[0.16,0.636],minsal_factor=[0.43,0.90])
        
        probability_not_vac_new_variant=[0.0530, 0.1117, 0.089, 0.0115, 0.072]
        prob_uci_vac1_new_variant_Pfizer,prob_uci_vac2_new_variant_Pfizer=genere_list_minsal_denis(np.array(probability_not_vac_new_variant),pt_factor=[0.524,0.909],minsal_factor=[0.99,0.984])
        prob_uci_vac1_new_variant_Sinovac,prob_uci_vac2_new_variant_Sinovac=genere_list_minsal_denis(np.array(probability_not_vac_new_variant),pt_factor=[0.16,0.636],minsal_factor=[0.43,0.90])
        
        
        p_shares_new_variant=[[0,0, 0.0, 0.269, 0.5, 0.739, 0.643],
                            [0,0, 0.0, 0.357, 0.517, 0.522, 0.833],
                            [0,0, 0.0, 0.462, 0.55, 0.818, 0.818],
                            [0,0, 0.057, 0.418, 0.465, 0.636, 0.62],
                            [0,0, 0.0, 0.333, 0.333, 0.571, 0.75]]
        ID_list_day_new_variant=[260,289,320,348,379,409,440]
        p_shares_b117=[[0,0, 0.062, 0.115, 0.096, 0.0, 0.0],
                        [0,0, 0.0, 0.0, 0.069, 0.0, 0.0],
                        [0,0, 0.0, 0.308, 0.1, 0.0, 0.0],
                        [0,0, 0.057, 0.114, 0.088, 0.029, 0.014],
                        [0,0, 0.0, 0.0, 0.167, 0.0, 0.0]]
        ID_list_day_b117=[260,289,320,348,379,409,440]
        shift = True
        uci=simulation_v9_fase_1(data,
                          increase_vac1_Pfizer,increase_vac2_Pfizer,
                          increase_vac1_Sinovac,increase_vac2_Sinovac,
                          prob_uci_vac1_Pfizer,prob_uci_vac2_Pfizer,
                          prob_uci_vac1_Sinovac,prob_uci_vac2_Sinovac,
                          window=29, 
                          probability_not_vac=probability_not_vac,
                          list_p_to_uci=[0.463, 0.2139, 0.1494, 0.3976, 0.2466],
                          list_n_to_uci= [8.8889, 2.6667, 2.2222, 5.1111, 3.6667],
                          list_p_to_uci_2021=[0.699, 0.402, 0.365, 0.508, 0.7  ],
                          list_n_to_uci_2021= [21.798 ,  6.2727,  6.5657,  9.202 , 22.0909],
                          ID_day_change_time=275,
                          list_p_in_uci=[0.0699, 0.0619, 0.0515, 0.0739, 0.0412],
                          list_n_in_uci=[1.9495, 1.4646, 1.4444, 2.0101, 1.303],
                          max_days_go_uci=30,
                          max_days_in_uci= 100,
                          window_slide_to_uci=0,shift=True)
        
        plot_uci_pred_sns(data, uci, W=29, pond=1, start_date='2020-07-01', end_date='2021-05-15', infected=False)

        if save_data:
            #save icu, dead people,dict_main_dead
            save_data_icu_simulation_2020(uci)
            
    else:
        #call data 
        #print las update
        uci = open_data_simulation_2020()
        
    return uci



def call_icu_MonteCarlos_simulation_2020(data,params,
                        save_data=True,
                        update_data=True):
    """
    Main idea:
    """
    if update_data:

        increase_vac1_Pfizer=[[0.0, 0.0, 0.524, 0.524, 0.524, 0.524],
            [0.0, 0.0, 0.524, 0.524, 0.524, 0.524],
            [0.0, 0.0, 0.524, 0.524, 0.524, 0.524],
            [0.0, 0.0, 0.524, 0.524, 0.524, 0.524],
            [0.0, 0.0, 0.524, 0.524, 0.524, 0.524]]
        increase_vac2_Pfizer=[[0.524, 0.524, 0.9, 0.9, 0.9, 0.9],
            [0.524, 0.524, 0.9, 0.9, 0.9, 0.9],
            [0.524, 0.524, 0.9, 0.9, 0.9, 0.9],
            [0.524, 0.524, 0.9, 0.9, 0.9, 0.9],
            [0.524, 0.524, 0.9, 0.9, 0.9, 0.9]]
        
        increase_vac1_Sinovac=[[0.0, 0.0, 0.16, 0.16, 0.16, 0.16],
            [0.0, 0.0, 0.16, 0.16, 0.16, 0.16],
            [0.0, 0.0, 0.16, 0.16, 0.16, 0.16],
            [0.0, 0.0, 0.16, 0.16, 0.16, 0.16],
            [0.0, 0.0, 0.16, 0.16, 0.16, 0.16]]
        increase_vac2_Sinovac=[[0.16, 0.16, 0.636, 0.636, 0.636, 0.636],
            [0.16, 0.16, 0.636, 0.636, 0.636, 0.636],
            [0.16, 0.16, 0.636, 0.636, 0.636, 0.636],
            [0.16, 0.16, 0.636, 0.636, 0.636, 0.636],
            [0.16, 0.16, 0.636, 0.636, 0.636, 0.636]]
        
        probability_not_vac=[0.011, 0.029, 0.081, 0.0032, 0.083] #strart parameters
        start_date='2020-05-01';end_date='2020-12-31'
        list_p_in_uci=[0.0699, 0.0619, 0.0515, 0.0739, 0.0412];
        list_n_in_uci=[1.9495, 1.4646, 1.4444, 2.0101, 1.303];
        list_p_to_uci=[0.463, 0.2139, 0.1494, 0.3976, 0.2466];
        list_n_to_uci= [8.8889, 2.6667, 2.2222, 5.1111, 3.6667];
        list_p_to_uci_2021=[0.699, 0.402, 0.365, 0.508, 0.7  ];
        list_n_to_uci_2021= [21.798 ,  6.2727,  6.5657,  9.202 , 22.0909];
        ID_day_change_time=275;
        list_p_in_uci=[0.0699, 0.0619, 0.0515, 0.0739, 0.0412];
        list_n_in_uci=[1.9495, 1.4646, 1.4444, 2.0101, 1.303];
        max_days_go_uci=30;
        max_days_in_uci= 100; 
        window_slide_to_uci=0
        probability_not_vac[:-1]=find_probability_icu(data, start_date, end_date,list_p_in_uci,list_n_in_uci)[:-1].tolist()
        
        prob_uci_vac1_Pfizer,prob_uci_vac2_Pfizer=genere_list_minsal_denis(np.array(probability_not_vac),pt_factor=[0.524,0.909],minsal_factor=[0.99,0.984])
        prob_uci_vac1_Sinovac,prob_uci_vac2_Sinovac=genere_list_minsal_denis(np.array(probability_not_vac),pt_factor=[0.16,0.636],minsal_factor=[0.43,0.90])
        
        
        probability_not_vac_b117=probability_not_vac.copy()
        aplha=2.03
        probability_not_vac_b117[:-1]=(np.array(probability_not_vac_b117[:-1])*aplha).tolist()
        prob_uci_vac1_b117_Pfizer,prob_uci_vac2_b117_Pfizer=genere_list_minsal_denis(np.array(probability_not_vac_b117),pt_factor=[0.524,0.909],minsal_factor=[0.99,0.984])
        prob_uci_vac1_b117_Sinovac,prob_uci_vac2_b117_Sinovac=genere_list_minsal_denis(np.array(probability_not_vac_b117),pt_factor=[0.16,0.636],minsal_factor=[0.43,0.90])
        
        probability_not_vac_new_variant=[0.0530, 0.1117, 0.089, 0.0115, 0.072]
        prob_uci_vac1_new_variant_Pfizer,prob_uci_vac2_new_variant_Pfizer=genere_list_minsal_denis(np.array(probability_not_vac_new_variant),pt_factor=[0.524,0.909],minsal_factor=[0.99,0.984])
        prob_uci_vac1_new_variant_Sinovac,prob_uci_vac2_new_variant_Sinovac=genere_list_minsal_denis(np.array(probability_not_vac_new_variant),pt_factor=[0.16,0.636],minsal_factor=[0.43,0.90])
        
        #make not circulations
        p_shares_new_variant=[[0,0, 0.0, 0.269, 0.5, 0.739, 0.643],
                            [0,0, 0.0, 0.357, 0.517, 0.522, 0.833],
                            [0,0, 0.0, 0.462, 0.55, 0.818, 0.818],
                            [0,0, 0.057, 0.418, 0.465, 0.636, 0.62],
                            [0,0, 0.0, 0.333, 0.333, 0.571, 0.75]]
        ID_list_day_new_variant=[260,289,320,348,379,409,440]
        p_shares_b117=[[0,0, 0.062, 0.115, 0.096, 0.0, 0.0],
                        [0,0, 0.0, 0.0, 0.069, 0.0, 0.0],
                        [0,0, 0.0, 0.308, 0.1, 0.0, 0.0],
                        [0,0, 0.057, 0.114, 0.088, 0.029, 0.014],
                        [0,0, 0.0, 0.0, 0.167, 0.0, 0.0]]
        ID_list_day_b117=[400, 405, 406, 407, 412, 425, 436]#[260,289,320,348,379,409,440]
        
        p_shares_new_variant=np.zeros((5,len(ID_list_day_new_variant))).tolist()
        p_shares_b117=np.zeros((5,len(ID_list_day_new_variant))).tolist()
        
        shift = True
        
        uci=call_icu_simulation_2020(data,params,
                            save_data=False,
                            update_data=False)
        
        result_dict={}
        CONFIDENCE = 0.95     
        ERROR = 5
        N_INICIAL = 100          
        
        print("Nivel de confianza	: {}".format(CONFIDENCE))
        print("Error			: {}".format(ERROR))
        print("Replicas Iniciales	: {}".format(N_INICIAL))			 
        print("--------------------")
        for key,item in data['groupID_to_group'].items():
            print(key,item)
            ID_grupo_etario=key;
            p_to_uci=list_p_to_uci[ID_grupo_etario];
            n_to_uci=list_n_to_uci[ID_grupo_etario];
            p_to_uci_2021=list_p_to_uci_2021[ID_grupo_etario];
            n_to_uci_2021=list_n_to_uci_2021[ID_grupo_etario];
            ID_day_change_time;
            p_in_uci=list_p_in_uci[ID_grupo_etario];
            n_in_uci=list_n_in_uci[ID_grupo_etario];
            shift = True
            camas_uci_group=  Camas_UCI_new_variant(data, 
                                          ID_grupo_etario, 
                                          increase_vac1_Pfizer[ID_grupo_etario],increase_vac2_Pfizer[ID_grupo_etario],
                                          increase_vac1_Sinovac[ID_grupo_etario],increase_vac2_Sinovac[ID_grupo_etario],
                                          prob_uci_vac1_Pfizer[ID_grupo_etario],prob_uci_vac2_Pfizer[ID_grupo_etario],
                                          prob_uci_vac1_Sinovac[ID_grupo_etario],prob_uci_vac2_Sinovac[ID_grupo_etario],
                                          prob_uci_vac1_b117_Pfizer[ID_grupo_etario],prob_uci_vac2_b117_Pfizer[ID_grupo_etario],
                                          prob_uci_vac1_b117_Sinovac[ID_grupo_etario],prob_uci_vac2_b117_Sinovac[ID_grupo_etario],
                                          prob_uci_vac1_new_variant_Pfizer[ID_grupo_etario],prob_uci_vac2_new_variant_Pfizer[ID_grupo_etario],
                                          prob_uci_vac1_new_variant_Sinovac[ID_grupo_etario],prob_uci_vac2_new_variant_Sinovac[ID_grupo_etario],
                                          p_shares_new_variant[ID_grupo_etario],
                                          ID_list_day_new_variant,
                                          p_shares_b117[ID_grupo_etario],
                                          ID_list_day_b117,
                                          probability_not_vac[ID_grupo_etario],
                                          probability_not_vac_b117[ID_grupo_etario],
                                          probability_not_vac_new_variant[ID_grupo_etario],
                                          p_to_uci,
                                          n_to_uci,
                                          p_to_uci_2021,
                                          n_to_uci_2021,
                                          ID_day_change_time,
                                          p_in_uci,
                                          n_in_uci)
            mean_reasult_group=camas_uci_group.ICU_Simulations_camas_vac(N_INICIAL,
                                                                              CONFIDENCE,
                                                                              ERROR,
                                                                              start_date='2020-07-20',
                                                                              end_date='2021-05-15',shift= shift)
            camas_uci_group.plot_uci_pred_2020(data, uci, W=29, pond=1, start_date='2020-07-01',end_date='2021-05-15', infected=False)
            
            result_dict[item]=camas_uci_group
        save_data_icu_simulation_montecarlos_2020(result_dict)
    
    result_dict=open_data_simulation_montecarlos_2020()
    return result_dict
    



def save_data_icu_simulation_2020(uci):
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]
    if path_projet[-1]!='/':
        path_projet+='/'
    path_projet+='ICU_Simulations/'
    path_data='Data/Output/ICU/'
    
    today= datetime.now().strftime('%Y-%m-%d')
    print(f"Update the file: {today}")
    path=path_projet+path_data+"icu_simulation_2020.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    
    dict_icu={'uci':uci.tolist(),
            'last_update':today
    }
    with open(path, "w") as out:
        json.dump(dict_icu,out)
        
        
        
def save_data_icu_simulation_montecarlos_2020(result_dict):
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]
    if path_projet[-1]!='/':
        path_projet+='/'
    path_projet+='ICU_Simulations/'
    path_data='Data/Output/ICU/'
    
    today= datetime.now().strftime('%Y-%m-%d')
    print(f"Update the file: {today}")
    path=path_projet+path_data+"icu_simulation_montecarlos_2020.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    result_dict_aux={'last_update':today}
    
    for key, item in result_dict.items():
        result_dict_aux[key]=item.uci_beds.tolist()
    
    
    with open(path, "w") as out:
        json.dump(result_dict_aux,out)
    

def open_data_simulation_montecarlos_2020():
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]
    if path_projet[-1]!='/':
        path_projet+='/'
    path_projet+='ICU_Simulations/'
    path_data='Data/Output/ICU/'
    path=path_projet+path_data+"icu_simulation_montecarlos_2020.json"
    with open(path) as f:
        result_dict_aux=json.load(f)
    
    for key, item in result_dict_aux.items():
        if key!='last_update':
            result_dict_aux[key]=np.array(item)
    

    
    last_update=result_dict_aux['last_update']
    print(f"Update the file: {last_update}")
    
    return result_dict_aux
    
    
    
    



def open_data_simulation_2020():
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]
    if path_projet[-1]!='/':
        path_projet+='/'
    path_projet+='ICU_Simulations/'
    path_data='Data/Output/ICU/'
    path=path_projet+path_data+"icu_simulation_2020.json"
    with open(path) as f:
        dict_icu=json.load(f)
        
    uci=np.array(dict_icu['uci'])

    
    last_update=dict_icu['last_update']
    print(f"Update the file: {last_update}")
    
    return uci

#strart parameters
params={
    'increase_vac1_Pfizer':[[0.0, 0.0, 0.524, 0.524, 0.524, 0.524],
     [0.0, 0.0, 0.524, 0.524, 0.524, 0.524],
     [0.0, 0.0, 0.524, 0.524, 0.524, 0.524],
     [0.0, 0.0, 0.524, 0.524, 0.524, 0.524],
     [0.0, 0.0, 0.524, 0.524, 0.524, 0.524]],
    'increase_vac2_Pfizer': [[0.524, 0.524, 0.9, 0.9, 0.9, 0.9],
     [0.524, 0.524, 0.9, 0.9, 0.9, 0.9],
     [0.524, 0.524, 0.9, 0.9, 0.9, 0.9],
     [0.524, 0.524, 0.9, 0.9, 0.9, 0.9],
     [0.524, 0.524, 0.9, 0.9, 0.9, 0.9]],
    'increase_vac1_Sinovac':[[0.0, 0.0, 0.16, 0.16, 0.16, 0.16],
     [0.0, 0.0, 0.16, 0.16, 0.16, 0.16],
     [0.0, 0.0, 0.16, 0.16, 0.16, 0.16],
     [0.0, 0.0, 0.16, 0.16, 0.16, 0.16],
     [0.0, 0.0, 0.16, 0.16, 0.16, 0.16]],
    'increase_vac2_Sinovac': [[0.16, 0.16, 0.636, 0.636, 0.636, 0.636],
     [0.16, 0.16, 0.636, 0.636, 0.636, 0.636],
     [0.16, 0.16, 0.636, 0.636, 0.636, 0.636],
     [0.16, 0.16, 0.636, 0.636, 0.636, 0.636],
     [0.16, 0.16, 0.636, 0.636, 0.636, 0.636]],
    'probability_not_vac':[0.011, 0.029, 0.081, 0.0032, 0.083],
    'alpha':2.03,
    'probability_not_vac_new_variant':[0.0530, 0.1117, 0.089, 0.0115, 0.072],
    'start_date':'2020-05-01',
    'end_date':'2020-12-31',
    'list_p_to_uci':[0.463, 0.2139, 0.1494, 0.3976, 0.2466],
    'list_n_to_uci':[8.8889, 2.6667, 2.2222, 5.1111, 3.6667],
    'list_p_to_uci_2021':[0.699, 0.402, 0.365, 0.508, 0.7  ],
    'list_n_to_uci_2021':[21.798 ,  6.2727,  6.5657,  9.202 , 22.0909],
    'list_p_in_uci':[0.0699, 0.0619, 0.0515, 0.0739, 0.0412],
    'list_n_in_uci':[1.9495, 1.4646, 1.4444, 2.0101, 1.303],
    'day_change_time':'2021-01-01',
    'max_days_go_uci':30,
    'max_days_in_uci':100,
    'shift': True,
    'window':29,
}
    
    

if __name__ == "__main__":
    #run experiment 
    data = read_data()
    uci= call_icu_simulation_2020(data,params,
                        save_data=False,
                        update_data=False)
    plot_uci_pred_sns(data, uci, W=29, pond=1, start_date='2020-07-01', end_date='2021-05-15', infected=False)

    result_dict=call_icu_MonteCarlos_simulation_2020(data,params,
                            save_data=False,
                            update_data=False)
    """
    plot_uci_pred_final_model_2020(data, uci,
                            result_dict,
                            W=29, pond=1,
                            start_date='2020-07-01',end_date='2021-05-15',
                            probability_not_vac_new_variant=None)
    """
    
