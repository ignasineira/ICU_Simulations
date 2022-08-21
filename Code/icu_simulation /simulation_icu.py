#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 29 18:16:33 2021

@author: ignas

codigo: version  9 

Nota:
Ahora los tiempo es uci distribuyen geometrico
y pueden variar en el tiempo

orden de los grupos etarios:

['40-49', '50-59', '60-69', '<=39', '>=70']
CONFIRMACION PCR desde INICIO DE LOS SINTOMAS': 

[8, 7, 8, 7, 4]

"""

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


from data_processing import read_data,prepare_population
import git_hub_commit
from fit_probability_to_UCI import find_probability_UCI



import seaborn as sns ## These Three lines are necessary for Seaborn to work   
import matplotlib.pyplot as plt 
from matplotlib.dates import date2num
import matplotlib.dates as mdates




data = read_data()
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
                    window=29, 
                    probability_not_vac = probability_not_vac,
                    probability_not_vac_b117 = probability_not_vac_b117,
                    probability_not_vac_new_variant = probability_not_vac_new_variant,
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
                    shift=shift)