#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 19 21:53:04 2022

@author: ineira
"""
import os 
import sys
import json

path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]
if path_projet[-1]!='/':
    path_projet+='/'
path_projet+='ICU_Simulations/'
path_EDA='Code/EDA'

module_path=path_projet+path_EDA
if module_path not in sys.path:
    sys.path.append(module_path)

from circulation_variants import plot_circulation_variants_Uchile

def call_vocs_circulation(update_data=False):
    """
    Call 

    Parameters
    ----------
    update_data : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    circulation_variants_month : TYPE
        DESCRIPTION.

    """
    if update_data: 
        plot_circulation_variants_Uchile(save_image=False, save_table=True)
    
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]
    if path_projet[-1]!='/':
        path_projet+='/'
    path_projet+='ICU_Simulations/'
    path_circulation_variants_month=path_projet+"Data/Output/Vocs/circulation_variants_month.json"
    
    
    with open(path_circulation_variants_month) as f:
        circulation_variants_month=json.load(f)
    
    return circulation_variants_month

if __name__=='__main__':
    circulation_variants_month=call_vocs_circulation(update_data=False)
    for key, item in circulation_variants_month.items():
        print(f"{key} \n")
        print(item)
        print('\n')