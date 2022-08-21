#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 19:09:21 2022

@author: ineira
"""
import os
import pandas as pd 
import numpy as np


def save_fatality_data(data,dict_main_dead,W=29):
    date='2021-01-01'
    start_date='2021-01-01';
    end_date='2021-05-15'
    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    
    groupID_to_group = data['groupID_to_group']
    
    list_df=[]
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
            if key=='total':
                total=item[g,dateID_date-(W-1):dateID_end_date+1-(W-1)]
        icu_gamma=np.array(variantes).sum(axis=0)
        icu_original=np.array(no_variante)
        total=np.array(total)
        #X=np.hstack((dead,icu_original,icu_original))
        df = pd.DataFrame({
           'datetimes': pd.date_range(
               start=start_date,
               end=end_date,
               freq='d',     # <--- try 'H''3h', '6h', '12h' if you want
               #closed='left'
           )
         })
        df['datetimes']=df.datetimes.dt.strftime('%Y-%m-%d')
        df['Grupo de edad']=groupID_to_group[g]
        df['dead']=dead
        df['icu_total']=total
        df['icu_original']=icu_original
        df['icu_gamma']=icu_gamma
        list_df.append(df)
        
        
    df_mg=pd.concat(list_df, ignore_index=True)
    
    
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Output/Fatality/'
    csv_file='timeseries_dead.csv'
    
    csv_save_file = path_projet+path_data+csv_file
    os.makedirs(os.path.dirname(csv_save_file), exist_ok=True)
    
    df_mg.to_csv(csv_save_file,index=False)

 
 