#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 15 22:05:16 2022

@author: ineira
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 

def fit_dead_2020(x,a):
     return a*x
 
    
def MSE(y_pred,y):
     # Mean Squared Error
     return np.square(np.subtract(y_pred,y)).mean() 

def main_fit_fatality_2020(data, dead,W=29):
    """
    1. give a range of time fit the curve 
    2. evaluate the MSE in beteween 01-07 to 31-12
    3. save result

    """
    list_start_date=[f"2020-0{item}-01" if item <10 else f"2020-{item}-01" for item in range(5,13)]
    start_date='2020-07-01';
    end_date='2020-12-31'
    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    
    groupID_to_group = data['groupID_to_group']
    frames=[]
    
    for g in range(len(groupID_to_group)):
        print(groupID_to_group[g])
        for date in list_start_date:
            dateID_date = data['date_to_dateID'][np.datetime64(date+"T00:00:00.000000000")]
            y=data['dead'][g,dateID_date:dateID_end_date+1]
            x= dead[g,dateID_date-(W-1):dateID_end_date+1-(W-1)]
            
            fit = curve_fit(fit_dead_2020, x, y)
            
            
            y_test=data['dead'][g,dateID_start_date:dateID_end_date+1]
            x_test= dead[g,dateID_start_date-(W-1):dateID_end_date+1-(W-1)]
            y_pred=x_test*fit[0][0]
            mse= MSE(y_pred,y_test)
            
            print(f"For {groupID_to_group[g]} and {pd.to_datetime(date).strftime('%B')} the fatality rate is: {fit[0][0]}")
            
            #save values
            frames.append([
                date,
                pd.to_datetime(date).strftime('%B'),
                groupID_to_group[g],
                fit[0][0],
                mse]
                )
    df= pd.DataFrame(frames,columns=['start_date','month',  'Grupo de edad', 'fatality_rate','mse'])
    
    
    
    list_order_group=['<=39','40-49', '50-59', '60-69', '>=70']
    
    mse=df.pivot(index="Grupo de edad",columns='month', values='mse').reindex(['<=39','40-49', '50-59', '60-69', '>=70'])
    fatality_rate= df.pivot(index="Grupo de edad",columns='month', values='fatality_rate').reindex(['<=39','40-49', '50-59', '60-69', '>=70'])
    order_column=[pd.to_datetime(item).strftime('%B')for item in list_start_date]
    fatality_rate=fatality_rate[order_column]
    mse=mse[order_column]
    
    save=True
    if save:
        mse_csv_file="mse_fatality.csv"
        fatality_rate_csv_file="fatality_rate.csv"
        experiment_outputs_csv_file="experiment_outputs_fatality.csv"
        path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
        path_data='Data/Output/Fatality/'
        
        list_name_file=[mse_csv_file,fatality_rate_csv_file,experiment_outputs_csv_file]
        list_file=[mse,fatality_rate,df]
        for name_file,file in zip(list_name_file,list_file):
            csv_save_file = path_projet+path_data+name_file
            os.makedirs(os.path.dirname(csv_save_file), exist_ok=True)
            file.round(3).to_csv(csv_save_file)
        
        today= datetime.now().strftime('%Y-%m-%d')
        print(f"Update the file: {today}")
        last_update_file=path_projet+path_data+"last_update_file_fatality.json"
        os.makedirs(os.path.dirname(last_update_file), exist_ok=True)
        with open(last_update_file, "w") as out:
            json.dump({'last_update':today},out)
    return df,mse,fatality_rate 
    
#df,mse,fatality_rate= main_fit_fatality_2020(data, dead,W=29)
    
    
    
    
    
    
    
    


