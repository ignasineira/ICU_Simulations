#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 16 15:58:19 2022

@author: ineira
"""
import requests
import io
import os
import json
from urllib.request import urlretrieve #open url

import pandas as pd
import numpy as np

pd.options.display.max_columns = None
pd.options.display.max_rows = None

from scipy.optimize import curve_fit 

from data_processing import prepare_producto_10,prepare_producto_16
from datetime import datetime,timedelta
pd.set_option('display.float_format', lambda x: '%.9f' % x)


from matplotlib import cm
cmap = cm.get_cmap('Spectral')
import matplotlib.pyplot as plt # plotting data 
import matplotlib.dates as mdates
from matplotlib.dates import date2num

import seaborn as sns ## These Three lines are necessary for Seaborn to work   


def main_fit_fatality_2020(data,dict_main_dead,W=29):
    """
    1. give a range of time fit the curve 
    2. evaluate the MSE in beteween 01-07 to 31-12
    3. save result
    
    
    """
    dead=dict_main_dead['dict_dead_variant']
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
            x= dead['Not variant'][g,dateID_date-(W-1):dateID_end_date+1-(W-1)]
            
            fit = curve_fit(fit_dead_2020, x, y)
            
            
            y_test=data['dead'][g,dateID_start_date:dateID_end_date+1]
            x_test= dead['Not variant'][g,dateID_start_date-(W-1):dateID_end_date+1-(W-1)]
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



#df_2020,mse,fatality_rate = main_fit_fatality_2020(data,dict_main_dead,W=29)

def main_fit_fatality_2021(data,dict_main_dead,fatality_rate, month=['May','July'],W=29):
    """
    1. give a range of time fit the curve 
    2. evaluate the MSE in beteween 01-07 to 31-12
    3. save result
    
    
    """
    dead=dict_main_dead['dict_dead_variant']
    list_start_date=[f"2020-0{item}-01" if item <10 else f"2020-{item}-01" for item in range(5,13)]
    date='2021-01-01'
    start_date='2020-07-01';
    end_date='2021-05-15'
    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    
    groupID_to_group = data['groupID_to_group']
    
    lower_bound=fatality_rate.to_dict()['July']
    upper_bound=fatality_rate.to_dict()['July']


    frames=[]
    
    dead_pred=[]
    for g in range(len(groupID_to_group)):
        group_name=groupID_to_group[g]
        print(group_name)
        
        dateID_date = data['date_to_dateID'][np.datetime64(date+"T00:00:00.000000000")]
        y=data['dead'][g,dateID_date:dateID_end_date+1]
        #x= dead['Not variant'][g,dateID_date-(W-1):dateID_end_date+1-(W-1)]
        x=np.stack([item[g,dateID_date-(W-1):dateID_end_date+1-(W-1)] for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
        
        fit = curve_fit(fit_dead_2021, x, y,bounds=([lower_bound[group_name],0,0],  #lower_bound[group_name]*0.9, lower_bound[group_name]*0.9
                                                    [upper_bound[group_name]*1.000001, 10,10]))
        
        
        y_test=data['dead'][g,dateID_start_date:dateID_end_date+1]
        x_test= np.stack([item[g,dateID_start_date-(W-1):dateID_end_date+1-(W-1)] for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
      
        y_pred=(x_test*np.array(fit[0]).reshape((3, -1))).sum(axis=0)
        mse= MSE(y_pred,y_test)
        
        #save data
        x_test= np.stack([item[g,:] for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
        y_pred=(x_test*np.array(fit[0]).reshape((3, -1))).sum(axis=0)
        dead_pred.append(y_pred)
        #print(f"For {groupID_to_group[g]} and {pd.to_datetime(date).strftime('%B')} the fatality rate is: {fit[0][0]}")
        
        #save values
        frames.append([
            start_date,
            groupID_to_group[g],
            fit[0][0],fit[0][1],fit[0][2],
            mse]
            )
    df= pd.DataFrame(frames,columns=['start_date',  'Grupo de edad',"f_no_variant","f_variant","f_b117",'mse'])
    dead_pred=np.stack(dead_pred,axis=0)
    plot_dead_pred_sns(data, dead_pred, W=29, prob_dead=[1,1,1,1,1], start_date='2020-07-01',end_date='2021-05-15', infected=False)
    
    list_order_group=['<=39','40-49', '50-59', '60-69', '>=70']


    return df



def main_fit_fatality_2021_v2_time_window(data,dict_main_dead,fatality_rate, month=['May','July'],W=29,end_date='2021-05-15' ):
    
    date_end = datetime.strptime(end_date, '%Y-%m-%d').date()
    
    
    list_end_date=[ (date_end-timedelta(i*15)).strftime('%Y-%m-%d')   for i in range(0,7)]
    
    df_list=[]
    for date_end in list_end_date:
        df=main_fit_fatality_2021_v2(data,dict_main_dead,fatality_rate, month=['May','July'],W=29,end_date=date_end )
        df_list.append(df)
        
    return list_end_date, df_list
  
#list_end_date, df_list=   main_fit_fatality_2021_v2_time_window(data,dict_main_dead,fatality_rate, month=['May','July'],W=29,end_date='2021-05-15' )

def main_fit_fatality_2021_v2(data,dict_main_dead,fatality_rate, month=['May','July'],W=29,end_date='2021-05-15' ):
    """
    1. give a range of time fit the curve 
    2. evaluate the MSE in beteween 01-07 to 31-12
    3. save result
    
    
    """
    dead=dict_main_dead['dict_dead_variant']
    list_start_date=[f"2020-0{item}-01" if item <10 else f"2020-{item}-01" for item in range(5,13)]
    date='2021-01-01'
    start_date='2020-07-01';
    
    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    
    groupID_to_group = data['groupID_to_group']
    
    lower_bound=fatality_rate.to_dict()['July']
    upper_bound=fatality_rate.to_dict()['July']


    frames=[]
    
    dead_pred_model_2021=[]
    dead_pred_model_2020=[]
    for g in range(len(groupID_to_group)):
        group_name=groupID_to_group[g]
        print(group_name)
        
        dateID_date = data['date_to_dateID'][np.datetime64(date+"T00:00:00.000000000")]
        y=data['dead'][g,dateID_date:dateID_end_date+1]
        #x= dead['Not variant'][g,dateID_date-(W-1):dateID_end_date+1-(W-1)]
        variantes=[]
        no_variante=[]
        for key,item in dict_main_dead['dict_dead_variant'].items():
            if key== 'Not variant':
                no_variante=item[g,dateID_date-(W-1):dateID_end_date+1-(W-1)]
            if key in ['variant', 'b117']:
                variantes.append(item[g,dateID_date-(W-1):dateID_end_date+1-(W-1)])
                
        array_variantes=np.array(variantes).sum(axis=0)
        array_no_variantes=np.array(no_variante)
                
        x=np.stack([array_no_variantes,array_variantes],axis=0)
        
        fit = curve_fit(fit_dead_2021_v2, x, y,bounds=([lower_bound[group_name],0],  #lower_bound[group_name]*0.9, lower_bound[group_name]*0.9
                                                    [upper_bound[group_name]*1.000001,10]))
        
        
        y_test=data['dead'][g,dateID_start_date:dateID_end_date+1]
        x_test= np.stack([item[g,dateID_start_date-(W-1):dateID_end_date+1-(W-1)] for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
        params_2021=np.array([fit[0][0],fit[0][1],fit[0][1]])
        y_pred_model_2021=(x_test*np.array(params_2021).reshape((3, -1))).sum(axis=0)
        mse_model_2021= MSE(y_pred_model_2021,y_test)
        
        params_2020=np.array([fit[0][0],fit[0][0],fit[0][0]])
        y_pred_model_2020=(x_test*np.array(params_2020).reshape((3, -1))).sum(axis=0)
        mse_model_2020= MSE(y_pred_model_2020,y_test)
        
        #save data all times series
        x_test= np.stack([item[g,:] for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
        y_pred_model_2021=(x_test*np.array(params_2021).reshape((3, -1))).sum(axis=0)
        dead_pred_model_2021.append(y_pred_model_2021)
        
        y_pred_model_2020=(x_test*np.array(params_2020).reshape((3, -1))).sum(axis=0)
        dead_pred_model_2020.append(y_pred_model_2020)
        
        
        #print(f"For {groupID_to_group[g]} and {pd.to_datetime(date).strftime('%B')} the fatality rate is: {fit[0][0]}")
        
        #save values
        frames.append([
            start_date,
            groupID_to_group[g],
            fit[0][0],fit[0][1],
            mse_model_2021,mse_model_2020]
            )
    df= pd.DataFrame(frames,columns=['start_date',  'Grupo de edad',"f_no_variant","f_variants",'mse'])
    dead_pred_2021=np.stack(dead_pred_model_2021,axis=0)
    dead_pred_2020=np.stack(dead_pred_model_2020,axis=0)

    plot_dead_pred_sns(data, dead_pred_2021, W=29, prob_dead=[1,1,1,1,1], start_date='2020-07-01',end_date=end_date, infected=False)
    
    list_order_group=['<=39','40-49', '50-59', '60-69', '>=70']


    return df, dead_pred_2021, dead_pred_2020

def main_fit_fatality_2021_v3(data,dict_main_dead,fatality_rate, month=['May','July'],W=29,end_date= ['2021-03-25','2021-03-12','2021-02-18','2021-05-14','2021-02-09']
):
    """
    given a moving windows
    1. give a range of time fit the curve 
    2. evaluate the MSE in beteween 01-07 to 31-12
    3. save result
    
    
    """
    dead=dict_main_dead['dict_dead_variant']
    date='2021-01-01'
    start_date='2020-07-01';
    
    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    #dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    
    groupID_to_group = data['groupID_to_group']
    
    lower_bound=fatality_rate.to_dict()['July']
    upper_bound=fatality_rate.to_dict()['July']


    frames=[]
    
    dead_pred_model_2021=[]
    dead_pred_model_2020=[]
    for g in range(len(groupID_to_group)):
        group_name=groupID_to_group[g]
        print(group_name)
        dateID_end_date = data['date_to_dateID'][np.datetime64(end_date[g]+"T00:00:00.000000000")]
        dateID_date = data['date_to_dateID'][np.datetime64(date+"T00:00:00.000000000")]
        y=data['dead'][g,dateID_date:dateID_end_date+1]
        #x= dead['Not variant'][g,dateID_date-(W-1):dateID_end_date+1-(W-1)]
        variantes=[]
        no_variante=[]
        for key,item in dict_main_dead['dict_dead_variant'].items():
            if key== 'Not variant':
                no_variante=item[g,dateID_date-(W-1):dateID_end_date+1-(W-1)]
            if key in ['variant', 'b117']:
                variantes.append(item[g,dateID_date-(W-1):dateID_end_date+1-(W-1)])
                
        array_variantes=np.array(variantes).sum(axis=0)
        array_no_variantes=np.array(no_variante)
                
        x=np.stack([array_no_variantes,array_variantes],axis=0)
        
        fit = curve_fit(fit_dead_2021_v2, x, y,bounds=([lower_bound[group_name],0],  #lower_bound[group_name]*0.9, lower_bound[group_name]*0.9
                                                    [upper_bound[group_name]*1.000001,10]))
        
        
        
        y_test=data['dead'][g,dateID_start_date:dateID_end_date+1]
        x_test= np.stack([item[g,dateID_start_date-(W-1):dateID_end_date+1-(W-1)] for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
        params_2021=np.array([fit[0][0],fit[0][1],fit[0][1]])
        y_pred_model_2021=(x_test*np.array(params_2021).reshape((3, -1))).sum(axis=0)
        mse_model_2021= MSE(y_pred_model_2021,y_test)
        
        params_2020=np.array([fit[0][0],fit[0][0],fit[0][0]])
        y_pred_model_2020=(x_test*np.array(params_2020).reshape((3, -1))).sum(axis=0)
        mse_model_2020= MSE(y_pred_model_2020,y_test)
        
        #save data all times series
        x_test= np.stack([item[g,:] for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
        y_pred_model_2021=(x_test*np.array(params_2021).reshape((3, -1))).sum(axis=0)
        dead_pred_model_2021.append(y_pred_model_2021)
        
        y_pred_model_2020=(x_test*np.array(params_2020).reshape((3, -1))).sum(axis=0)
        dead_pred_model_2020.append(y_pred_model_2020)
        
        
        y_test=data['dead'][g,dateID_start_date:dateID_end_date+1]
        x_test= np.stack([item[g,dateID_start_date-(W-1):dateID_end_date+1-(W-1)] for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
        params=np.array([fit[0][0],fit[0][1],fit[0][1]])
        y_pred=(x_test*np.array(params).reshape((3, -1))).sum(axis=0)
        mse= MSE(y_pred,y_test)
        
        #save values
        frames.append([
            end_date[g],
            groupID_to_group[g],
            fit[0][0],fit[0][1],
            mse_model_2021,mse_model_2020]
            )
    df= pd.DataFrame(frames,columns=['end_date',  'Grupo de edad',"f_no_variant","f_variants",'mse 2021','mse 2020'])
    dead_pred_2021=np.stack(dead_pred_model_2021,axis=0)
    dead_pred_2020=np.stack(dead_pred_model_2020,axis=0)
    plot_dead_pred_sns_moving_windows(data, dead_pred_2021,dead_pred_2020, W=29, prob_dead=[1,1,1,1,1], start_date='2020-07-01',end_date=end_date, infected=False)
    
    list_order_group=['<=39','40-49', '50-59', '60-69', '>=70']
    

    return df, dead_pred_2021, dead_pred_2020

# df, dead_pred_2021, dead_pred_2020=main_fit_fatality_2021_v3(data,dict_main_dead,fatality_rate, month=['May','July'],W=29,end_date= ['2021-03-25','2021-03-12','2021-02-18','2021-05-14','2021-02-09'])
def main_fit_fatality_2020_v4(data,dict_main_dead,W=29):
    """
    1. give a range of time fit the curve 
    2. evaluate the MSE in beteween 01-07 to 31-12
    3. save result
    
    
    """
    dead=dict_main_dead['dict_dead_variant']
    list_start_date=[f"2020-0{item}-01" if item <10 else f"2020-{item}-01" for item in range(5,13)]
    start_date='2020-07-01';
    end_date='2020-12-31'
    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    
    groupID_to_group = data['groupID_to_group']


    frames=[]
    
   
    for date in list_start_date:
        dateID_date = data['date_to_dateID'][np.datetime64(date+"T00:00:00.000000000")]
        y=data['dead'][:4,dateID_date:dateID_end_date+1].sum(axis=0)
        x= dead['Not variant'][:4,dateID_date-(W-1):dateID_end_date+1-(W-1)].sum(axis=0)
        
        fit = curve_fit(fit_dead_2020, x, y)
        
        
        y_test=data['dead'][:4,dateID_start_date:dateID_end_date+1].sum(axis=0)
        x_test= dead['Not variant'][:4,dateID_start_date-(W-1):dateID_end_date+1-(W-1)].sum(axis=0)
        y_pred=x_test*fit[0][0]
        mse= MSE(y_pred,y_test)
        
        print(f"For {pd.to_datetime(date).strftime('%B')} the fatality rate is: {fit[0][0]}")
        
        #save values
        frames.append([
            date,
            pd.to_datetime(date).strftime('%B'),
            'total',
            fit[0][0],
            mse]
            )
    df= pd.DataFrame(frames,columns=['start_date','month',  'Grupo de edad', 'fatality_rate','mse'])
    

    
    mse=df.pivot(index="Grupo de edad",columns='month', values='mse')
    fatality_rate= df.pivot(index="Grupo de edad",columns='month', values='fatality_rate')
    order_column=[pd.to_datetime(item).strftime('%B')for item in list_start_date]
    fatality_rate=fatality_rate[order_column]
    mse=mse[order_column]
    

    return df,mse,fatality_rate 

#df,mse,fatality_rate= main_fit_fatality_2020_v4(data,dict_main_dead,W=29)
#df1, dead_pred_2021, dead_pred_2020=main_fit_fatality_2021_v4(data,dict_main_dead,fatality_rate, month=['May','July'],W=29,end_date_list= ['2021-03-25','2021-03-12','2021-02-18','2021-05-14','2021-02-09'])

def main_fit_fatality_2021_v4(data,dict_main_dead,fatality_rate, month=['May','July'],W=29,end_date_list= ['2021-03-25','2021-03-12','2021-02-18','2021-05-14','2021-02-09']
):
    """
    ucis a muerte agregados
    given a moving windows
    1. give a range of time fit the curve 
    2. evaluate the MSE in beteween 01-07 to 31-12
    3. save result
    
    
    """
    dead=dict_main_dead['dict_dead_variant']
    date='2021-01-01'
    start_date='2020-07-01';
    
    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    #dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    
    groupID_to_group = data['groupID_to_group']
    
    lower_bound=fatality_rate.to_dict()['July']
    upper_bound=fatality_rate.to_dict()['July']
    group_name='total'

    frames=[]
    
    dead_pred_model_2021=[]
    dead_pred_model_2020=[]
    for end_date in end_date_list:
        dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
        dateID_date = data['date_to_dateID'][np.datetime64(date+"T00:00:00.000000000")]
        y=data['dead'][:4,dateID_date:dateID_end_date+1].sum(axis=0)
        #x= dead['Not variant'][g,dateID_date-(W-1):dateID_end_date+1-(W-1)]
        variantes=[]
        no_variante=[]
        for key,item in dict_main_dead['dict_dead_variant'].items():
            if key== 'Not variant':
                no_variante=item[:4,dateID_date-(W-1):dateID_end_date+1-(W-1)]
            if key in ['variant', 'b117']:
                variantes.append(item[:4,dateID_date-(W-1):dateID_end_date+1-(W-1)])
                
        array_variantes=np.array(variantes).sum(axis=0).sum(axis=0)
        array_no_variantes=np.array(no_variante).sum(axis=0)
                
        x=np.stack([array_no_variantes,array_variantes],axis=0)
        
        fit = curve_fit(fit_dead_2021_v2, x, y,bounds=([lower_bound[group_name],0],  #lower_bound[group_name]*0.9, lower_bound[group_name]*0.9
                                                    [upper_bound[group_name]*1.000001,10]))
        
        
        
        y_test=data['dead'][:4,dateID_start_date:dateID_end_date+1].sum(axis=0)
        x_test= np.stack([item[:4,dateID_start_date-(W-1):dateID_end_date+1-(W-1)].sum(axis=0) for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
        params_2021=np.array([fit[0][0],fit[0][1],fit[0][1]])
        y_pred_model_2021=(x_test*np.array(params_2021).reshape((3, -1))).sum(axis=0)
        mse_model_2021= MSE(y_pred_model_2021,y_test)
        
        params_2020=np.array([fit[0][0],fit[0][0],fit[0][0]])
        y_pred_model_2020=(x_test*np.array(params_2020).reshape((3, -1))).sum(axis=0)
        mse_model_2020= MSE(y_pred_model_2020,y_test)
        
        #save data all times series
        x_test= np.stack([item[:4,:].sum(axis=0) for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
        y_pred_model_2021=(x_test*np.array(params_2021).reshape((3, -1))).sum(axis=0)
        dead_pred_model_2021.append(y_pred_model_2021)
        
        y_pred_model_2020=(x_test*np.array(params_2020).reshape((3, -1))).sum(axis=0)
        dead_pred_model_2020.append(y_pred_model_2020)
        
        
        y_test=data['dead'][:4,dateID_start_date:dateID_end_date+1].sum(axis=0)
        x_test= np.stack([item[:4,dateID_start_date-(W-1):dateID_end_date+1-(W-1)].sum(axis=0) for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
        params=np.array([fit[0][0],fit[0][1],fit[0][1]])
        y_pred=(x_test*np.array(params).reshape((3, -1))).sum(axis=0)
        mse= MSE(y_pred,y_test)
        
        #save values
        frames.append([
            end_date,
            'total',
            fit[0][0],fit[0][1],
            mse_model_2021,mse_model_2020]
            )
    df= pd.DataFrame(frames,columns=['end_date',  'Grupo de edad',"f_no_variant","f_variants",'mse 2021','mse 2020'])
    dead_pred_2021=np.stack(dead_pred_model_2021,axis=0)
    dead_pred_2020=np.stack(dead_pred_model_2020,axis=0)
    #plot_dead_pred_sns_moving_windows(data, dead_pred_2021,dead_pred_2020, W=29, prob_dead=[1,1,1,1,1], start_date='2020-07-01',end_date=end_date, infected=False)
    
    list_order_group=['<=39','40-49', '50-59', '60-69', '>=70']
    
    
    
    

    return df, dead_pred_2021, dead_pred_2020



def fit_dead_2020(x,a):
     return a*x
 
    
def MSE(y_pred,y):
     # Mean Squared Error
     return np.square(np.subtract(y_pred,y)).mean()
 
    
def fit_dead_2021(x,f_no_variant,f_variant,f_b117):
    """
    # es importate el orden
    Not variant
    variant
    b117

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    f_no_variant : TYPE
        DESCRIPTION.
    f_variant : TYPE
        DESCRIPTION.
    f_b117 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
   

    return x[0]*f_no_variant+x[1]*f_variant+x[2]*f_b117

def fit_dead_2021_v2(x,f_no_variant,f_variants):
    """
    # es importate el orden
    Not variant
    variant
    b117

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    f_no_variant : TYPE
        DESCRIPTION.
    f_variants : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
   

    return x[0]*f_no_variant+x[1]*f_variants
    


def plot_dead_pred_sns(data, dead, W=29, prob_dead=[1,1,1,1,1], start_date='2020-07-01',end_date='2021-05-15', infected=False):
    groupID_to_group = data['groupID_to_group']
    dateID_to_date = data['dateID_to_date']
    date_to_dateID = data["date_to_dateID"]
    #start_date='2020-10-01'  

    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    cols =['Grupo de edad', 'start_date','inf', 'dead', 'type']
    lst = []
    
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Image/Fatality/'
    path=path_projet+path_data+end_date+'/'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    
    for g in range(len(groupID_to_group)):
        for date in range(dateID_start_date,dateID_end_date+1): 
            info = [groupID_to_group[g],dateID_to_date[date],data['inf'][g,date]*prob_dead[g]]
            info1=info.copy()
            info2=info.copy()
            info3=info.copy()
            info1.extend([data['dead'][g,date], 'real'])
            info2.extend([dead[g,date-(W-1)]*prob_dead[g], 'predicted'])
            info3.extend([dead[g,date-(W-1)], 'Pacientes OUT UCI'])
            lst.append(info1)
            lst.append(info2)
            lst.append(info3)
    df_res = pd.DataFrame(lst, columns=cols)
    
    #palette = sns.cubehelix_palette(light=.7, n_colors=6)
    for i, (group_name, group_df) in enumerate(df_res.groupby(["Grupo de edad"])):
    #for i in range(2):
    
        #plt.subplots(figsize=(15, 9))
        # plot the group on these axes
            
        #fig, ax = plt.subplots()
        plt.subplots( figsize=(15, 9))
        ax=sns.lineplot(x="start_date", y="dead",
                                       hue="type",
                                       palette= 'bright'#'Set3'
                                       ,legend=True, data=group_df,sizes=((15, 9)))
        
        #ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred'])
        # set the title
        ax.set(xlim=[group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
        
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        
        ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,3)), 
               label="older adults",color="red", alpha=0.3)
        
        
        
        ax.axes.set_title("Timeseries dead for "+group_name ,fontsize=15)
        ax.set_ylabel('N° Deads', fontsize=10)
        ax.set_xlabel('Date', fontsize=10)
        #ax.legend(['Dead today hampel', 'Dead today predicted (factor: '+ str(prob_dead[i])+')','Pacientes OUT UCI'],loc='upper left')
        ax.legend(['Dead today hampel', 'Dead today predicted','Pacientes OUT UCI'],loc='upper left')

        #g.ax.legend(bbox_to_anchor=(1.01, 1.02), loc='upper left')
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        try: 
            ax.figure.savefig(path+'Image/Fatality/'+"Fatality_rate_time_window"+group_name+".png")
        except:
            ax.figure.savefig(path+"Fatality_rate_time_window"+group_name[:-2]+".png")
        plt.show()
        
        
def plot_dead_pred_sns_moving_windows(data, dead_pred_2021,dead_pred_2020, W=29, prob_dead=[1,1,1,1,1], start_date='2020-07-01',end_date=['2021-03-25','2021-03-12','2021-02-18','2021-05-14','2021-02-09'], infected=False):
    groupID_to_group = data['groupID_to_group']
    dateID_to_date = data['dateID_to_date']
    date_to_dateID = data["date_to_dateID"]
    #start_date='2020-10-01'  

    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    #dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    cols =['Grupo de edad', 'start_date','inf', 'dead', 'type']
    lst = []
    
    
    end_date_list_ID=[data['date_to_dateID'][np.datetime64(item+"T00:00:00.000000000")] for item in end_date]
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Image/Fatality/movable windows'
    path=path_projet+path_data+''+'/'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    
    for g in range(len(groupID_to_group)):
        for date in range(dateID_start_date,end_date_list_ID[g]+1):
            #if date<=end_date_list_ID[g]:
            info = [groupID_to_group[g],dateID_to_date[date],data['inf'][g,date]*prob_dead[g]]
            info1=info.copy()
            info2=info.copy()
            info3=info.copy()
            info1.extend([data['dead'][g,date], 'real'])
            info2.extend([dead_pred_2021[g,date-(W-1)], 'predicted_2021'])
            info3.extend([dead_pred_2020[g,date-(W-1)], 'predicted_2020'])
            lst.append(info1)
            lst.append(info2)
            lst.append(info3)
    df_res = pd.DataFrame(lst, columns=cols)
    
    #palette = sns.cubehelix_palette(light=.7, n_colors=6)
    for i, (group_name, group_df) in enumerate(df_res.groupby(["Grupo de edad"])):
    #for i in range(2):
    
        #plt.subplots(figsize=(15, 9))
        # plot the group on these axes
            
        #fig, ax = plt.subplots()
        plt.subplots( figsize=(15, 9))
        ax=sns.lineplot(x="start_date", y="dead",
                                       hue="type",
                                       palette= 'bright'#'Set3'
                                       ,legend=True, data=group_df,sizes=((15, 9)))
        
        #ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred'])
        # set the title
        ax.set(xlim=[group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
        
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        
        ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,3)), 
               label="older adults",color="red", alpha=0.3)
        
        
        
        ax.axes.set_title("Timeseries dead for "+group_name ,fontsize=15)
        ax.set_ylabel('N° Deads', fontsize=10)
        ax.set_xlabel('Date', fontsize=10)
        #ax.legend(['Dead today hampel', 'Dead today predicted (factor: '+ str(prob_dead[i])+')','Pacientes OUT UCI'],loc='upper left')
        ax.legend(['Dead today hampel', 'Dead today predicted model 2021','Dead today predicted model 2020'],loc='upper left')

        #g.ax.legend(bbox_to_anchor=(1.01, 1.02), loc='upper left')
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        try: 
            ax.figure.savefig(path+'Fatality/'+"Fatality_rate_time_window"+group_name+".png")
        except:
            ax.figure.savefig(path+"Fatality_rate_time_window"+group_name[:-2]+".png")
        plt.show()
        
        
def plot_dead_pred_sns_moving_windows_model_2021_v4(data, dead_pred_2021,dead_pred_2020, W=29, prob_dead=[1,1,1,1,1], start_date='2020-07-01',end_date_list=['2021-03-25','2021-03-12','2021-02-18','2021-05-14','2021-02-09'], infected=False):
    groupID_to_group = data['groupID_to_group']
    dateID_to_date = data['dateID_to_date']
    date_to_dateID = data["date_to_dateID"]
    #start_date='2020-10-01'  

    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    #dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    cols =['Grupo de edad', 'start_date','inf', 'dead', 'type']
    #lst = []
    
    
    end_date_list_ID=[data['date_to_dateID'][np.datetime64(item+"T00:00:00.000000000")] for item in end_date_list]
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Image/Fatality/movable windows group by all'
    path=path_projet+path_data+''+'/'
    os.makedirs(os.path.dirname(path), exist_ok=True)

    for end_date in range(len(end_date_list_ID)):
        lst = []
        for date in range(dateID_start_date,end_date_list_ID[end_date]+1):
            #if date<=end_date_list_ID[g]:
            info = ['total',dateID_to_date[date],data['inf'][:4,date].sum(axis=0)*prob_dead[0]]
            info1=info.copy()
            info2=info.copy()
            info3=info.copy()
            info1.extend([data['dead'][:4,date].sum(), 'real'])
            info2.extend([dead_pred_2021[end_date,date-(W-1)], 'predicted_2021'])#malo
            info3.extend([dead_pred_2020[end_date,date-(W-1)], 'predicted_2020'])#malo
            lst.append(info1)
            lst.append(info2)
            lst.append(info3)
        df_res = pd.DataFrame(lst, columns=cols)
        
    
        plt.subplots( figsize=(15, 9))
        ax=sns.lineplot(x="start_date", y="dead",
                                       hue="type",
                                       palette= 'bright'#'Set3'
                                       ,legend=True, data=df_res,sizes=((15, 9)))
        
        #ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred'])
        # set the title
        ax.set(xlim=[df_res["start_date"].min()+pd.DateOffset(-2),df_res["start_date"].max()+pd.DateOffset(2)])
        
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        
        ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,3)), 
               label="older adults",color="red", alpha=0.3)
        
        
        
        ax.axes.set_title("Timeseries dead for all the groups" ,fontsize=15)
        ax.set_ylabel('N° Deads', fontsize=10)
        ax.set_xlabel('Date', fontsize=10)
        #ax.legend(['Dead today hampel', 'Dead today predicted (factor: '+ str(prob_dead[i])+')','Pacientes OUT UCI'],loc='upper left')
        ax.legend(['Dead today hampel', 'Dead today predicted model 2021','Dead today predicted model 2020'],loc='upper left')
    
        #g.ax.legend(bbox_to_anchor=(1.01, 1.02), loc='upper left')
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
    
        ax.figure.savefig(path+"Fatality_rate_time_window"+end_date_list[end_date]+".png")
        plt.show()
        


#plot_dead_pred_sns(data, dead, W=29, prob_dead=[0.33,0.33,0.8,0.22,1], start_date='2020-07-01',end_date='2021-05-15', infected=False)
    


"""
#from  df,mse,fatality_rate=main_fit_fatality_2020(data,dict_main_dead,W=29)
#july
fatality_rate_aux={
'<=39':	0.3821395835749183,
'40-49':	0.4587238816489043,
'50-59':	0.46569657311118495,
'60-69':	0.7527682102124622,
'>=70':	2.361623375121345}





30% de vacunacion
 

 ['2021-03-25','2021-03-12','2021-02-18','2021-05-15','2021-02-09']

df,mse,fatality_rate=main_fit_fatality_2020(data,dict_main_dead,W=29)

df1,dead_pred_2021,dead_pred_2020=main_fit_fatality_2021_v3(data,dict_main_dead,fatality_rate, month=['May','July'],W=29,end_date= ['2021-03-25','2021-03-12','2021-02-18','2021-05-15','2021-02-09'])

df[df.month=='July']
df1


start_date='2020-07-01'
end_date='2021-05-15'
dead_2021=dead_pred_2021.sum(axis=0)
dead_2020=dead_pred_2020.sum(axis=0)
groupID_to_group = data['groupID_to_group']
dateID_to_date = data['dateID_to_date']
date_to_dateID = data["date_to_dateID"]
#start_date='2020-10-01'  

dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
cols =['Grupo de edad', 'start_date', 'dead', 'type']
    
W=29
for date in range(dateID_start_date,dateID_end_date):
    #if date<=end_date_list_ID[g]:
    info = ['total',dateID_to_date[date]]
    info1=info.copy()
    info2=info.copy()
    info3=info.copy()
    info1.extend([data['dead'][:,date].sum(), 'real'])
    info2.extend([dead_pred_2021[date-(W-1)], 'predicted_2021'])
    info3.extend([dead_pred_2020[date-(W-1)], 'predicted_2020'])
    lst.append(info1)
    lst.append(info2)
    lst.append(info3)
df_res = pd.DataFrame(lst, columns=cols)




"""