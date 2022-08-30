#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 29 20:22:47 2022

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





#'inf_no_variant', 'inf_new_variant', 'inf_b117'

def main_fit_fatality_2020_using_inf(data,dict_main_dead,W=0):
    """
    1. give a range of time fit the curve 
    2. evaluate the MSE in beteween 01-07 to 31-12
    3. save result
    
    
    """
    inf_no_variant=data['inf_no_variant']
    inf_new_variant=data['inf_new_variant']
    inf_b117=data['inf_b117']
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
            x= inf_no_variant[g,dateID_date-(W-1):dateID_end_date+1-(W-1)]
            
            fit = curve_fit(fit_dead_2020, x, y)
            
            
            y_test=data['dead'][g,dateID_start_date:dateID_end_date+1]
            x_test= inf_no_variant[g,dateID_start_date-(W-1):dateID_end_date+1-(W-1)]
            y_pred=x_test*fit[0][0]
            plt.scatter(y_pred, y_test)
            plt.show()
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
    
    save=False
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


def main_fit_fatality_2021_v3(data,dict_main_dead,fatality_rate, month=['May','July'],W=29,end_date= ['2021-03-25','2021-03-12','2021-02-18','2021-05-14','2021-02-09']
):
    """
    given a moving windows
    1. give a range of time fit the curve 
    2. evaluate the MSE in beteween 01-07 to 31-12
    3. save result
    
    
    """
    inf_no_variant=data['inf_no_variant']
    inf_new_variant=data['inf_new_variant']
    inf_b117=data['inf_b117']
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
        no_variante=inf_no_variant[g,dateID_date-(W-1):dateID_end_date+1-(W-1)]
        variantes=[inf_new_variant[g,dateID_date-(W-1):dateID_end_date+1-(W-1)],inf_b117[g,dateID_date-(W-1):dateID_end_date+1-(W-1)]]
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
    plot_dead_pred_sns_moving_windows(data, dead_pred_2021,dead_pred_2020, W=W, prob_dead=[1,1,1,1,1], start_date='2020-07-01',end_date=end_date, infected=False)
    
    list_order_group=['<=39','40-49', '50-59', '60-69', '>=70']
    

    return df, dead_pred_2021, dead_pred_2020


#df,mse,fatality_rate=main_fit_fatality_2020_using_inf(data,dict_main_dead,W=0)
#df1, dead_pred_2021, dead_pred_2020= main_fit_fatality_2021_v3(data,dict_main_dead,fatality_rate, month=['May','July'],W=0,end_date= ['2021-03-25','2021-03-12','2021-02-18','2021-05-14','2021-02-09'])

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
    path_data='Image/Fatality using inf/movable windows'
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
            ax.figure.savefig(path+"Fatality_rate_time_window"+group_name+".png")
        except:
            ax.figure.savefig(path+"Fatality_rate_time_window"+group_name[:-2]+".png")
        plt.show()
        
        
def fit_dead_2020(x,a):
     return a*x
 
    
def MSE(y_pred,y):
     # Mean Squared Error
     return np.square(np.subtract(y_pred,y)).mean()
 
    


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
