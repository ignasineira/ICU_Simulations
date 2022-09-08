#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  3 18:18:08 2022

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




#df,mse,fatality_rate= main_fit_fatality_2020_v4(data,dict_main_dead,W=29)
#df1, dead_pred_2021, dead_pred_2020=main_fit_fatality_2021_v4(data,dict_main_dead,fatality_rate, month=['May','July'],W=29,end_date_list= ['2021-03-25','2021-03-12','2021-02-18','2021-05-14','2021-02-09'])


def main_fit_fatality_2020_v4(data,dict_main_dead,W=29):
    """
    **product 57
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
    index_70=list(groupID_to_group.keys())[list(groupID_to_group.values()).index('>=70')]

    frames=[]
    
   
    for date in list_start_date:
        dateID_date = data['date_to_dateID'][np.datetime64(date+"T00:00:00.000000000")]
        y= (data['dead_icu']-data['dead'][index_70,:])[dateID_date:dateID_end_date+1]
        #y= (data['dead_icu'])[dateID_date:dateID_end_date+1]
        x= dead['Not variant'][:4,dateID_date-(W-1):dateID_end_date+1-(W-1)].sum(axis=0)
        
        fit = curve_fit(fit_dead_2020, x, y)
        
        
        y_test=(data['dead_icu']-data['dead'][index_70,:])[dateID_start_date:dateID_end_date+1]
        #y_test=(data['dead_icu'])[dateID_start_date:dateID_end_date+1]
        x_test= dead['Not variant'][:4,dateID_start_date-(W-1):dateID_end_date+1-(W-1)].sum(axis=0)
        y_pred=x_test*fit[0][0]
        aux_date=[ data['dateID_to_date'][key] for key in range(dateID_start_date-(W-1),dateID_end_date+1-(W-1))]
        plt.plot(aux_date, y_pred, label='dead pred')
        plt.plot(aux_date, y_test, label='dead')
        plt.grid(axis='x', color='0.95')
        plt.legend( loc='upper right',fontsize=7)#(title='git hub product'
        plt.title('National dead time series')
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        plt.show()
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
    index_70=list(groupID_to_group.keys())[list(groupID_to_group.values()).index('>=70')]

    lower_bound=fatality_rate.to_dict()['July']
    upper_bound=fatality_rate.to_dict()['July']
    group_name='total'

    frames=[]
    
    dead_pred_model_2021=[]
    dead_pred_model_2020=[]
    for end_date in end_date_list:
        dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
        dateID_date = data['date_to_dateID'][np.datetime64(date+"T00:00:00.000000000")]
        y=(data['dead_icu']-data['dead'][index_70,:])[dateID_date:dateID_end_date+1]
        #y=(data['dead_icu'])[dateID_date:dateID_end_date+1]
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
        
        
        
        y_test=(data['dead_icu']-data['dead'][index_70,:])[dateID_start_date:dateID_end_date+1]
        #y_test=(data['dead_icu'])[dateID_start_date:dateID_end_date+1]
        x_test= np.stack([item[:4,dateID_start_date-(W-1):dateID_end_date+1-(W-1)].sum(axis=0) for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
        params_2021=np.array([fit[0][0],fit[0][1],fit[0][1]])
        y_pred_model_2021=(x_test*np.array(params_2021).reshape((3, -1))).sum(axis=0)
        mse_model_2021= MSE(y_pred_model_2021,y_test)
        
        params_2020=np.array([fit[0][0],fit[0][0],fit[0][0]])
        y_pred_model_2020=(x_test*np.array(params_2020).reshape((3, -1))).sum(axis=0)
        mse_model_2020= MSE(y_pred_model_2020,y_test)
        
# =============================================================================
#         aux_date=[ data['dateID_to_date'][key] for key in range(dateID_start_date-(W-1),dateID_end_date+1-(W-1))]
#         plt.plot(aux_date, y_pred_model_2020, label='dead pred 2020')
#         plt.plot(aux_date, y_pred_model_2021, label='dead pred 2021')
#         plt.plot(aux_date, y_test, label='dead')
#         plt.grid(axis='x', color='0.95')
#         plt.legend( loc='upper right',fontsize=7)#(title='git hub product'
#         plt.title('National dead time series')
#         plt.tight_layout()
#         plt.gcf().autofmt_xdate()
#         plt.show()
#         
# =============================================================================
        
        #save data all times series
        x_test= np.stack([item[:4,:].sum(axis=0) for key, item in dict_main_dead['dict_dead_variant'].items() if key!='total'],axis=0)
        y_pred_model_2021=(x_test*np.array(params_2021).reshape((3, -1))).sum(axis=0)
        dead_pred_model_2021.append(y_pred_model_2021)
        
        y_pred_model_2020=(x_test*np.array(params_2020).reshape((3, -1))).sum(axis=0)
        dead_pred_model_2020.append(y_pred_model_2020)
        
        
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
    plot_dead_pred_sns_moving_windows_model_2021_v4(data, dead_pred_2021,dead_pred_2020, W=29, prob_dead=[1,1,1,1,1], start_date='2020-07-01',end_date_list=end_date_list, infected=False)
    
    
    return df, dead_pred_2021, dead_pred_2020

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
    path_data='Image/Fatality/ICU dead/movable windows group by all'
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
            info1.extend([(data['dead_icu']-data['dead'][4,:])[date], 'real'])
            #info1.extend([(data['dead_icu'])[date], 'real'])
            info2.extend([dead_pred_2021[end_date,date-(W-1)], 'predicted_2021'])
            info3.extend([dead_pred_2020[end_date,date-(W-1)], 'predicted_2020'])
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



"""
aux_date=[ data['dateID_to_date'][key] for key in range(len(data['dateID_to_date']))]
for key,item in data['groupID_to_group'].items():
    plt.plot(aux_date,data['dead'][key,:], label='dead pred '+item)

plt.plot(aux_date, data['dead_icu'], label='dead uci')
plt.xlim([aux_date[91],aux_date[409]])
plt.grid(axis='x', color='0.95')
plt.legend( loc='upper right',fontsize=7)#(title='git hub product'
plt.title('Dead time series')
plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.show()

"""