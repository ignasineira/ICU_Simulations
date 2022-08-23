# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 14:23:39 2021

@author: ignas


find the best probability  parameters going to ICU using little law


start_date='2020-08-08';end_date='2021-01-31'
aux=find_probability_UCI(data, start_date, end_date,list_p_in_uci,list_n_in_uci)
np.round((np.abs(1-aux*(1/np.array([0.0103398,0.0293233,0.0732187,0.00304418,0.073859])))*100),2).tolist()
"""

import pandas as pd
import numpy as np 

from scipy.stats import nbinom


list_p_in_uci=[0.0511, 0.0702, 0.0513, 0.0496, 0.0424];
list_n_in_uci=[1.408, 1.9843, 1.4427, 1.3219, 1.3486]


start_date='2020-05-01';end_date='2020-12-31'
def find_probability_icu(data, start_date, end_date,list_p_in_uci,list_n_in_uci):
    
    groupID_to_group = data['groupID_to_group']
    dateID_to_date = data['dateID_to_date']
    date_to_dateID = data["date_to_dateID"]
    #start_date='2020-10-01'  

    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    aux=np.array([list_n_in_uci,list_p_in_uci]).T
    #shif curve
    shif_curve=[8, 7, 8, 7, 4]
    shif_curve2=[10, 10, 12, 8, 11]
    inf = data['inf'].copy()
    """
    for i,item in enumerate(shif_curve):
        inf[i,0:-(item-1)]=data['inf'][i,item-1:]
    #for i,item in enumerate(shif_curve2):
        #inf[i,item-1:]=inf[i,0:-(item-1)]
    """
    factor=0
    L=np.mean(data['uci'][:,dateID_start_date:dateID_end_date+1], axis=1)
    W=np.array([int(nbinom(item[0],item[1]).mean()) for item in aux ])
    λ = np.mean(inf[:, dateID_start_date+factor:dateID_end_date+1+factor], axis=1)
    probability_not_vac = L/ (W * λ)
    
    return probability_not_vac


def find_probability_icu_v2(data, start_date, end_date,list_mu_in_uci):
    
    groupID_to_group = data['groupID_to_group']
    dateID_to_date = data['dateID_to_date']
    date_to_dateID = data["date_to_dateID"]
    #start_date='2020-10-01'  

    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    aux=np.array([list_n_in_uci,list_p_in_uci]).T
    #shif curve
    shif_curve=[8, 7, 8, 7, 4]
    shif_curve2=[10, 10, 12, 8, 11]
    
    inf = data['inf'].copy()
    
    """
    for i,item in enumerate(shif_curve):
        inf[i,0:-(item-1)]=data['inf'][i,item-1:]
    #for i,item in enumerate(shif_curve2):
        #inf[i,item-1:]=inf[i,0:-(item-1)]
    """
    factor=0
    L=np.mean(data['uci'][:,dateID_start_date:dateID_end_date+1], axis=1)
    W=np.array(list_mu_in_uci)
    λ = np.mean(inf[:, dateID_start_date+factor:dateID_end_date+1+factor], axis=1)
    probability_not_vac = L/ (W * λ)
    
    return probability_not_vac


if __name__ == "__main__":
    import os
    import sys
    
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_ETL='Code/ETL'

    module_path=path_projet+path_ETL
    if module_path not in sys.path:
        sys.path.append(module_path)

    from data_processing import read_data
    
    from datetime import datetime
    
    import matplotlib.pyplot as plt 
    from matplotlib.dates import date2num
    import matplotlib.dates as mdates
    import seaborn as sns
    
    data = read_data()
    list_day=[]
    list_prob=[]
    j=5
    for i in range(j,17):
        
        year_aux=(i)//12
        moth_aux=i+1-year_aux*12
        
        start_date=datetime(2020,j,1).strftime('%Y-%m-%d')
        end_date=datetime(2020+year_aux,moth_aux,1).strftime('%Y-%m-%d')
        list_day.append(end_date)
        #print(datetime(2021,i,1).strftime( '%Y-%m-%d'))
        list_prob.append(find_probability_icu(data, start_date, end_date,list_p_in_uci,list_n_in_uci))
        print(find_probability_icu(data, start_date, end_date,list_p_in_uci,list_n_in_uci))
        
    groupID_to_group=data['groupID_to_group'] 
    cols =['Grupo de edad', 'start_date','prob']
    lst = []
    for g in range(len(groupID_to_group)):
        for i,date in enumerate(list_day): 
            info = [groupID_to_group[g],date,round(list_prob[i][g],3)]
            lst.append(info)
    df_res = pd.DataFrame(lst, columns=cols)
    df_res['start_date']=pd.to_datetime(df_res['start_date'])

    g=sns.relplot(x="start_date", y="prob",
                                hue="Grupo de edad",
                                #palette=palette,
                                kind="line", data=df_res,sizes=((12, 24)))

    g.set(xlim=[df_res["start_date"].min()+pd.DateOffset(-2),df_res["start_date"].max()+pd.DateOffset(2)])
    g.ax.grid()

    g.ax.xaxis_date()

    g.ax.xaxis.set_major_locator(mdates.MonthLocator())
    g.ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    g.ax.axes.set_title(" Change in the probability of going to ICU \n by increasing the time window (every one month). " ,fontsize=15)
    g.ax.set_ylabel('Prob  UCI', fontsize=10)
    g.ax.set_xlabel('Date', fontsize=10)
    #plt.tight_layout()
    plt.gcf().autofmt_xdate()


    """
    start_date='2020-05-01';end_date='2020-12-31'

    list_mu_in_uci=np.round(np.array(mu_in_uci)-1.96*(1/np.sqrt(n_obs_in_uci*I_mu_in_uci)),3)

    probability_not_vac=find_probability_UCI(data, start_date, end_date,list_p_in_uci,list_n_in_uci)


    list_mu_in_uci=np.round(np.array(mu_in_uci)+1.96*(1/np.sqrt(n_obs_in_uci*I_mu_in_uci)),3)

    probability_not_vac=find_probability_UCI_v2(data, start_date, end_date,list_mu_in_uci)

    """




