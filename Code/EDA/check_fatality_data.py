#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:10:14 2022

@author: ineira
"""
import os 
import sys

import numpy as np
import pandas as pd

path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]
if path_projet[-1]!='/':
    path_projet+='/'
path_projet+='ICU_Simulations/'
path_ETL='Code/ETL'

module_path=path_projet+path_ETL
if module_path not in sys.path:
    sys.path.append(module_path)

from data_processing import prepare_producto_10,prepare_producto_16,read_data


def plot_dead(df_dead,start_date='2020-07-01',end_date='2021-5-15'):
    df=df_dead[(df_dead.start_date>=start_date)&(df_dead.start_date<=end_date)].copy()
    
    
    
    for i, (group_name, group_df) in enumerate(df.groupby(["Grupo de edad"])):
        
        dfm = group_df[['start_date', 'dead_today',
               'dead_today_v0', 'mean_dead_hampel']].melt('start_date', var_name='cols', value_name='vals')
        
        
        plt.subplots( figsize=(15, 9))
        ax=sns.lineplot(
            data=dfm,
            x='start_date', y ='vals'
            , palette= 'bright'#'Set3'
            ,legend=True,hue='cols'
        )
        ymax = dfm.vals.max()
        xpos = dfm.vals.idxmax()
        xmax = dfm['start_date'].iloc[xpos]
        
        ax.annotate('local max: '+str(int(ymax)), xy=(mdates.date2num(xmax), ymax), xytext=(mdates.date2num(xmax)+12, ymax+0.5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
        
        
        ax.set_xlim([dfm["start_date"].min()+pd.DateOffset(-2),dfm["start_date"].max()+pd.DateOffset(2)])
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.axes.set_title("Timeseries dead people for "+group_name+ "(Producto 10)",fontsize=15)
        ax.set_ylabel('Nº Dead', fontsize = 15)
        ax.legend(['Dead today hampel', 'Dead today (Producto 10)','Dead today hampel and rolling average'])
        plt.xlabel('Time', fontsize = 15)
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        plt.show()
def plot_dead_pred_sns(data, dead, W=29, prob_dead=[0.25,0.15,0.36,0.06,0.60], start_date='2020-07-01',end_date='2021-05-15', infected=False):
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
        ax.legend(['Dead today hampel', 'Dead today predicted (factor: '+ str(prob_dead[i])+')','Pacientes OUT UCI'],loc='upper left')
        #g.ax.legend(bbox_to_anchor=(1.01, 1.02), loc='upper left')
        plt.tight_layout()
        plt.gcf().autofmt_xdate()

        plt.show()
if __name__=='__main__':
    #call data
    data=read_data()
    df_dead=prepare_producto_10()
    df_16=prepare_producto_16()
    start_date='2020-07-01';end_date='2021-5-15'

    plot_dead(df_dead,start_date='2020-07-01',end_date='2021-5-15')
            
    df=pd.merge(df_dead,df_16, how='inner', on=['Grupo de edad','start_date'])

    df=df[(df.start_date>=start_date)&(df.start_date<=end_date)].copy()
    df['fatality']=df['dead_today']/df['infected_today']
    df['fatality_acc']=df['accumulated_dead']/df['accumulated_infected']



    dict_factor=dict()
    dict_factor['dict_factor']=[0.011799503728593824,
    0.035921368014992974,
    0.07184934248279035,
    0.003324302819280508,
    0.083]

    index=[0, 1, 2, 3, 4]
    dict_factor['Grupo de edad']=['40-49', '50-59', '60-69', '<=39', '>=70']

    df_factor=pd.DataFrame.from_dict(dict_factor,)

    df=pd.merge(df,df_factor, how='inner', on=['Grupo de edad'])


    df['fatality_uci']=df['dead_today']/(df['sintomatic_today']*df['dict_factor'])


    plt.subplots( figsize=(15, 9))
    ax=sns.lineplot(
        data=df,
        x='start_date', y ='fatality'
        , palette= 'bright'#'Set3'
        ,legend=True,hue='Grupo de edad'
    )
    text="Mean fatality"
    for i, (group_name, group_df) in enumerate(df.groupby(["Grupo de edad"])):
        text=text+"""
        """+group_name+': '+str(round(group_df.fatality.mean(),3))
        

    ax.set_xlim([df["start_date"].min()+pd.DateOffset(-2),df["start_date"].max()+pd.DateOffset(2)])
    xmin, xmax = ax.get_xlim()
    ax.text(xmin+30, 0.3, text, {'color': 'black', 'fontsize': 13})
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.axes.set_title("Timeseries fatality (dead today / infected today)",fontsize=15)
    ax.set_ylabel('Fatality: Prob(dead|infected)', fontsize = 15)
    plt.xlabel('Time', fontsize = 15)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()


    plt.subplots( figsize=(15, 9))
    ax=sns.lineplot(
        data=df,
        x='start_date', y ='fatality_acc'
        , palette= 'bright'#'Set3'
        ,legend=True,hue='Grupo de edad'
    )

    ax.set_xlim([df["start_date"].min()+pd.DateOffset(-2),df["start_date"].max()+pd.DateOffset(2)])
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.axes.set_title("Timeseries fatality (dead accumulated today / infected accumulated today)",fontsize=15)
    ax.set_ylabel('Fatality: Prob(dead|infected)', fontsize = 15)
    plt.xlabel('Time', fontsize = 15)
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()



    plt.subplots( figsize=(15, 9))
    ax=sns.lineplot(
        data=df,
        x='start_date', y ='fatality_uci'
        , palette= 'bright'#'Set3'
        ,legend=True,hue='Grupo de edad'
    )
    text="Mean fatality icu"
    for i, (group_name, group_df) in enumerate(df.groupby(["Grupo de edad"])):
        text=text+"""
        """+group_name+': '+str(round(group_df.fatality_uci.mean(),3))
        

    ax.set_xlim([df["start_date"].min()+pd.DateOffset(-2),df["start_date"].max()+pd.DateOffset(2)])
    xmin, xmax = ax.get_xlim()
    ax.text(xmin+30, 0.6, text, {'color': 'black', 'fontsize': 13})
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.axes.set_title("Timeseries fatality UCI (dead today /( sintomatic today*severidad 2020))",fontsize=15)
    ax.set_ylabel('Fatality UCI: Prob(dead|UCI)', fontsize = 15)
    plt.xlabel('Time', fontsize = 15)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()




    

    plot_dead_pred_sns(data, dead, W=29, prob_dead=[0.33,0.33,0.8,0.22,1], start_date='2020-07-01',end_date='2021-05-15', infected=False)