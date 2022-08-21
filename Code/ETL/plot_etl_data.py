#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 17:00:36 2022

@author: ineira

plot products of interest


"""
import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np 
import itertools
import time

from campana_vacunacion import Avance_vacunacion
from productos_min_ciencia import producto_5,producto_9,producto_10,producto_16,producto_21, producto_26, producto_39, producto_77

from data_processing import prepare_producto_16

import matplotlib.pyplot as plt 
from matplotlib.dates import date2num
import matplotlib.dates as mdates
from datetime import  datetime
import seaborn as sns ## These Three lines are necessary for Seaborn to work   
sns.set(color_codes=True)

plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
sns.set(rc={"figure.dpi":400, 'savefig.dpi':400})



def plot_infected(df_16):
    """
    ** Use prepare_producto_16
    Main idea: Plot 

    Parameters
    ----------
    df_16 : Dataframe
        columns: ['Grupo de edad','start_date','accumulated_infected',
               'sintomatic_today','asintomatic_today','infected_today_all',
               'infected_today','infected_sintomatic_today'].

    Returns
    -------
    None.
    """
    df_16['proc'] = df_16['infected_sintomatic_today']/df_16['infected_today']
    for i, (group_name, group_df) in enumerate(df_16.groupby(["Grupo de edad"])):
        plt.subplots( figsize=(15, 9))
        ax=sns.lineplot(
            data=group_df,
            x='start_date', y ='proc'
            , palette= 'bright'#'Set3'
            ,legend=False
        )
        ax.axes.set_title("Timeseries proportion of symptomatic infected for "+group_name ,fontsize=15)
        degrees = 70
        plt.xticks(rotation=degrees)
        plt.ylabel('Proportion of symtomotic infectd', fontsize = 15) # x-axis label with fontsize 15 
        plt.xlabel('Time', fontsize = 15)
        plt.tight_layout()
        plt.show()
        
        fig, ax =plt.subplots( figsize=(15, 9))
        sns.lineplot(
            data=group_df,
            x='start_date', y ='infected_today'
            , palette= 'bright'#'Set3'
            ,legend=False, ax=ax
        )
        sns.lineplot(
            data=group_df,
            x='start_date', y ='infected_sintomatic_today'
            , palette= 'bright'#'Set3'
            ,legend=False, ax=ax
        )
        #group_df.plot( kind='line', x="start_date", y=['infected_today','infected_sintomatic_today'],label=['infected','symptomatic infected' ], ax=ax)
        ax.axes.set_title("Timeseries symptomatic infected and total infected for "+group_name ,fontsize=15)
        ax.legend(['infected','symptomatic infected' ],bbox_to_anchor=(1.01, 1.02), loc='upper left')
        degrees=70
        plt.xticks(rotation=degrees)
        plt.ylabel('Nº Infected', fontsize = 15) # x-axis label with fontsize 15 
        plt.xlabel('Time', fontsize = 15)
        plt.tight_layout()
        plt.show()

def plot_producto_5(df_5):
    """
    Main idea: Plot 

    Parameters
    ----------
    df_5 : Dataframe 
        casos nuevos confirmados.

    Returns
    -------
    None.

    """
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 9*4))
    df_5.plot( kind='line', x="start_date", y=['Casos nuevos con sintomas','Casos nuevos sin sintomas'],ax=axes[0])
    axes[0].set_xlim([df_5["start_date"].min()+pd.DateOffset(-2),df_5["start_date"].max()+pd.DateOffset(2)])
    axes[0].xaxis_date()
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[0].axes.set_title("Timeseries new cases by symptoms (Producto 5)",fontsize=15)
    axes[0].set_ylabel('Nº Infected', fontsize = 15)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    
    
    df_5.plot( kind='line', x="start_date", y=['Casos nuevos totales','Casos nuevos totales check'],ax=axes[1])
    axes[1].set_xlim([df_5["start_date"].min()+pd.DateOffset(-2),df_5["start_date"].max()+pd.DateOffset(2)])
    axes[1].xaxis_date()
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[1].axes.set_title("Timeseries total new cases by day (Producto 5)" ,fontsize=15)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    
    
    df_5.plot( kind='line', x="start_date", y=['diff Casos nuevos totales'],ax=axes[2])
    axes[2].set_xlim([df_5["start_date"].min()+pd.DateOffset(-2),df_5["start_date"].max()+pd.DateOffset(2)])
    axes[2].xaxis_date()
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[2].axes.set_title("Timeseries diff new cases (Producto 5)" ,fontsize=15)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    
    
    
    df_5.plot( kind='line', x="start_date", y=['proc diff Casos nuevos totales'],ax=axes[3])
    axes[3].set_xlim([df_5["start_date"].min()+pd.DateOffset(-2),df_5["start_date"].max()+pd.DateOffset(2)])
    axes[3].xaxis_date()
    axes[3].xaxis.set_major_locator(mdates.MonthLocator())
    axes[3].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[3].axes.set_title("Timeseries porc diff new cases (Producto 5)" ,fontsize=15)
    
    plt.xlabel('Time', fontsize = 15)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()

    plt.show()
    
def plot_producto_5_16(df_5, df_16):
    """
    Main idea: 
        Plot 1: check if there are substantial differences between the 
        number of new cases depending on the product used 
        Plot 2: 
    

    Parameters
    ----------
    df_5 : DatFrame 
        DESCRIPTION.
    df_16 : DatFrame
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    fig, axes = plt.subplots(2, 1, figsize=(15,9*2))
    sns.lineplot(
        data=df_5,
        x='start_date', y ='Casos nuevos totales'
        , palette= 'bright'#'Set3'
        ,legend=True, ax=axes[0],label= 'infected today '+'(Producto 5)'
    )
    sns.lineplot(
        data=df_5,
        x='start_date', y ='Casos nuevos totales check'
        , palette= 'bright'#'Set3'
        ,legend=True, ax=axes[0],label= 'infected today check '+'(Producto 5)'
    )
    sns.lineplot(
        data=df_16,
        x='start_date', y ='infected_today_all'
        , palette= 'bright'#'Set3'
        ,legend=True, ax=axes[0],label= 'infected today '+'(Producto 16)'
    )
    axes[0].set_xlim([df_5["start_date"].min()+pd.DateOffset(-2),df_5["start_date"].max()+pd.DateOffset(2)])
    axes[0].xaxis_date()
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[0].axes.set_title("Timeseries total new cases by day ",fontsize=15)
    axes[0].set_ylabel('Nº Infected', fontsize = 15)
    axes[0].axes.xaxis.set_visible(True)    
    
    
    #group by product 16:
    g_sum = df_16[['start_date', 'accumulated_infected']].groupby(['start_date']).sum()  
    sns.lineplot(
        data=df_5,
        x='start_date', y ='Casos totales'
        , palette= 'bright'#'Set3'
        ,legend=True, ax=axes[1],label= 'accumulated infected'+'(Producto 5)'
    )
    sns.lineplot(
        data=g_sum,
        x='start_date', y ='accumulated_infected'
        , palette= 'bright'#'Set3'
        ,legend=True, ax=axes[1],label= 'accumulated infected'+'(Producto 16)'
    )

    axes[1].set_xlim([df_5["start_date"].min()+pd.DateOffset(-2),df_5["start_date"].max()+pd.DateOffset(2)])
    axes[1].xaxis_date()
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[1].axes.set_title("Timeseries accumulated  cases by day " ,fontsize=15)
    axes[1].set_ylabel('Nº Infected', fontsize = 15)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.xlabel('Time', fontsize = 15)
    plt.show()
    
def plot_producto_5_16_26(df_5, df_16,df_26):
    """
    ** warning: Producto 5 nivel nacional, producto 16 grupo etario , producto 26 regional
    
    Main idea: 
        Plot 1: check if there are substantial differences between the 
        number of new cases depending on the product used 
        Plot 2: check if there are substantial differences between the 
        number of asymtomatic and symtomatic cases depending on the product used 
        Plot 3: check if there are substantial differences between the 
        number of asymtomatic and symtomatic cases depending on the product used
    
    Parameters
    ----------
    df_5 : TYPE
        DESCRIPTION.
    df_16 : TYPE
        DESCRIPTION.
    df_26 : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    df26=df_26[df_26['Region']=='Total'].copy()
    df26['infected_today_26']=df26[[ 'sintomatic_today', 'asintomatic_today']].sum(axis=1)
    df26['accumulated_infected_26']=df26['infected_today_26'].cumsum()
    fig, axes = plt.subplots(3, 1, figsize=(15, 9*3))
    sns.lineplot(
        data=df_5,
        x='start_date', y ='Casos nuevos totales'
        , palette= 'bright'#'Set3'
        ,legend=True, ax=axes[0],label= 'infected today '+'(Producto 5)'
    )
    sns.lineplot(
        data=df_16,
        x='start_date', y ='infected_today_all'
        , palette= 'bright'#'Set3'
        ,legend=True, ax=axes[0],label= 'infected today '+'(Producto 16)'
    )
    sns.lineplot(
        data=df26,
        x='start_date', y ='infected_today_26'
        , palette= 'bright'#'Set3'
        ,legend=True, ax=axes[0],label= 'infected today '+'(Producto 26)'
    )
    
    axes[0].set_xlim([df_5["start_date"].min()+pd.DateOffset(-2),df_5["start_date"].max()+pd.DateOffset(2)])
    axes[0].xaxis_date()
    axes[0].xaxis.set_major_locator(mdates.MonthLocator())
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[0].axes.set_title("Timeseries total new cases by day (Porduct 5,16,26)",fontsize=15)    
    axes[0].set_ylabel('Nº Infected', fontsize = 15)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    
    
    sns.lineplot(
        data=df_5,
        x='start_date', y ='Casos nuevos con sintomas'
        , palette= 'bright'#'Set3'
        ,legend=True, ax=axes[1],label= 'sintomatic today (Producto 5)'
    )
    sns.lineplot(
        data=df_5,
        x='start_date', y ='Casos nuevos sin sintomas'
        , palette= 'bright'#'Set3'
        ,legend=True, ax=axes[1],label= 'asintomatic today (Producto 5)'
    )
    
    sns.lineplot(
        data=df26,
        x='start_date', y ='asintomatic_today'
        , palette= 'bright'#'Set3'
        ,legend=True, ax=axes[1],label= 'asintomatic today (Producto 26)'
    )
    sns.lineplot(
        data=df26,
        x='start_date', y ='sintomatic_today'
        , palette= 'bright'#'Set3'
        ,legend=True, ax=axes[1],label= 'sintomatic today (Producto 26)'
    )
    axes[1].set_xlim([df_5["start_date"].min()+pd.DateOffset(-2),df_5["start_date"].max()+pd.DateOffset(2)])
    axes[1].xaxis_date()
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[1].axes.set_title("Timeseries new cases by symptoms and  day (Porduct 5,26) " ,fontsize=15)
    axes[1].set_ylabel('Nº Infected', fontsize = 15)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    
    #group by product 16:
    g_sum = df_16[['start_date', 'accumulated_infected']].groupby(['start_date']).sum()  
    sns.lineplot(
        data=df_5,
        x='start_date', y ='Casos totales'
        , palette= 'bright'#'Set3'
        ,legend=True, ax=axes[2],label= 'accumulated infected'+'(Producto 5)'
    )
    sns.lineplot(
        data=g_sum,
        x='start_date', y ='accumulated_infected'
        , palette= 'bright'#'Set3'
        ,legend=True, ax=axes[2],label= 'accumulated infected'+'(Producto 16)'
    )
    sns.lineplot(
        data=df26,
        x='start_date', y ='accumulated_infected_26'
        , palette= 'bright'#'Set3'
        ,legend=True, ax=axes[2],label= 'accumulated infected'+' (Producto 26)'
    )
    axes[2].set_xlim([df_5["start_date"].min()+pd.DateOffset(-2),df_5["start_date"].max()+pd.DateOffset(2)])
    axes[2].xaxis_date()
    axes[2].xaxis.set_major_locator(mdates.MonthLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[2].axes.set_title("Timeseries accumulated  cases by day " ,fontsize=15)
    axes[2].set_ylabel('Nº Infected', fontsize = 15)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.xlabel('Time', fontsize = 15)
    plt.show()
    
    
def plot_77(df_77, all_chile=False):
    """
    Warning: use producto_77()

    Parameters
    ----------
    df_77 : TYPE
        DESCRIPTION.
    all_chile : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """
    if all_chile:
        var_row = 'Region'
        list_dosis = df_77["Dosis"].unique().tolist()
        list_reg = df_77["Region"].unique().tolist()
        for i in range(2):
            for j in range(len(list_reg)):
                plt.subplots( figsize=(15, 9))
                ax=sns.lineplot(
                    data=df_77[(df_77[var_row]==list_reg[j])&(df_77['Dosis']==list_dosis[i])],
                    x='start_date', y ='accumulated_vaccinated',
                    hue='Grupo de edad')
                
                ax.axvspan(date2num(datetime(2021,2,3)), date2num(datetime(2021,2,15)), 
                       label="older adults",color="green", alpha=0.3)
                ax.axvspan(date2num(datetime(2021,2,22)), date2num(datetime(2021,2,23)), 
                       label="older adults",color="red", alpha=0.3)
                ax.axvspan(date2num(datetime(2021,3,3)), date2num(datetime(2021,3,15)), 
                       label="2 vaccine older adults ",color="blue", alpha=0.3)
                ax.axes.set_title("Timeseries accumulated vaccinated "+list_dosis[i]+ " dosis for "+list_reg[j],fontsize=20)
                plt.savefig('image/Timeseries_accumulated_vaccinated_'+list_dosis[i]+ "_dosis_"+list_reg[j]+'.png')
    else:
        var_row = 'Region'
        list_dosis = df_77["Dosis"].unique().tolist()
        list_reg = df_77["Region"].unique().tolist()
        for i in range(2):
            plt.subplots( figsize=(15, 9))
            ax=sns.lineplot(
                data=df_77[(df_77[var_row]=='Total')&(df_77['Dosis']==list_dosis[i])],
                x='start_date', y ='accumulated_vaccinated',
                hue='Grupo de edad')
            
            ax.axvspan(date2num(datetime(2021,2,3)), date2num(datetime(2021,2,15)), 
                   label="older adults",color="green", alpha=0.3)
            ax.axvspan(date2num(datetime(2021,2,22)), date2num(datetime(2021,2,23)), 
                   label="older adults",color="red", alpha=0.3)
            ax.axvspan(date2num(datetime(2021,3,3)), date2num(datetime(2021,3,15)), 
                   label="2 vaccine older adults ",color="blue", alpha=0.3)
            ax.axes.set_title("Timeseries accumulated vaccinated "+list_dosis[i]+ " dosis for "+list_reg[j],fontsize=20)
            #plt.savefig('image/Timeseries_accumulated_vaccinated_'+list_dosis[i]+ "_dosis_"+list_reg[j]+'.png')
           

def plot_aux(df_77):
    start_date='2020-07-01',
    end_date='2021-05-15'
    var_row = 'Region'
    list_dosis = df_77["Dosis"].unique().tolist()
    list_reg = df_77["Region"].unique().tolist()
    for i in range(2):
    
        plt.subplots( figsize=(15, 9))
        ax=sns.lineplot(
            data=df_77[(df_77[var_row]=='Total')&(df_77['Dosis']==list_dosis[i])&(df_77[df_77.start_date>='2020-07-01'])],
            x='start_date', y ='accumulated_vaccinated',
            hue='Grupo de edad')
    
        ax.axvspan(date2num(datetime(2021,2,3)), date2num(datetime(2021,2,15)), 
               label="older adults",color="green", alpha=0.3)
        ax.axvspan(date2num(datetime(2021,2,22)), date2num(datetime(2021,2,23)), 
               label="older adults",color="red", alpha=0.3)
        ax.axvspan(date2num(datetime(2021,3,3)), date2num(datetime(2021,3,15)), 
               label="2 vaccine older adults ",color="blue", alpha=0.3)
        ax.axes.set_title("Timeseries accumulated vaccinated "+list_dosis[i]+ " dosis for Total",fontsize=20)
      

def plot_10(df_10,start_date='2020-07-01',end_date='2021-05-15'):
    
    df=df_10[(df_10['start_date']>=start_date)&(df_10['start_date']<=end_date)]
    
    
    group_name_position={
    '40-49':(1),
    '50-59':(2),
    '60-69':(3), 
    '<=39':(0)
    }
    group_name_position={
    '40-49':(0,1),
    '50-59':(1,0),
    '60-69':(1,1), 
    '<=39':(0,0)
    }
    fig, axes = plt.subplots(2, 2, figsize=(15*2, 9*2))
    for i, (group_name, group_df) in enumerate(df.groupby(["Grupo de edad"])):
        if group_name in list(group_name_position.keys()):
            #fig, ax = plt.subplots(figsize=(30, 13))
            index_group_name=group_name_position[group_name]
            ax=axes[index_group_name]
                
                
            sns.lineplot(
                data=group_df,
                x='start_date', y ='dead_today'
                , palette= 'bright'#'Set3'
                ,legend=False, ax=ax,label= 'Dead today'
            )
            sns.lineplot(
                data=group_df,
                x='start_date', y ='dead_today'
                , palette= 'bright'#'Set3'
                ,legend=False, ax=ax,label= 'Dead today'
            )
            ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
            ax.xaxis_date()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.axes.set_title("Timeseries dead by day for "+ group_name,fontsize=15)
            ax.set_ylabel('Nº Dead', fontsize = 15)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.xlabel('Time', fontsize = 15)
    plt.show()
    
    

    
def plot_dead_pred_2020(data, dead, W=29, pond=1, start_date='2020-07-01',end_date='2021-04-23', infected=False):
    groupID_to_group = data['groupID_to_group']
    dateID_to_date = data['dateID_to_date']
    date_to_dateID = data["date_to_dateID"]
    #start_date='2020-10-01'  

    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    cols =['Grupo de edad', 'start_date','inf', 'dead_real', 'dead_pred']
    lst = []
    for g in range(len(groupID_to_group)):
        for date in range(dateID_start_date,dateID_end_date): 
            info = [groupID_to_group[g],dateID_to_date[date],data['inf'][g,date]*pond,data['dead'][g,date]]
            info.append(dead[g,date-(W-1)])
            lst.append(info)
    df_res = pd.DataFrame(lst, columns=cols)
    for i, (group_name, group_df) in enumerate(df_res.groupby(["Grupo de edad"])):
    #for i in range(2):
    
        #plt.subplots(figsize=(15, 9))
        # plot the group on these axes
            
        if infected:
            ax=group_df.plot( kind='line', x="start_date", y=['dead_real', 'dead_pred','inf'])
        else:
            ax=group_df.plot( kind='line', x="start_date", y=['dead_real', 'dead_pred'])
        # set the title
        ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
        ax.grid()
        
        ax.xaxis_date()
        """
        tks = 30
        #locator = mdates.AutoDateLocator(minticks=tks, maxticks=tks)
        locator = mdates.DateFormatter('%Y-%m-%d')
        formatter = mdates.ConciseDateFormatter(locator)
        #ax.xaxis.set_major_locator(locator)
        #ax.xaxis.set_major_formatter(formatter)
        """
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        
        ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,3)), 
               label="older adults",color="red", alpha=0.3)
    
        
        ax.axes.set_title("Timeseries DEAD for "+group_name ,fontsize=15)
        ax.set_ylabel('N° beds', fontsize=10)
        ax.set_xlabel('Date', fontsize=10)
        """
        try:
            plt.savefig('image/UCI/Timeseries_DEAD_'+str(group_name) +'.png')
        except:
            plt.savefig('image/UCI/Timeseries_DEAD_'+str(group_name)[2:] +'.png')
        """
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        #plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
        plt.show()

#df_77=Avance_vacunacion()
#df_pop = prepare_population()
        
def plot_vaccine_final(data,df_77,df_pop,start_date='2021-02-01',end_date='2021-05-15'):
    #style = 'seaborn-dark'
    
    

    groupID_to_group = data['groupID_to_group']
    dateID_to_date = data['dateID_to_date']
    date_to_dateID = data["date_to_dateID"]
    #start_date='2020-10-01'  

    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    n_moth=int(len(df_77['start_date'].dt.strftime("%m/%y").unique().tolist())/5)
    
    
    group_name_position={
        '40-49':(1),
        '50-59':(2),
        '60-69':(3), 
        '<=39':(0)
        }
    
    
    group_name_columns=df_77["Grupo de edad"].unique().tolist()
    group_name_columns=[group_name_columns[3]]+group_name_columns[0:3]
    figs, axs = plt.subplots(3,1,figsize=(20, 12*3),sharex=True, sharey=True)
    df_res=pd.merge(df_77,df_pop, how='left', on=["Grupo de edad"])
    df_res['accumulated_vaccinated']= df_res['accumulated_vaccinated']/ df_res['Personas']
    
    df_res=df_res.pivot(index=[ 'start_date', 'Laboratorio', 'Dosis', 'Region'],columns='Grupo de edad',values='accumulated_vaccinated').reset_index()
    add_title=[' (A)',' (B)',' (C)']
    for i, (group_name, group_df) in enumerate(df_res[(df_res['Dosis']=='Primera')&(df_res['start_date']>=np.datetime64(start_date+"T00:00:00.000000000"))&(df_res['start_date']>=np.datetime64(start_date+"T00:00:00.000000000"))&(df_res['start_date']<=np.datetime64(end_date+"T00:00:00.000000000"))].groupby(["Laboratorio"])):
        
        
        ax=axs[i]
        group_df.plot( kind='line', x="start_date",
                              y=group_name_columns,
                              ax=ax,
                              lw=6,
                              label=group_name_columns)
    
        # set the title
        ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
        #ax.grid()
        
        ax.xaxis_date()

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=n_moth))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        

        ax.axes.set_title("Cumulative vaccination of population in time by "+group_name[20:-1] + add_title[i],fontsize= 26)
        ax.set_ylabel('', fontsize=27)
        #ax.set_xlabel('Date \n', fontsize=20)
        
        #ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,2)), color="green", alpha=0.3)#label="older adults",


        #ax.label_outer()
        ax.xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=22)
        # plot the group on these axes
        plt.tight_layout()
        ax.legend(loc='upper left', fontsize=25)

        #plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
    #plt.setp(axs[0,0].get_xticklabels(), visible=False)
    porc=0.3
    var_row = 'Region'
    list_dosis = df_77["Dosis"].unique().tolist()
    list_reg = df_77["Region"].unique().tolist()
    data_porc=[]
    for i, (group_name, group_df) in enumerate(df_77.groupby(["Grupo de edad"])):
        group_df=group_df.groupby(["Grupo de edad",'start_date','Region','Dosis']).agg('sum').reset_index()
        group_df['accumulated_vaccinated']=group_df['accumulated_vaccinated']/df_pop[df_pop["Grupo de edad"]==group_name]['Personas'].values[0]
        value=group_df[(group_df['Region']=='Total')&(group_df['Dosis']==list_dosis[0])&(group_df['accumulated_vaccinated']>=porc)]['start_date'].values[0]
        print(value)
        data_porc.append(value)
        
        
    df_res=df_77.groupby(["Grupo de edad",'start_date','Region','Dosis']).agg('sum').reset_index()
    
    df_res=pd.merge(df_res,df_pop, how='left', on=["Grupo de edad"])
    df_res['accumulated_vaccinated']= df_res['accumulated_vaccinated']/ df_res['Personas']
    df_res=df_res[(df_res['start_date']>=np.datetime64(start_date+"T00:00:00.000000000"))&(df_res['start_date']<=np.datetime64(end_date+"T00:00:00.000000000"))].pivot(index=[ 'start_date', 'Dosis', 'Region'],columns='Grupo de edad',values='accumulated_vaccinated').reset_index()
    ax=axs[2]
    df_res[(df_res['Dosis']=='Primera')].plot( kind='line', x="start_date",
                              y=group_name_columns,
                              ax=ax,
                              lw=6,
                              label=group_name_columns)
    
    # set the title
    ax.set_xlim([df_res["start_date"].min()+pd.DateOffset(-2),df_res["start_date"].max()+pd.DateOffset(2)])
    #ax.grid()
    
    ax.xaxis_date()

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=n_moth))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    
    ax.axes.set_title("Cumulative vaccination of population in time " + add_title[2],fontsize= 26)
    ax.set_ylabel('', fontsize=27)
    #ax.set_xlabel('Date \n', fontsize=20)
    ax.set_xlabel('', fontsize=20)
    
    #ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,2)), color="green", alpha=0.3)#label="older adults",
    for i,color in enumerate(['darkorange','green','red','blue']):#purple
        #ax.axvspan(date2num(data_porc[i]), date2num(data_porc[i])+0.5, 
           #color=color, alpha=0.3,linestyle='dashed')#label="older adults",
        #ax.axhline(x=date2num(data_porc[i]), color='r', linestyle='-')
        #hlines=[40,50]
        #ax.vlines(date2num(data_porc[i]), 1, 100, color='g')
        if i in [1,2]:
            ax.axvline(x=date2num(data_porc[i])-0.5,color=color,linestyle='dashed' )
        else:
            ax.axvline(x=date2num(data_porc[i]),color=color,linestyle='dashed' )
    #ax.axhline(y=0.3)
    #ax.label_outer()
    ax.xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=18)
    # plot the group on these axes
    plt.tight_layout()
    ax.legend(loc='upper left', fontsize=25)

    
    
    axs[0].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=22)
    axs[1].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=22)
    axs[2].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=22)
    axs[0].yaxis.set_tick_params(labelbottom=True,  labelcolor="black",labelsize=22)
    axs[1].yaxis.set_tick_params(labelbottom=True,  labelcolor="black",labelsize=22)
    axs[2].yaxis.set_tick_params(labelbottom=True, labelcolor="black",labelsize=22)
    ax.legend( fontsize=30)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.gcf().autofmt_xdate()
    plt.show()
    
        
    figs, axs = plt.subplots(1,1,figsize=(20, 12),sharex=True, sharey=True)
    df_res[(df_res['Dosis']=='Primera')].plot( kind='line', x="start_date",
                              y=group_name_columns,
                              ax=axs,
                              lw=6,
                              label=group_name_columns)
    
    # set the title
    axs.set_xlim([df_res["start_date"].min()+pd.DateOffset(-2),df_res["start_date"].max()+pd.DateOffset(2)])
    #ax.grid()
    
    axs.xaxis_date()

    axs.xaxis.set_major_locator(mdates.MonthLocator(interval=n_moth))
    axs.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    
    axs.axes.set_title("Serie de tiempo de vacunación acumulada a nivel nacional ",fontsize= 30)
    axs.set_ylabel('Nª personsas cada millón de habitantes', fontsize=27)
    #ax.set_xlabel('Date \n', fontsize=20)
    axs.set_xlabel('Tiempo', fontsize=27)
    
    #ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,2)), color="green", alpha=0.3)#label="older adults",
    for i,color in enumerate(['darkorange','green','red','blue']):#purple
        #ax.axvspan(date2num(data_porc[i]), date2num(data_porc[i])+0.5, 
           #color=color, alpha=0.3,linestyle='dashed')#label="older adults",
        #ax.axhline(x=date2num(data_porc[i]), color='r', linestyle='-')
        #hlines=[40,50]
        #ax.vlines(date2num(data_porc[i]), 1, 100, color='g')
        if i in [1,2]:
            axs.axvline(x=date2num(data_porc[i])-0.5,color=color,linestyle='dashed' )
        else:
            axs.axvline(x=date2num(data_porc[i]),color=color,linestyle='dashed' )
    #ax.axhline(y=0.3)
    #ax.label_outer()
    axs.xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=18)
    # plot the group on these axes
    plt.tight_layout()
    axs.legend(loc='upper left', fontsize=25)


    axs.xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=22)
    #axs[1].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=22)
    #axs[2].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=22)
    axs.yaxis.set_tick_params(labelbottom=True,  labelcolor="black",labelsize=22)
    #axs[1].yaxis.set_tick_params(labelbottom=True,  labelcolor="black",labelsize=22)
    #∫axs[2].yaxis.set_tick_params(labelbottom=True, labelcolor="black",labelsize=22)
    #ax.legend( fontsize=30)
    plt.xticks(size = 20)
    plt.yticks(size = 20)
    plt.gcf().autofmt_xdate()
    plt.show()
    
