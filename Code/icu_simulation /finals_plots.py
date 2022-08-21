import requests
import io
import sys
import os
import warnings
warnings.filterwarnings("ignore")

import json
from urllib.request import urlretrieve #open url

import pandas as pd
import numpy as np 

from datetime import datetime,timedelta

#plot package
import matplotlib.pyplot as plt 
from matplotlib.dates import date2num
import matplotlib.dates as mdates
import seaborn as sns

plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
sns.set(rc={"figure.dpi":400, 'savefig.dpi':400})

path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]
if path_projet[-1]!='/':
    path_projet+='/'
path_projet+='ICU_Simulations/'
path_ETL='Code/ETL'

module_path=path_projet+path_ETL
if module_path not in sys.path:
    sys.path.append(module_path)

from data_processing import read_data,prepare_population




#camas_uci=[camas_uci_group_0,camas_uci_group_1,camas_uci_group_2,camas_uci_group_3,camas_uci_group_4]
def plot_uci_pred_final_model_2021(data, uci,
                        camas_uci,
                        W=29, pond=1,
                        start_date='2020-07-01',end_date='2021-05-15',
                        probability_not_vac_new_variant=None):
    
    #mean_to_uci=[10.327, 9.127, 11.543, 7.,477, 10.866]
    #mean_in_uci=[23.307, 23.782, 23.831, 20.489, 27.321]
    mean_to_uci=[10.327, 9.127, 11.543, 7.477, 10.866]
    mean_in_uci=[26.169,  22.294, 26.657,25.319 , 30.453]
    
    groupID_to_group = data['groupID_to_group']
    dateID_to_date = data['dateID_to_date']
    date_to_dateID = data["date_to_dateID"]
    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    cols =['Grupo de edad', 'start_date', 'uci_real', 'uci_pred','uci_pred_05','uci_pred_95','uci_pred_50','uci_pred_mean']
    lst = []
    for key,item in groupID_to_group.items():
        for date in range(dateID_start_date,dateID_end_date+1):
            uci_beds=camas_uci[item]
            uci_pred_05=np.percentile(uci_beds, 5,axis=0)
            uci_pred_95=np.percentile(uci_beds, 95,axis=0)
            uci_pred_50=np.percentile(uci_beds, 50,axis=0)
            uci_pred_mean=np.mean(uci_beds, axis=0,dtype=np.float64)
            
            info = [groupID_to_group[key],dateID_to_date[date],data['uci'][key,date]]
            info.append(uci[key,date-(W-1)])
            info.append(uci_pred_05[date-(W-1)])
            info.append(uci_pred_95[date-(W-1)])
            info.append(uci_pred_50[date]-(W-1))
            info.append(uci_pred_mean[date-(W-1)])
            lst.append(info)
    df_res = pd.DataFrame(lst, columns=cols)
    df_res["start_date"] = pd.to_datetime(df_res.start_date)
    #print(df_res['start_date'].dt.month.unique().tolist())
    print(df_res['start_date'].dt.strftime("%m/%y").unique().tolist())
    
    n_moth=int(len(df_res['start_date'].dt.strftime("%m/%y").unique().tolist())/5)
    
    
    group_name_position={
        '40-49':(1),
        '50-59':(2),
        '60-69':(3), 
        '<=39':(0)
        }
    figs, axs = plt.subplots(4,figsize=(30, 13*4),sharex=True, sharey=False)
    for i, (group_name, group_df) in enumerate(df_res.groupby(["Grupo de edad"])):
        if group_name=='>=70':
            fig, ax = plt.subplots(figsize=(30, 13))
        else:
            index_group_name=group_name_position[group_name]
            ax=axs[index_group_name]
        # plot the group on these axes
        if group_name=='>=70':
            group_df.plot( kind='line', x="start_date",
                          y=['uci_real', 'uci_pred'], ax=ax, 
                          #title="",
                          lw=6,
                          label=['Observed ICU occupancy', 'Expected ICU occupancy'])
            
        else:
            group_df.plot( kind='line', x="start_date",
                            y=['uci_real', 'uci_pred'],
                            ax=ax,
                            lw=6,
                            label=['Observed ICU occupancy', 'Expected ICU occupancy'])
        
        # Provide tick lines across the plot to help your viewers trace along    
        # the axis ticks. Make sure that the lines are light and small so they    
        # don't obscure the primary data lines.    
        
        # Define the upper limit, lower limit, interval of Y axis and colors
        y_LL = 100
        y_UL = int(max(group_df['uci_pred'].max(),group_df['uci_real'].max())*1.1)
        y_interval = 100
        # Draw Tick lines  
        
        if min(group_df['uci_pred'].min(),group_df['uci_real'].min())>200:
            y_LL=200
        for y in range(y_LL, y_UL, y_interval):    
            ax.hlines(y, xmin=group_df["start_date"].min()+pd.DateOffset(-2), xmax=group_df["start_date"].max()+pd.DateOffset(2), colors='black', alpha=0.4, linestyles="--", lw=0.55)
        
        
        # set the title
        ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
        ax.grid()
        
        ax.xaxis_date()

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=n_moth))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        """
        ax.axes.set_title("Timeseries UCI for "+group_name +""
                          "\n"+case_list[case],fontsize= 25)
        """
        ax.axes.set_title("Age: "+group_name ,fontsize= 32)
        ax.set_ylabel('ICU occupancy', fontsize=27)
        ax.set_xlabel('Date \n', fontsize=20)
        """
        ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,2)), 
               label="older adults",color="green", alpha=0.3)
        ax.axvspan(date2num(data_porc[i]), date2num(data_porc[i])+1, 
               label="older adults",color="blue", alpha=0.3)

        
        add_days=21
        ax.axvspan(date2num(data_porc[i])+add_days, date2num(data_porc[i])+1+add_days, 
               label="older adults",color="red", alpha=0.3)
        """
        #ax.label_outer()
        ax.xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=16)
        ax.fill_between( x=group_df["start_date"].values,y1=group_df['uci_pred_05'],y2=group_df['uci_pred_95'], facecolor='orange', alpha=0.2)
        
        """
        
        if case ==2 |case ==3:
            ax.annotate(""
                    "\n* Data source: https://github.com/MinCiencia/Datos-COVID19" 
                    "\n** The flow model was fitted with a probability of going to the ICU:  "+str(round(probability_not_vac[i],3))+""
                    "\n** The flow model was fitted with a probability of going to the ICU for new variant:  "+str(round(probability_not_vac_new_variant[i],3))+""
                    "\n*** The time to ICU is modeled as a negative binomial distribution with mean:  "+str(round(mean_to_uci[i],1))+""
                    "\n*** The time in ICU is modeled as a negative binomial distribution with mean:  "+str(round(mean_in_uci[i],1))
                    , xy=(0.0, -0.3), xycoords="axes fraction",fontsize = 17)#color = '#f0f0f0', backgroundcolor = 'grey'
        else:
            ax.annotate(""
                        "\n* Data source: https://github.com/MinCiencia/Datos-COVID19" 
                        "\n** The flow model was fitted with a probability of going to the ICU:  "+str(round(probability_not_vac[i],3))+""
                        "\n*** The time to ICU is modeled as a negative binomial distribution with mean:  "+str(round(mean_to_uci[i],1))+""
                        "\n*** The time in ICU is modeled as a negative binomial distribution with mean:  "+str(round(mean_in_uci[i],1))
                        , xy=(0.0, -0.25), xycoords="axes fraction",fontsize = 17)#color = '#f0f0f0', backgroundcolor = 'grey'
        """
        # plot the group on these axes
        plt.tight_layout()
    """
        try:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name) +'.png')
        except:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name)[2:] +'.png')
    """
        #plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
    #plt.setp(axs[0,0].get_xticklabels(), visible=False)
    axs[0].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=14)
    axs[1].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=14)
    axs[2].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=14)
    axs[3].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=14)
    plt.gcf().autofmt_xdate()
    plt.show()
    
    group_name_position={
        '40-49':(0,1),
        '50-59':(1,0),
        '60-69':(1,1), 
        '<=39':(0,0)
        }
    figs, axs = plt.subplots(2,2,figsize=(20*2, 13*2),sharex=True, sharey=False)
    for i, (group_name, group_df) in enumerate(df_res.groupby(["Grupo de edad"])):        
        if group_name=='>=70':
            fig, ax = plt.subplots(figsize=(30, 13))
        else:
            index_group_name=group_name_position[group_name]
            ax=axs[index_group_name]
        # plot the group on these axes
        if group_name=='>=70':
            group_df.plot( kind='line', x="start_date",
                          y=['uci_real', 'uci_pred'], ax=ax, 
                          #title="",
                          lw=6,
                          label=['UCI beds today (Producto 9)', 'Predicted UCI beds today (Flow model)'])
            #plt.legend(['UCI beds today (Producto 9)', 'Predicted UCI beds today (Flow model)'], fontsize = 15)
        else:
    
            
            group_df.plot( kind='line', x="start_date",
                            y=['uci_real', 'uci_pred'],
                            ax=ax,
                            lw=6,
                            label=['Observed ICU occupancy', 'Expected ICU occupancy'])
            #Changing text fontsize in legend 
            plt.legend(['Observed ICU occupancy', 'Expected ICU occupancy'], fontsize = 15)
        
        # Provide tick lines across the plot to help your viewers trace along    
        # the axis ticks. Make sure that the lines are light and small so they    
        # don't obscure the primary data lines.    
        
        # Define the upper limit, lower limit, interval of Y axis and colors
        y_LL = 100
        y_UL = int(max(group_df['uci_pred'].max(),group_df['uci_real'].max())*1.1)
        y_interval = 100
        # Draw Tick lines  
        
        if min(group_df['uci_pred'].min(),group_df['uci_real'].min())>200:
            y_LL=200
        for y in range(y_LL, y_UL, y_interval):    
            ax.hlines(y, xmin=group_df["start_date"].min()+pd.DateOffset(-2), xmax=group_df["start_date"].max()+pd.DateOffset(2), colors='black', alpha=0.4, linestyles="--", lw=0.55)
        
        
        # set the title
        ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
        ax.grid()
        
        ax.xaxis_date()

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        ax.axes.set_title("Timeseries UCI for "+group_name,fontsize= 26)
        
        ax.set_ylabel('N° beds', fontsize=27)
        #ax.set_xlabel('Date \n', fontsize=20)
        ax.set_xlabel('', fontsize=27)
        
        ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,2)), 
              color="green", alpha=0.3)
        """
        ax.axvspan(date2num(data_porc[i]), date2num(data_porc[i])+1, 
               label="older adults",color="blue", alpha=0.3)

        add_days=21
        ax.axvspan(date2num(data_porc[i])+add_days, date2num(data_porc[i])+1+add_days, 
               label="older adults",color="red", alpha=0.3)
        """
        #ax.label_outer()
        ax.xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=14)
        ax.fill_between( x=group_df["start_date"].values,y1=group_df['uci_pred_05'],y2=group_df['uci_pred_95'], facecolor='orange', alpha=0.2)

        plt.tight_layout()
        ax.legend(loc='upper left', fontsize=25)
        ax.yaxis.set_tick_params(labelbottom=True,  labelcolor="black",labelsize=22)
        plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
    """
        try:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name) +'.png')
        except:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name)[2:] +'.png')
    """
        
    #plt.setp(axs[0,0].get_xticklabels(), visible=False)
    for i in range(2):
        for j in range(2):
            axs[i,j].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=15)
    axs[1,1].legend(loc='upper left', fontsize=25)
    plt.gcf().autofmt_xdate()
    
    figs.savefig(path_projet+'Image/Final_plots/'+"Figure 4 Observed and predicted (expected) ICU occupancy by age during 2020 and 2021 adjusting for VOCs.png")

    
    plt.show()

def plot_uci_pred_final_model_2020(data, uci,
                        camas_uci,
                        W=29, pond=1,
                        start_date='2020-07-01',end_date='2021-05-15',
                        probability_not_vac_new_variant=None):
    
    #mean_to_uci=[10.327, 9.127, 11.543, 7.,477, 10.866]
    #mean_in_uci=[23.307, 23.782, 23.831, 20.489, 27.321]
    mean_to_uci=[10.327, 9.127, 11.543, 7.477, 10.866]
    mean_in_uci=[26.169,  22.294, 26.657,25.319 , 30.453]
    
    groupID_to_group = data['groupID_to_group']
    dateID_to_date = data['dateID_to_date']
    date_to_dateID = data["date_to_dateID"]
    dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
    dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
    cols =['Grupo de edad', 'start_date', 'uci_real', 'uci_pred','uci_pred_05','uci_pred_95','uci_pred_50','uci_pred_mean']
    lst = []
    for key,item in groupID_to_group.items():
        for date in range(dateID_start_date,dateID_end_date+1):
            uci_beds=camas_uci[item]
            uci_pred_05=np.percentile(uci_beds, 5,axis=0)
            uci_pred_95=np.percentile(uci_beds, 95,axis=0)
            uci_pred_50=np.percentile(uci_beds, 50,axis=0)
            uci_pred_mean=np.mean(uci_beds, axis=0,dtype=np.float64)
            
            info = [groupID_to_group[key],dateID_to_date[date],data['uci'][key,date]]
            info.append(uci[key,date-(W-1)])
            info.append(uci_pred_05[date-(W-1)])
            info.append(uci_pred_95[date-(W-1)])
            info.append(uci_pred_50[date]-(W-1))
            info.append(uci_pred_mean[date-(W-1)])
            lst.append(info)
    df_res = pd.DataFrame(lst, columns=cols)
    df_res["start_date"] = pd.to_datetime(df_res.start_date)
    #print(df_res['start_date'].dt.month.unique().tolist())
    print(df_res['start_date'].dt.strftime("%m/%y").unique().tolist())
    
    n_moth=int(len(df_res['start_date'].dt.strftime("%m/%y").unique().tolist())/5)
    
    
    group_name_position={
        '40-49':(1),
        '50-59':(2),
        '60-69':(3), 
        '<=39':(0)
        }
    figs, axs = plt.subplots(4,figsize=(30, 13*4),sharex=True, sharey=False)
    for i, (group_name, group_df) in enumerate(df_res.groupby(["Grupo de edad"])):
        if group_name=='>=70':
            fig, ax = plt.subplots(figsize=(30, 13))
        else:
            index_group_name=group_name_position[group_name]
            ax=axs[index_group_name]
        # plot the group on these axes
        if group_name=='>=70':
            group_df.plot( kind='line', x="start_date",
                          y=['uci_real', 'uci_pred'], ax=ax, 
                          #title="",
                          lw=6,
                          label=['Observed ICU occupancy', 'Expected ICU occupancy'])
            
        else:
            group_df.plot( kind='line', x="start_date",
                            y=['uci_real', 'uci_pred'],
                            ax=ax,
                            lw=6,
                            label=['Observed ICU occupancy', 'Expected ICU occupancy'])
        
        # Provide tick lines across the plot to help your viewers trace along    
        # the axis ticks. Make sure that the lines are light and small so they    
        # don't obscure the primary data lines.    
        
        # Define the upper limit, lower limit, interval of Y axis and colors
        y_LL = 100
        y_UL = int(max(group_df['uci_pred'].max(),group_df['uci_real'].max())*1.1)
        y_interval = 100
        # Draw Tick lines  
        
        if min(group_df['uci_pred'].min(),group_df['uci_real'].min())>200:
            y_LL=200
        for y in range(y_LL, y_UL, y_interval):    
            ax.hlines(y, xmin=group_df["start_date"].min()+pd.DateOffset(-2), xmax=group_df["start_date"].max()+pd.DateOffset(2), colors='black', alpha=0.4, linestyles="--", lw=0.55)
        
        
        # set the title
        ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
        ax.grid()
        
        ax.xaxis_date()

        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=n_moth))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        """
        ax.axes.set_title("Timeseries UCI for "+group_name +""
                          "\n"+case_list[case],fontsize= 25)
        """
        ax.axes.set_title("Age: "+group_name ,fontsize= 32)
        ax.set_ylabel('ICU occupancy', fontsize=27)
        ax.set_xlabel('Date \n', fontsize=20)
        """
        ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,2)), 
               label="older adults",color="green", alpha=0.3)
        ax.axvspan(date2num(data_porc[i]), date2num(data_porc[i])+1, 
               label="older adults",color="blue", alpha=0.3)

        
        add_days=21
        ax.axvspan(date2num(data_porc[i])+add_days, date2num(data_porc[i])+1+add_days, 
               label="older adults",color="red", alpha=0.3)
        """
        #ax.label_outer()
        ax.xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=16)
        ax.fill_between( x=group_df["start_date"].values,y1=group_df['uci_pred_05'],y2=group_df['uci_pred_95'], facecolor='orange', alpha=0.2)
        
        """
        
        if case ==2 |case ==3:
            ax.annotate(""
                    "\n* Data source: https://github.com/MinCiencia/Datos-COVID19" 
                    "\n** The flow model was fitted with a probability of going to the ICU:  "+str(round(probability_not_vac[i],3))+""
                    "\n** The flow model was fitted with a probability of going to the ICU for new variant:  "+str(round(probability_not_vac_new_variant[i],3))+""
                    "\n*** The time to ICU is modeled as a negative binomial distribution with mean:  "+str(round(mean_to_uci[i],1))+""
                    "\n*** The time in ICU is modeled as a negative binomial distribution with mean:  "+str(round(mean_in_uci[i],1))
                    , xy=(0.0, -0.3), xycoords="axes fraction",fontsize = 17)#color = '#f0f0f0', backgroundcolor = 'grey'
        else:
            ax.annotate(""
                        "\n* Data source: https://github.com/MinCiencia/Datos-COVID19" 
                        "\n** The flow model was fitted with a probability of going to the ICU:  "+str(round(probability_not_vac[i],3))+""
                        "\n*** The time to ICU is modeled as a negative binomial distribution with mean:  "+str(round(mean_to_uci[i],1))+""
                        "\n*** The time in ICU is modeled as a negative binomial distribution with mean:  "+str(round(mean_in_uci[i],1))
                        , xy=(0.0, -0.25), xycoords="axes fraction",fontsize = 17)#color = '#f0f0f0', backgroundcolor = 'grey'
        """
        # plot the group on these axes
        plt.tight_layout()
    """
        try:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name) +'.png')
        except:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name)[2:] +'.png')
    """
        #plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
    #plt.setp(axs[0,0].get_xticklabels(), visible=False)
    axs[0].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=14)
    axs[1].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=14)
    axs[2].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=14)
    axs[3].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=14)
    plt.gcf().autofmt_xdate()
    plt.show()
    
    group_name_position={
        '40-49':(0,1),
        '50-59':(1,0),
        '60-69':(1,1), 
        '<=39':(0,0)
        }
    figs, axs = plt.subplots(2,2,figsize=(20*2, 13*2),sharex=True, sharey=False)
    for i, (group_name, group_df) in enumerate(df_res.groupby(["Grupo de edad"])):        
        if group_name=='>=70':
            fig, ax = plt.subplots(figsize=(30, 13))
        else:
            index_group_name=group_name_position[group_name]
            ax=axs[index_group_name]
        # plot the group on these axes
        if group_name=='>=70':
            group_df.plot( kind='line', x="start_date",
                          y=['uci_real', 'uci_pred'], ax=ax, 
                          #title="",
                          lw=6,
                          label=['UCI beds today (Producto 9)', 'Predicted UCI beds today (Flow model)'])
            #plt.legend(['UCI beds today (Producto 9)', 'Predicted UCI beds today (Flow model)'], fontsize = 15)
        else:
    
            
            group_df.plot( kind='line', x="start_date",
                            y=['uci_real', 'uci_pred'],
                            ax=ax,
                            lw=6,
                            label=['Observed ICU occupancy', 'Expected ICU occupancy'])
            #Changing text fontsize in legend 
            plt.legend(['Observed ICU occupancy', 'Expected ICU occupancy'], fontsize = 15)
        
        # Provide tick lines across the plot to help your viewers trace along    
        # the axis ticks. Make sure that the lines are light and small so they    
        # don't obscure the primary data lines.    
        
        # Define the upper limit, lower limit, interval of Y axis and colors
        y_LL = 100
        y_UL = int(max(group_df['uci_pred'].max(),group_df['uci_real'].max())*1.1)
        y_interval = 100
        # Draw Tick lines  
        
        if min(group_df['uci_pred'].min(),group_df['uci_real'].min())>200:
            y_LL=200
        for y in range(y_LL, y_UL, y_interval):    
            ax.hlines(y, xmin=group_df["start_date"].min()+pd.DateOffset(-2), xmax=group_df["start_date"].max()+pd.DateOffset(2), colors='black', alpha=0.4, linestyles="--", lw=0.55)
        
        
        # set the title
        ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
        ax.grid()
        
        ax.xaxis_date()

        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        
        ax.axes.set_title("Timeseries UCI for "+group_name,fontsize= 26)
        
        ax.set_ylabel('N° beds', fontsize=27)
        #ax.set_xlabel('Date \n', fontsize=20)
        ax.set_xlabel('', fontsize=27)
        
        ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,2)), 
              color="green", alpha=0.3)
        """
        ax.axvspan(date2num(data_porc[i]), date2num(data_porc[i])+1, 
               label="older adults",color="blue", alpha=0.3)

        add_days=21
        ax.axvspan(date2num(data_porc[i])+add_days, date2num(data_porc[i])+1+add_days, 
               label="older adults",color="red", alpha=0.3)
        """
        #ax.label_outer()
        ax.xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=14)
        ax.fill_between( x=group_df["start_date"].values,y1=group_df['uci_pred_05'],y2=group_df['uci_pred_95'], facecolor='orange', alpha=0.2)

        plt.tight_layout()
        ax.legend(loc='upper left', fontsize=25)
        ax.yaxis.set_tick_params(labelbottom=True,  labelcolor="black",labelsize=22)
        plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
    """
        try:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name) +'.png')
        except:
            plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name)[2:] +'.png')
    """
        
    #plt.setp(axs[0,0].get_xticklabels(), visible=False)
    for i in range(2):
        for j in range(2):
            axs[i,j].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=15)
    axs[1,1].legend(loc='upper left', fontsize=25)
    plt.gcf().autofmt_xdate()
    
    figs.savefig(path_projet+'Image/Final_plots/'+"Figure 2 Observed and predicted (expected) ICU occupancy by age during 2020 and 2021.png")

    
    plt.show()