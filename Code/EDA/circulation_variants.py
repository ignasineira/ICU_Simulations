# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 16:07:42 2021

@author: ignasi
"""
import requests
import io
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

from dateutil.parser import parse
import calendar
import itertools



def ci_circulation_variants():
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]
    if path_projet[-1]!='/':
        path_projet+='/'
    path_projet+='ICU_Simulations/'
    path_data='Data/Input/Circulation_variants/'
    path=path_projet+path_data
    df_P1 = pd.read_csv(path+'informe_variantes_P1.csv')
    df_B117 = pd.read_csv(path+'informe_variantes_B117.csv')
    df_total = pd.read_csv(path+'informe_variantes_total.csv')
    df_P1.rename(columns={'Region':'Month'}, inplace=True)
    df_B117.rename(columns={'Region':'Month'}, inplace=True)
    df_total.rename(columns={'Region':'Month'}, inplace=True)

    path=path_projet+'Data/Input/github_product/'
    df_26=pd.read_csv(path+'Producto_26.csv')
    df_26['start_date']=np.where(df_26['Region'].str[-2:]=='20',df_26['Region']+'20',df_26['Region'].str[:-2]+'2021')
    df_26['start_date']= pd.to_datetime(df_26['start_date'],format="%d-%m-%Y")
    
    
    df_26=df_26[df_26['start_date']>='01-01-2021']
    df_26['Month']=df_26['start_date'].dt.month_name()
    df_26.drop(columns=['Region','start_date'],inplace=True)
    
    columns_name=df_26.columns.tolist()
    month_list=df_26['Month'].unique().tolist()
    df_26=df_26[[columns_name[-1]]+columns_name[:-1]]
    
    
    df_26=df_26.groupby(['Month'])[columns_name[:-1]].agg('sum').reset_index()
    
    
    regiones_to_regionesID = {region:regionID for regionID, region in enumerate(columns_name[:-2])}
    month_to_monthID = {month:monthID for monthID, month in enumerate(month_list[:5])}
    
    inf_nationwide=df_26[df_26['Month'].isin(list(month_to_monthID.keys()))][list(regiones_to_regionesID.keys())].values.T
    inf_total=df_26[df_26['Month'].isin(list(month_to_monthID.keys()))][columns_name[-2]].values
    inf_nationwide=(inf_nationwide.copy()/inf_total)
    
    
    
    info_P1=df_P1[df_P1['Month'].isin(list(month_to_monthID.keys()))][list(regiones_to_regionesID.keys())].values.T
    info_B117=df_B117[df_B117['Month'].isin(list(month_to_monthID.keys()))][list(regiones_to_regionesID.keys())].values.T
    info_total=df_total[df_total['Month'].isin(list(month_to_monthID.keys()))][list(regiones_to_regionesID.keys())].values.T
    
    info_P1=np.nan_to_num(info_P1/info_total)
    info_B117=np.nan_to_num(info_B117/info_total)
    
    
    p_total=np.sum(np.where(info_total>0,inf_nationwide,0), axis=0)
    p_shares_P1_region=info_P1*inf_nationwide/p_total
    p_shares_B117_region=info_B117*inf_nationwide/p_total
    
    
    p_shares_P1=np.sum(p_shares_P1_region,axis=0)
    p_shares_B117=np.sum(p_shares_B117_region,axis=0)
    
    #revisar
    var_p_shares_P1=np.sum(np.nan_to_num((inf_nationwide**2)*p_shares_P1_region*(1-p_shares_P1_region)),axis=0)/np.sum(info_total,axis=0)
    var_p_shares_B117=np.sum(np.nan_to_num((inf_nationwide**2)*p_shares_B117_region*(1-p_shares_B117_region)),axis=0)/np.sum(info_total,axis=0)
    
    std_p_shares_P1=np.sqrt(var_p_shares_P1).round(4)
    std_p_shares_B117=np.sqrt(var_p_shares_B117).round(4)
    
    
    s_list=[]
    for i,(p,std) in enumerate(zip(p_shares_P1.tolist(),std_p_shares_P1.tolist())):
        s = np.random.normal(p, std, 1000)
        s_list.append(s)
    
    return std_p_shares_P1,std_p_shares_B117

def plot_circulation_variants_Uchile(save_image=True, save_table=True):
    """ 
    Main idea:
     circulation of the variants using U. Chile source ( by age group)
     
    1. Generate final plot
    2. Save table count positive case by variante and month
    3. Save json file with circulation variants for icu simulation
    

    Parameters
    ----------
    save_image : boolean, optional
        DESCRIPTION. The default is True.
    save_table : boolean, optional
        DESCRIPTION. The default is True.

    Returns
    -------
    None.

    """
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]
    if path_projet[-1]!='/':
        path_projet+='/'
    path_projet+='ICU_Simulations/'
    path_data='Data/Input/Circulation_variants/'
    df=pd.read_excel(path_projet+path_data+'Base datos variantes Uchile.xlsx')
    
    df[ 'FECHA toma M'] = pd.to_datetime(df[ 'FECHA toma M'],format="%d-%m-%Y")
    df['Fecha Nacimiento'] = pd.to_datetime(df['Fecha Nacimiento'],format="%d-%m-%Y")
    df['edad']=(df[ 'FECHA toma M']-df['Fecha Nacimiento']).astype('timedelta64[Y]')#.astype('int')

    df['week']=df['FECHA toma M'].dt.week#strftime('%w')
    df['month_name']=df['FECHA toma M'].dt.month_name()#strftime('%M')
    df['year']=df['FECHA toma M'].dt.strftime('%Y')
    df.dropna(subset=['Fecha Nacimiento', 'Interpretacion'],inplace=True)
    
    df['Grupo de edad']=np.nan
    df['Grupo de edad']=np.where(df['edad']<=39,'<=39',df['Grupo de edad'] )
    df['Grupo de edad']=np.where((df['edad']>39)&(df['edad']<=49),'40-49',df['Grupo de edad'] )
    df['Grupo de edad']=np.where((df['edad']>49)&(df['edad']<=59),'50-59',df['Grupo de edad'] )
    df['Grupo de edad']=np.where((df['edad']>59)&(df['edad']<=69), '60-69',df['Grupo de edad'] )
    df['Grupo de edad']=np.where(df['edad']>69,'>=70',df['Grupo de edad'] )

    df['Interpretacion']=np.where(df['Interpretacion']=='P.1 Brasil', 'P1',df['Interpretacion'])
    df['Interpretacion']=np.where(df['Interpretacion']=='UK', 'B117',df['Interpretacion'])
    
    df_final=df.groupby(["Grupo de edad",'month_name','Interpretacion']).size().reset_index(name='Count')

    d={i:e for e,i in enumerate(calendar.month_name)}
    #month_list=[item for item in list(d.keys())[1:] if item in df.month.unique().tolist()]
    #df_final['month']= pd.Categorical(df_final['month'],categories=month_list, ordered=True)
    df_final['month']=df_final['month_name'].map(d)
    df_final.sort_values(by=["Grupo de edad",'month'],inplace=True)
    
    
    df_final_toltal=df_final[["Grupo de edad",'month','Count']].groupby(["Grupo de edad",'month']).agg('sum').reset_index()
    df_final_toltal.rename(columns={'Count':'Count_total'},inplace=True)
    
    df_aux=pd.DataFrame(
            list(itertools.product(
                    df_final['Grupo de edad'].unique(),
                   df_final['month'].unique(),
                   df_final['Interpretacion'].unique())))

    df_aux.columns = ['Grupo de edad', 'month','Interpretacion']
    df_final=pd.merge(df_aux,df_final, how='left', on=['Grupo de edad', 'month','Interpretacion'])
    df_final.fillna(0,inplace=True)
    
    df_final=pd.merge(df_final,df_final_toltal,how='left', on=["Grupo de edad",'month'])
    
    
    
    df_final['Porc']=df_final['Count']/df_final['Count_total']
    df_final['h']=1.96*np.sqrt(df_final['Porc']*(1-df_final['Porc'])/df_final['Count_total'])
    df_final['h_+']=df_final['Porc']+df_final['h']
    df_final['h_-']=df_final['Porc']-df_final['h']
    df_res=df_final.pivot_table(index=[ 'month','Grupo de edad'],columns='Interpretacion',values='Porc',fill_value=0).reset_index()
    
    df_plus=df_final.pivot_table(index=[ 'month','Grupo de edad'],columns='Interpretacion',values='h_+',fill_value=0).reset_index()
    
    df_minus=df_final.pivot_table(index=[ 'month','Grupo de edad'],columns='Interpretacion',values='h_-',fill_value=0).reset_index()
    
    df_res['year']=2021
    df_res['day']=15
    
    df_res['start_date']=pd.to_datetime(df_res[ ['year','month','day']])
    n_moth=int(len(df_res['start_date'].dt.strftime("%m/%y").unique().tolist())/5)
    
    
    group_name_position={
        '40-49':(0,1),
        '50-59':(1,0),
        '60-69':(1,1), 
        '<=39':(0,0)
        }
    
    figs, axs = plt.subplots(2,2,figsize=(20*2, 13*2),sharex=True, sharey=False)
    
    for i, (group_name, group_df) in enumerate(df_res.groupby(["Grupo de edad"])):
        if group_name=='>=70':
            #fig, ax = plt.subplots(figsize=(30, 13))
            pass
        else:
            index_group_name=group_name_position[group_name]
            ax=axs[index_group_name]
            group_df.plot( kind='line', x="start_date",
                                 y=[ 'P1', 'B117'],#df_res.columns.tolist()[-4:][ 'P1', 'B117']
                                 ax=ax,
                                 lw=6,
                                 label=[ 'gamma', 'alpha'])
                   #Changing text fontsize in legend
            ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
            ax.xaxis_date()
    
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=n_moth))
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.axes.set_title("Estimated circulation of VoCs during the first six months of 2021 "+group_name +""
                              "\n",fontsize= 30)
            ax.set_ylabel('Percentage of infections', fontsize=27)
            ax.set_xlabel('', fontsize=3)
            ax.legend(loc='upper left', fontsize=25)
            ax.xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=22)
            
            
            
            for j in [ 'P1', 'B117']:
                ax.fill_between( x=group_df["start_date"].values,
                                y1=df_minus[df_minus["Grupo de edad"]==group_name][j],
                                y2=df_plus[df_plus["Grupo de edad"]==group_name][j], 
                               alpha=0.2)
            
            
            #ax.grid(which='major', axis='both', linestyle='--')
            ax.yaxis.grid() # horizontal lines
            plt.tight_layout()
            
    axs[0,1].xaxis.set_tick_params(labelbottom=True, rotation=45, labelcolor="black",labelsize=22)
    axs[0,0].yaxis.set_tick_params(labelbottom=True,  labelcolor="black",labelsize=22)
    axs[0,1].yaxis.set_tick_params(labelbottom=True,  labelcolor="black",labelsize=22)
    axs[1,0].yaxis.set_tick_params(labelbottom=True,  labelcolor="black",labelsize=22)
    axs[1,1].yaxis.set_tick_params(labelbottom=True, labelcolor="black",labelsize=22)
    if save_image:
        
        plt.savefig(path_projet+'Image/EDA/'+"Estimated circulation of VoCs during the first six months of 2021 (UCH).png")
        plt.savefig(path_projet+'Image/Final_plots/'+"Figure 3 Estimated circulation of VoCs during the first six months of 20.png")
    plt.show()
    
    
    if save_table:
        for i, (group_name, group_df) in enumerate(df_final.groupby(['Interpretacion',"Grupo de edad"])):
            if group_name[0] in ['UK','P.1 Brasil']:
                #print(group_name)
                print(np.round(group_df['Porc'].values,3).tolist())
        df_reas_aux=df_final.pivot_table(index=['Grupo de edad'],columns=['month_name', 'Interpretacion'],values='Count',fill_value=0).reset_index()
        
        df_reas_aux.to_csv(path_projet+'Data/Output/Vocs/table_count_vocs.csv',index=False)
    
    df_res.to_csv(path_projet+'Data/Output/Vocs/table_circulation_variants_month.csv',index=False)
    
    
    aux=df_res.pivot_table(index=['Grupo de edad'],columns=['month'],values='B117',fill_value=0)
    #add extra date 2020-12-15
    aux[0]=0
    p_shares_b117=aux.round(3).to_numpy().tolist()
    
    aux=df_res.pivot_table(index=['Grupo de edad'],columns=['month'],values= 'P1',fill_value=0)

    aux[0]=0
    p_shares_new_variant=aux.round(3).to_numpy().tolist()
    list_day=['2020-12-15']+df_res.start_date.dt.strftime('%Y-%m-%d').unique().tolist()
    
    
    
    groups = np.sort(df_res["Grupo de edad"].unique())
    group_to_groupID = {group:groupID for groupID, group in enumerate(groups)}
                        
    circulation_variants_month={
        'p_shares_new_variant':p_shares_new_variant,
        'list_day_new_variant':list_day,
        'p_shares_b117':p_shares_b117,
        'ID_list_day_b117':list_day, 
        'group_to_groupID': group_to_groupID
        }
    today= datetime.now().strftime('%Y-%m-%d')
    print(f"Update the file: {today}")
    
    path_circulation_variants_month=path_projet+"Data/Output/Vocs/circulation_variants_month.json"
    os.makedirs(os.path.dirname(path_circulation_variants_month), exist_ok=True)
    
    with open(path_circulation_variants_month, "w") as out:
        json.dump(circulation_variants_month,out)
    
    last_update_file=path_projet+"Data/Output/Vocs/last_update_circulation_variants_month.json"
    os.makedirs(os.path.dirname(last_update_file), exist_ok=True)
    
    with open(last_update_file, "w") as out:
        json.dump({'last_update':today},out)
    
    
    
if __name__=='__main__':
    plot_circulation_variants_Uchile()


