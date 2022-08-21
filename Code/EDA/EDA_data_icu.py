# -*- coding: utf-8 -*-
"""
Created on Mon May 10 10:58:58 2021

@author: ignas

obs=[19, 54, 65, 20, 59]
mean_in_uci=[26.169,  22.294, 26.657,25.319 , 30.453]# 
mean_in_uci=[3.0375,
 2.715088383838384,
 4.194313417190775,
 1.7893663194444445,
 4.465447845804989]
mean_to_uci=[10.327, 9.127, 11.543, 7.477, 10.866]

negative binomial 
p_in_uci= [0.05105597703344134,
 0.07016922366801419,
 0.051343040281427714,
 0.049618536915375266,
 0.04240692648425665]

p_in_uci=[round(num, 4) for num in p_in_uci] #[0.0511, 0.0702, 0.0513, 0.0496, 0.0424]

n_in_uci=[1.4079851475669516,
 1.9842546575184772,
 1.442722322721286,
 1.3218558234371203,
 1.3486142688140486]

n_in_uci=[round(num, 4) for num in n_in_uci]#[1.408, 1.9843, 1.4427, 1.3219, 1.3486]

p_to_uci=[0.432730994787251,
 0.20674466013386023,
 0.11251935125423387,
 0.3756948544519454,
 0.254068260971279]

p_to_uci=[round(num, 4) for num in p_to_uci]#[0.4327, 0.2067, 0.1125, 0.3757, 0.2541]


n_to_uci=[8.09184723041811,
 2.6380152428163095,
 1.6356501123883693,
 4.827138775907815,
 3.891130257758304]

n_to_uci=[round(num, 4) for num in n_to_uci]#[8.0918, 2.638, 1.6357, 4.8271, 3.8911]

"""
import os
import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd 
import numpy as np
from scipy.stats import poisson, geom,nbinom


import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import matplotlib.dates as mdates

plt.style.use('bmh')
plt.rcParams['figure.dpi'] = 400
plt.rcParams['savefig.dpi'] = 400
sns.set(rc={"figure.dpi":400, 'savefig.dpi':400})

def data(show_plot=False):
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'/ICU_Simulations/'
    path_data="Data/Input/UC_data/"
    path=path_projet+path_data
    df = pd.read_excel(path+'BaseCOVID19UPC_DATA_2020.xlsx')
    
    columns= df.columns
    
    date_time_columns=[]
    for column in columns:
        if df.dtypes[column] == '<M8[ns]':
            df[column]=pd.to_datetime(df[column])
            date_time_columns.append(column)
    
    
    
    time_diff_columns=[]
    for i in range(len(date_time_columns)-1):
        for j in range(i+1,len(date_time_columns)):
            name_column= "T. "+ date_time_columns[j]+" desde "+date_time_columns[i]
            time_diff_columns.append(name_column)
            print(name_column)
            df[name_column] = (df[date_time_columns[j]] -df[date_time_columns[i]])/ np.timedelta64(1, 'D')
            if show_plot:
                plt.figure(figsize=(9, 8))
                sns.distplot(df[name_column], color='g', bins=100, hist_kws={'alpha': 0.4})
                plt.savefig(path_projet+'Image/EDA/Distibution'+name_column.replace(" ","_")+'.png')
                plt.title(name_column)
                plt.show()
    
    #df[date_time_columns].hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
    #df[time_diff_columns].hist(figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)
    
    df.rename(columns={'EDAD': 'edad'}, inplace=True)
    #plt.figure(figsize=(9, 8))
    #sns.distplot(df['edad'], color='g', bins=100, hist_kws={'alpha': 0.4})
    #plt.show()
    
    df['Grupo de edad']=np.nan
    df['Grupo de edad']=np.where(df['edad']<=39,'<=39',df['Grupo de edad'] )
    df['Grupo de edad']=np.where((df['edad']>39)&(df['edad']<=49),'40-49',df['Grupo de edad'] )
    df['Grupo de edad']=np.where((df['edad']>49)&(df['edad']<=59),'50-59',df['Grupo de edad'] )
    df['Grupo de edad']=np.where((df['edad']>59)&(df['edad']<=69), '60-69',df['Grupo de edad'] )
    df['Grupo de edad']=np.where(df['edad']>69,'>=70',df['Grupo de edad'] )

    return df,time_diff_columns

if __name__=='__main__':
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    df,time_diff_columns =data()

    fig, axes = plt.subplots(round(len(time_diff_columns) / 3), 3, figsize=(12, 30))

    for i, ax in enumerate(fig.axes):
        if i < len(time_diff_columns):
            ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
            sns.boxplot(x='Grupo de edad', y=time_diff_columns[i], data=df,ax=ax)
            plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
            plt.xticks(rotation=45)

    fig.tight_layout()
    plt.savefig(path_projet+'Image/EDA/data_UC_Boxplot by age group.png')

    fig, axes = plt.subplots(round(len(time_diff_columns) / 3), 3, figsize=(12, 30))

    for i, ax in enumerate(fig.axes):
        if i < len(time_diff_columns):
            ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
            sns.boxplot(x='Grupo de edad', y=time_diff_columns[i], data=df[(df[time_diff_columns] >= 0).all(axis=1)],ax=ax)
            plt.setp(ax.artists, alpha=.5, linewidth=2, edgecolor="k")
            plt.xticks(rotation=45)

    fig.tight_layout()
    plt.savefig(path_projet+'Image/EDA/data_UC_Boxplot by age group without outliers.png')       
    #first iterations where the times were distributed geo
    list_mu_in_uci=[15, 16,18.5, 14, 20]
    mean_in_uci_poi=[12.307, 12.782, 14.831, 11.489, 15.321]
    mean_in_uci_geo=[37.307, 37.782, 38.831, 31.489, 45.321]
    list_ponderaciones_in_uci=[0.6, 0.63, 0.63, 0.6, 0.85]
    list_group_id=['40-49', '50-59','60-69', '<=39','>=70']
    max_days_in_uci=100
    x = np.arange(max_days_in_uci)
    p_in_uci=[]
    n_in_uci=[]
    for i, (group_name, group_df) in enumerate(df.groupby(["Grupo de edad"])):
        print(group_name)
        poi_param= list_mu_in_uci[list_group_id.index(group_name)]
        rv=poisson(poi_param)
        fig, ax = plt.subplots(figsize=(8, 4))
        bins=max(len(group_df)//2-6,11)
        ax.hist(group_df['T. EGRESO UCI desde INGRESO UCI'], bins, density=True, histtype='step',
                                cumulative=False, label='Empirical')
        
        ax.plot(x, rv.pmf(x)/rv.cdf(max_days_in_uci-1),'k--', linewidth=1.5, label='Theoretical')
        pond= list_ponderaciones_in_uci[list_group_id.index(group_name)]
        rv_poi=poisson(mean_in_uci_poi[i])
        rv_geo=geom(1/mean_in_uci_geo[i])
        aux=(pond)*(rv_geo.pmf(x)/rv_geo.cdf(max_days_in_uci-1))+(1-pond)*(rv_poi.pmf(x)/rv_poi.cdf(max_days_in_uci-1))
        print(np.multiply(x,aux).sum())
        ax.plot(x, aux,'k--',color='green', linewidth=1.5, label='Theoretical 2.0')
        x1= pd.Series(group_df['T. EGRESO UCI desde INGRESO UCI'])
        mean = x1.mean()
        if group_name=='50-59':
            mean+=4
        var=x1.var()
        p=mean/var
        n=(mean*p)/(1-p)
        print(mean)
        print(n,p)
        p_in_uci.append(p)
        n_in_uci.append(n)
        rv_nb=nbinom(n, p)
        aux=rv_nb.pmf(x)/rv_nb.cdf(max_days_in_uci-1)
        ax.plot(x, aux,'k--',color='blue', linewidth=1.5, label='Theoretical 3.0 (bn)')
        
        ax.grid(True)
        ax.legend(loc='right')
        ax.set_title('PDF T. EGRESO UCI desde INGRESO UCI '+group_name )
        ax.set_xlabel('T. EGRESO UCI desde INGRESO UCI')
        #ax.set_ylabel('Likelihood of occurrence')
        ax.set_ylabel('Density')
        plt.show()


    geo_param=4.1
    poi_param=9.2
    list_ponderaciones_to_uci=[0.5, 0.65, 0.65, 0.4, 0.75]
    mean_to_uci_v2_poi=[14.327, 15.127, 16.543, 14., 16.866]
    mean_to_uci_v2_poi2=[6.327, 5.4, 6.5, 6.3, 6.5]
    list_ponderaciones_to_uci_v2=[0.5,.62,0.5,0.5,0.55]
    max_days_to_uci=50
    x = np.arange(max_days_to_uci)
    p_to_uci=[]
    n_to_uci=[]
    for i, (group_name, group_df) in enumerate(df.groupby(["Grupo de edad"])):
        print(group_name)
        pond= list_ponderaciones_to_uci[list_group_id.index(group_name)]
        fig, ax = plt.subplots(figsize=(8, 4))
        bins=max(len(group_df)//2,12)
        #print(bins)
        ax.hist(group_df['T. INGRESO UCI desde INICIO DE LOS SINTOMAS'], bins, density=True, histtype='step',
                                cumulative=False, label='Empirical')
                                
        rv_geo=geom(1/geo_param)
        rv_poi=poisson(poi_param)
        aux=(pond)*(rv_geo.pmf(x)/rv_geo.cdf(max_days_to_uci-1))+(1-pond)*(rv_poi.pmf(x)/rv_poi.cdf(max_days_to_uci-1))
        ax.plot(x+3, aux,'k--', linewidth=1.5, label='Theoretical')
        x1= pd.Series(group_df['T. INGRESO UCI desde INICIO DE LOS SINTOMAS'])
        mean = x1.mean()
        p = 1 / mean
        print("New value "+ str(mean))
        pond= list_ponderaciones_to_uci_v2[list_group_id.index(group_name)]
        rv_poi1=poisson(mean_to_uci_v2_poi[i])
        rv_poi2=poisson(mean_to_uci_v2_poi2[i])
        aux=(pond)*(rv_poi2.pmf(x)/rv_poi2.cdf(max_days_to_uci-1))+(1-pond)*(rv_poi1.pmf(x)/rv_poi1.cdf(max_days_to_uci-1))
        ax.plot(x, aux,'k--',color='green', linewidth=1.5, label='Theoretical 2.0')
        #print(np.multiply(x,aux).sum())
        x1= pd.Series(group_df['T. INGRESO UCI desde INICIO DE LOS SINTOMAS'])
        mean = x1.mean()
        var=x1.var()
        p=mean/var
        n=(mean*p)/(1-p)
        print(n,p)
        p_to_uci.append(p)
        n_to_uci.append(n)
        rv_nb=nbinom(n, p)
        aux=rv_nb.pmf(x)/rv_nb.cdf(max_days_to_uci-1)
        ax.plot(x, aux,'k--',color='blue', linewidth=1.5, label='Theoretical 3.0 (bn)')
        ax.grid(True)
        ax.legend(loc='right')
        ax.set_title('PDF T. INGRESO UCI desde INICIO DE LOS SINTOMAS '+group_name )
        ax.set_xlabel('T. INGRESO UCI desde INICIO DE LOS SINTOMAS')
        #ax.set_ylabel('Likelihood of occurrence')
        ax.set_ylabel('Density')
        plt.show()
        
    # Tableau 20 Colors
    tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),  
                (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),  
                (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),  
                (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),  
                (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]


    # Rescale to values between 0 and 1 
    for i in range(len(tableau20)):  
        r, g, b = tableau20[i]  
        tableau20[i] = (r / 255., g / 255., b / 255.)

    colors = tableau20[::2]


    fig, ax = plt.subplots()
    for i, (group_name, group_df) in enumerate(df.groupby(["Grupo de edad"])):
        print(group_name)
        group_df.plot.scatter( x="INGRESO UCI", y='T. EGRESO UCI desde INGRESO UCI',ax=ax,label=group_name,color=colors[i])
    ax.set_xlim([group_df["INGRESO UCI"].min()+pd.DateOffset(-2),group_df["INGRESO UCI"].max()+pd.DateOffset(2)])
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.axes.set_title("Scatter for time in UCI" ,fontsize=15)
    plt.tight_layout()
    plt.gcf().autofmt_xdate()

    plt.savefig(path_projet+'Image/EDA/data_UC_Scatter for time in UCI.png')
    plt.show()


    fig, axes = plt.subplots(5, 1, figsize=(16, 12*5))
    for i, (group_name, group_df) in enumerate(df.groupby(["Grupo de edad"])):
        print(group_name)
        print(group_df.shape)
        group_df.plot.scatter( x="INGRESO UCI", y='T. EGRESO UCI desde INGRESO UCI',ax=axes[i],label=group_name,color=colors[i],s=100)
        ax=axes[i]
        ax.set_xlim([group_df["INGRESO UCI"].min()+pd.DateOffset(-2),group_df["INGRESO UCI"].max()+pd.DateOffset(2)])
        #ax.grid()
        
        ax.xaxis_date()
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.axes.set_title("Scatter for time in UCI for "+group_name ,fontsize=15)
        plt.tight_layout()
        plt.gcf().autofmt_xdate()
        
    plt.savefig(path_projet+'Image/EDA/data_UC_Scatter for time in UCI all groups.png')
    plt.show()  
    
