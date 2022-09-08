# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 18:38:29 2021

@author: ignasi


#df_9=producto_9()
#df_16 = prepare_producto_16()
#df_26 = producto_26()
#df_population = prepare_population()
#path="Input/"
#dict_between_df_9_16 = pd.read_csv(path+'dict_between_df_9_16_over_15_yearscsv')
# df_77 = df_77 = git_hub_commit.producto_77()

	# Get {date:dateID} dictionary from infections data
#dates = np.sort(df_inf["start_date"].unique())
#date_to_dateID = {date:dateID for dateID, date in enumerate(dates)}

#date_to_dateID, group_to_groupID, dosis_to_dosisID


#df_9.to_csv(path+'Producto_9.csv',index=False)
#df_16.to_csv(path+'Producto_16.csv',index=False)

"""
import os
import warnings
import numpy as np
import pandas as pd
import itertools
import time
# OWN MODULES
from call_vaccine_campaign import Avance_vacunacion
from products_min_ciencia import producto_5,producto_9,producto_10,producto_16,producto_21, producto_26, producto_39,prodcuto_57, producto_77
from call_vocs_circulation import call_vocs_circulation

import matplotlib.pyplot as plt 
from matplotlib.dates import date2num
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

def read_data(sintomatic_value=True, git_hub_info=False, max_date_check='2021-07-01'):
    """
    Main idea:
       1.Extract data.
       2.Transform data.
       3.Loads data.

    Parameters
    ----------
    sintomatic_value : boolean, optional
        On infected data ( product 16) If true use column infected_sintomatic_today 
        else infected_today. The default is True.
    git_hub_info : boolean, optional
        depends on the data source we choose for the vaccination data. 
        If False use Avance_vaccination, else producto_77. The default is False.
    Returns
    -------
    data : dict
    A dict of containing all data.
    """
    start_time = time.time()
    # call dataframe
    df_9 = producto_9(max_date_check=max_date_check)
    df_10 = prepare_producto_10()
    df_16 = prepare_producto_16(max_date_check=max_date_check)
    df_57 = prepare_producto_57()
    df_pop = prepare_population()
    if git_hub_info:
        df_77 = producto_77()
        # Filter data for region
        df_77 = df_77 [df_77.Region=='Total']
        lab_to_labID=None

    else:
        df_77=Avance_vacunacion()
        # Get {dosis:dosisID} dictionary from vaccinated
        lab = np.sort(df_77["Laboratorio"].unique())
        lab_to_labID = {lab:labID for labID, lab in enumerate(lab)}
    
    
    #save date start second dosis
    date_firt_dosis = df_77[(df_77.Dosis=="Primera")&(df_77.accumulated_vaccinated>0)]['start_date'].unique()[0]
    date_second_dosis = df_77[(df_77.Dosis=="Segunda")&(df_77.accumulated_vaccinated>0)]['start_date'].unique()[0]
    
    # Get {date:dateID} dictionary from infections data
    dates = np.sort(np.intersect1d(df_16["start_date"].unique(),df_9["start_date"].unique()))
    date_to_dateID = {date:dateID for dateID, date in enumerate(dates)}
    
    dateID_firt_dosis = date_to_dateID[date_firt_dosis]
    dateID_second_dosis = date_to_dateID[date_second_dosis]
    
    # Get {group:groupID} dictionary from uci beds
    groups = np.sort(df_9["Grupo de edad"].unique())
    group_to_groupID = {group:groupID for groupID, group in enumerate(groups)}
    
    # Get {dosis:dosisID} dictionary from vaccinated
    dosis = np.sort(df_77["Dosis"].unique())
    dosis_to_dosisID = {dosis:dosisID for dosisID, dosis in enumerate(dosis)}
    
    
    # Filter for our data range 
    df_9 = df_9[df_9["start_date"].isin(date_to_dateID)]
    
    #dead time sries is shorter than the others
# =============================================================================
#     df_10 = pd.merge(df_16[["start_date",'Grupo de edad']],
#                      df_10, on=['start_date','Grupo de edad'], 
#                      how='left').fillna(0)
# =============================================================================
    df_10 = df_10[df_10["start_date"].isin(date_to_dateID)]
    df_16 = df_16[df_16["start_date"].isin(date_to_dateID)]
    
    df_57 = df_57[df_57["start_date"].isin(date_to_dateID)]
    df_77 = df_77[df_77["start_date"].isin(date_to_dateID)]
    
    #prepare new dataframe
    df_not_vacc = prepare_not_vac(df_77, df_pop)
    
    
    # Get data matrices
    pop = get_pop(df_pop)
    uci = get_uci(df_9)
    dead = get_dead(df_10)
    inf = get_inf(df_16,sintomatic_value)
    dead_icu = get_dead_icu(df_57)
    vac,vac_acc = get_vac(df_77, date_to_dateID, group_to_groupID, dosis_to_dosisID,lab_to_labID,git_hub_info)
    not_vac = get_not_vac(df_not_vacc,pop, date_to_dateID, group_to_groupID)
    
    
    # Useful dictionaries for external analysis
    dateID_to_date = {dateID:date for dateID, date in enumerate(dates)}
    groupID_to_group = {groupID:group for groupID, group in enumerate(groups)}
    
    data = {'vac':vac, 'inf':inf, 'uci':uci,'dead':dead,'dead_icu':dead_icu,
            'pop':pop, 
            'not_vac': not_vac,'vac_acc': vac_acc,
            'dateID_firt_dosis' : dateID_firt_dosis,
            'dateID_second_dosis' : dateID_second_dosis,
            'group_to_groupID':group_to_groupID,
            'groupID_to_group':groupID_to_group, 
            'dateID_to_date':dateID_to_date, 
            'date_to_dateID':date_to_dateID,
            'vocs_circulation':call_vocs_circulation()}
    
    
    if git_hub_info==False:

        data['lab_to_labID']=lab_to_labID
    
    end_time = round(time.time()-start_time,2)
    #print("Producto 9 is ready")
    print("Execution time:" + str(end_time) +" s.")
    print("Data is  ready")
    print("="*40)
    return data




def get_pop(df_pop):
  """Get population data.

	Parameters
	----------
	df_pop : pandas.DataFrame
		DataFrame containing population features.

	Returns
	-------
	pop : numpy.array-->(G,)
		Array containing population features.
    
  """
  
  df_pop_copy = df_pop.copy()
  df_pop_copy = df_pop_copy.sort_values(by=["Grupo de edad"], ascending=True)
  pop = df_pop_copy.iloc[:,1].to_numpy()
  return pop

def get_uci(df_9):
    """
    Get uci data.

	Parameters
	----------
	df_9 : pandas.DataFrame
		DataFrame containing uci features.

	Returns
	-------
	uci : numpy.array-->(G,T)
		Array containing population features.
    """
    df_uci_copy = df_9.copy()
    df_uci_copy = df_uci_copy.sort_values(by=["Grupo de edad","start_date"], ascending=True)
    df_uci_copy = df_uci_copy.pivot(index="Grupo de edad",columns="start_date",values="uci_beds")
    uci = df_uci_copy.to_numpy()
    return uci


def get_dead(df_10):
    """
    Get inf data.
    
	Parameters
	----------
	df_10 : pandas.DataFrame
		DataFrame containing dead features.

	Returns
	-------
	dead : numpy.array-->(G,T)
		Array containing dead features.
    """
    df_dead_copy = df_10.copy()
    df_dead_copy = df_dead_copy.sort_values(by=["Grupo de edad","start_date"], ascending=True)
    df_dead_copy = df_dead_copy.pivot(index="Grupo de edad",columns="start_date",values="dead_today")
    dead = df_dead_copy.to_numpy()
    
    return dead


def get_inf(df_16,sintomatic_value=True):

    """
    Get inf data.
    ** only give three features "Grupo de edad","start_date", "infected_sintomatic_today"
	Parameters
	----------
	df_16 : pandas.DataFrame
		DataFrame containing infected features.
    sintomatic_value : boolean, optional
        If true use column infected_sintomatic_today else infected_today.
        The default is True.

	Returns
	-------
	dem : numpy.array-->(G,T)
		Array containing infected features.
    """
    df_inf_copy = df_16.copy()
    df_inf_copy = df_inf_copy.sort_values(by=["Grupo de edad","start_date"], ascending=True)
    if sintomatic_value:
        df_inf_copy = df_inf_copy.pivot(index="Grupo de edad",columns="start_date",values="infected_sintomatic_today")
    else:
        df_inf_copy = df_inf_copy.pivot(index="Grupo de edad",columns="start_date",values="infected_today")
    inf = df_inf_copy.to_numpy()
    
    inf[:,48:52]=np.expand_dims((inf[:,47]+inf[:,52])/2,axis=(-1))
    
    
    return inf

def get_vac(df_77, date_to_dateID, group_to_groupID, dosis_to_dosisID,lab_to_labID,git_hub_info):
    """
    Create lab-dosis-group-date matrix: M[l,i,j,t] 
    
    Parameters
    ----------
    df_77 : pandas.DataFrame
        Dataframe of accumulated vaccinated data at age group.
    
    date_to_dateID : dict
        Dictionary mapping date to dateID.	
    group_to_groupID : dict
        Dictionary mapping group to groupID.
    dosis_to_dosisID : dict
        Dictionary mapping dosis to dosisID.
        
    Returns
    -------
    if git_hub_info: True
        vac : ndarray-->shape (D,N,T)
            3D array with  vaccinated by date from dosis i group j at time t in M[i,j,t].
            
        vac_acc : ndarray--> shape (D,N,T)
            3D array with accumulated vaccinated from dosis i group j at time t in M[i,j,t].
    else: 
        
        vac_acc : ndarray-->shape (L,D,N,T)
            4D array with  vaccinated by date from lab l dosis i group j at time t in M[l,i,j,t].
            
        vac_acc : ndarray--> shape (L,D,N,T)
            4D array with accumulated vaccinated from lab l dosis i group j at time t in M[l,i,j,t].
    """
    df_vac_copy = df_77.copy()
    if git_hub_info:
        D = len(dosis_to_dosisID)
        N = len(group_to_groupID)
        T = len(date_to_dateID)
        vac = np.zeros((D, N, T))
        vac_acc= np.zeros((D, N, T))
        #edge_list = zip(df_vac_copy["Dosis"],df_vac_copy["Grupo de edad"],df_vac_copy["start_date"],df_vac_copy["accumulated_vaccinated"])
        edge_list = zip(df_vac_copy["Dosis"],df_vac_copy["Grupo de edad"],df_vac_copy["start_date"],df_vac_copy["vaccinated_today"],df_vac_copy["accumulated_vaccinated"])
        for edge_entry in edge_list:
            dosis, grupo_edad, date_t, vac_ijt,vac_acc_ijk = edge_entry
            i = dosis_to_dosisID[dosis]
            j = group_to_groupID[grupo_edad]
            t = date_to_dateID[date_t.to_datetime64()]
            vac[i,j,t] += vac_ijt
            vac_acc[i,j,t] += vac_acc_ijk
    else:
        L = len(lab_to_labID)
        D = len(dosis_to_dosisID)
        N = len(group_to_groupID)
        T = len(date_to_dateID)
        vac = np.zeros((L,D, N, T))
        vac_acc= np.zeros((L,D, N, T))
        #edge_list = zip(df_vac_copy["Dosis"],df_vac_copy["Grupo de edad"],df_vac_copy["start_date"],df_vac_copy["accumulated_vaccinated"])
        edge_list = zip(df_vac_copy["Laboratorio"],df_vac_copy["Dosis"],df_vac_copy["Grupo de edad"],df_vac_copy["start_date"],df_vac_copy["vaccinated_today"],df_vac_copy["accumulated_vaccinated"])
        for edge_entry in edge_list:
            lab,dosis, grupo_edad, date_t, vac_ijt,vac_acc_ijk = edge_entry
            l = lab_to_labID[lab]
            i = dosis_to_dosisID[dosis]
            j = group_to_groupID[grupo_edad]
            t = date_to_dateID[date_t.to_datetime64()]
            vac[l,i,j,t] += vac_ijt
            vac_acc[l,i,j,t] += vac_acc_ijk
        
    
    return vac, vac_acc


def get_not_vac(df_not_vacc,pop, date_to_dateID, group_to_groupID):
    """
    Create not vaccinated tensor: M[j,t] 
    
    Parameters
    ----------
    df_not_vacc : pandas.DataFrame
        Dataframe of get the not vaccinated popultion at age group.
    pop: numpy.array-->(G,)
		Array containing population features.
        
    date_to_dateID : dict
        Dictionary mapping date to dateID.	
    group_to_groupID : dict
        Dictionary mapping group to groupID.

     
    Returns
    -------
    vac_acc : ndarray--> shape (G,T)
        42D array with no vaccinated from group j at time t in M[j,t].
    """
    
    df_not_vacc_copy = df_not_vacc.copy()
    
    N = len(group_to_groupID)
    T = len(date_to_dateID)
    
    not_vacc = np.ones((N, T))
    not_vacc = np.multiply(not_vacc,np.expand_dims(pop,axis=(-1)))
    edge_list = zip(df_not_vacc_copy["Grupo de edad"],df_not_vacc_copy["start_date"],df_not_vacc_copy["not_vaccinated"])
    for edge_entry in edge_list:
        grupo_edad, date_t, not_vac_ijt = edge_entry
        j = group_to_groupID[grupo_edad]
        t = date_to_dateID[date_t.to_datetime64()]
        not_vacc[j,t] = not_vac_ijt
    
    
    not_vacc=np.where(not_vacc>=0, not_vacc, 0)
    return not_vacc


def get_dead_icu(df_57):
    df_57_copy = df_57.copy()
    df_57_copy = df_57_copy.sort_values(by=["start_date"], ascending=True)
    
        
    df_57_copy=df_57_copy[['dead_hospitalizados']].unstack()
    dead_icu = df_57_copy.to_numpy()
    
    return dead_icu
    

def prepare_producto_10():
    """
    call producto_10
    transform the data
    
    Doble check: 
        1. check the columns it is always incremental
        2. sometimes the data is not continuous ( daily ) we want to make sure that the deaths are daily
        3. Apply hampel function 
        4. Rolling time series for 7 perios, centered on value , get the mean 

    Returns
    -------
    df : Dataframe
        columns: 'Grupo de edad', 'start_date', 'accumulated_dead', 'dead_today',
               'mean_dead_hampel'].

    """
    start_time = time.time()
    df = producto_10()
    
    df= df[['Grupo de edad','start_date','accumulated_dead']]
    
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/github_product/'
    path=path_projet+path_data
    dict_between_df_9_10 = pd.read_csv(path+'dict_between_df_9_10.csv')
    df=pd.merge(df,dict_between_df_9_10, how='left', left_on=['Grupo de edad'], right_on=['1'])
    df.drop(columns=['1','Grupo de edad'], inplace= True)
    df = df.groupby(['0', 'start_date']).agg('sum').reset_index()
    df.rename(columns={'0': 'Grupo de edad'}, inplace=True)


    """
    Doble check: 
        1. check the columns it is always incremental
        2. sometimes the data is not continuous ( daily ) we want to make sure that the deaths are daily
        3. Apply hampel function 
        4. Rolling time series for 7 perios, centered on value , get the mean 
    """
    df['accumulated_dead']=df.groupby(
            ["Grupo de edad"])['accumulated_dead'].cummax()
    df['dead_today'] = df.groupby(["Grupo de edad"])['accumulated_dead'].diff()/(df.groupby(["Grupo de edad"])['start_date'].diff()/ np.timedelta64(1, 'D'))
    df['dead_today_v0']=df['dead_today']
    df['mean_dead_hampel']=df.groupby(["Grupo de edad"])['dead_today'].apply(hampel)
    f1 = lambda x:  x.rolling(7,center=True, min_periods=7).mean()
    df['dead_today']=df.groupby(["Grupo de edad"])['mean_dead_hampel'].apply(f1)
    
    
    end_time = round(time.time()-start_time,2)
    print("Execution time:" + str(end_time) +" s.")
    print("Porducto 10 ready")
    print("="*40)
    return df
    



def hampel(x, k=7, nsigma=4):
    """
    

    Parameters
    ----------
    x : dataframe
        pandas series of values from which to remove outliers.
    k : int, optional
        size of window (including the sample; 7 is equal to 3 on either side of value). The default is 7.
    nsigma : int, optional
        specifies a number of standard deviations  by which a sample of x. The default is 4.

    Returns
    -------
    pandas series of values.

    """
    #Make copy so original not edited
    vals = x.copy()

    #Hampel Filter
    L = 1.4826
    rolling_median = vals.rolling(window=k, center=True).mean()#meadian()
    MAD = lambda x: np.median(np.abs(x - np.median(x)))
    rolling_MAD = vals.rolling(window=k, center=True).apply(MAD)
    threshold = nsigma * L * rolling_MAD
    difference = np.abs(vals - rolling_median)

    '''
    Perhaps a condition should be added here in the case that the threshold value
    is 0.0; maybe do not mark as outlier. MAD may be 0.0 without the original values
    being equal. See differences between MAD vs SDV.
    '''

    outlier_idx = difference > threshold
    vals[outlier_idx] = rolling_median[outlier_idx] 
    return(vals)


def prepare_producto_16(max_date_check='2021-07-01'):
    """
    call producto_16
    transform the data
    join with call producto 26_27 get the sintomatic data

    Returns
    -------
    df : Dataframe
        columns: ['Grupo de edad','start_date','accumulated_infected',
               'sintomatic_today','asintomatic_today','infected_today_all',
               'infected_today','infected_sintomatic_today'].

    """
    start_time = time.time()
    df = producto_16()
    
    #firts drop
    df = df[df.start_date!=np.datetime64('2020-10-05')]
    df = df[df.start_date<max_date_check]
    #df = df[df.start_date!=np.datetime64('2020-06-19')]
    #df = df[df.start_date!=np.datetime64('2020-06-23')]
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/github_product/'
    path=path_projet+path_data
    
    dict_between_df_9_16 = pd.read_csv(path+'dict_between_df_9_16_over_15_years.csv')
    #product 16: delete gender column and group
    df.drop(columns=['Sexo'], inplace= True)
    df.dropna(inplace=True)
    df = df.groupby(['Grupo de edad', 'start_date']).agg('sum').reset_index()
    df=pd.merge(df,dict_between_df_9_16, how='left', left_on=['Grupo de edad'], right_on=['1'])
    df.drop(columns=['1','Grupo de edad'], inplace= True)
    df = df.groupby(['0', 'start_date']).agg('sum').reset_index()
    df.rename(columns={'0': 'Grupo de edad'}, inplace=True)
    
    df_date, df_date_columns = get_df_date_16(df, missing_values=True)
    df= pd.concat([ df, df_date], ignore_index=True)
    df = df.sort_values(by=df_date_columns).reset_index(drop=True)
    df['accumulated_infected']=df.groupby(
        df_date_columns[:-1])['accumulated_infected'].apply(lambda group: group.interpolate(limit=30)).apply(np.floor)
    
    
    df['infected_today'] = df.groupby(df_date_columns[:-1])['accumulated_infected'].diff()
    aux = df[['start_date', 'infected_today']].copy()
    aux = aux.groupby(['start_date']).agg('sum').reset_index().rename(columns={'infected_today': 'infected_today_all'})
    df = pd.merge(aux,df, how='left', on=['start_date'])
    
    
    #call producto 26 for sintomatic,continuo 
    df_26 = producto_26()
    df_26 = df_26[df_26.Region=='Total'][['start_date', 'sintomatic_today', 'asintomatic_today']].copy()
    
    df = pd.merge(df,df_26,how='inner',on=['start_date'] )
    #df['infected_sintomatic_today'] = df['sintomatic_today']*(df['infected_today']/df['infected_today_all'])
    df['infected_sintomatic_today'] = df['infected_today']*(
        df['sintomatic_today']/(df['sintomatic_today']+df['asintomatic_today']))
    
    df =df[['Grupo de edad','start_date','accumulated_infected',
           'sintomatic_today','asintomatic_today','infected_today_all',
           'infected_today','infected_sintomatic_today']]
    
    
    #df['estimado_mayor_real_v1']=np.where(df['infected_sintomatic_today']>df['infected_today'],1,0)
    #df['estimado_mayor_real_v2']=np.where(df['infected_sintomatic_today_2']>df['infected_today'],1,0)
    end_time = round(time.time()-start_time,2)
    print("Execution time:" + str(end_time) +" s.")
    print("Porducto 16 ready")
    print("="*40)
    return df


def get_df_date_16(df, missing_values=True):
    """
    Main idea: 
        generate all the combinations for the set age and dates

    Parameters
    ----------
    df : Dataframe
        df from producto 16.
    missing_values : bolean, optional
        if we are interested in all possible dates True, otherwise False ,
        combinations are generated from start_date and end_date dates. 
        The default is True.
    
    Returns
    -------
    df_date : Dataframe
        DESCRIPTION.
    df_date_columns : list of string
        ['Grupo de edad', 'start_date'].
    
    """
    if missing_values:
        date_arr=np.setdiff1d(
            pd.date_range(
                df.start_date.unique()[0],
                df.start_date.unique()[-1],
                freq='1d'),
            df.start_date.unique())
    
    df_date=pd.DataFrame(
            list(itertools.product(
                    df['Grupo de edad'].unique(),
                    date_arr)))
    df_date_columns = ['Grupo de edad', 'start_date']
    df_date.columns=df_date_columns
    df_date['accumulated_infected']= np.nan
    
    return df_date, df_date_columns


def prepare_producto_57():
    df=prodcuto_57()
    df['total_deads_p_57']=df['total_deads_p_57'].rolling(21,center=True, min_periods=7).mean()
    df['dead_hospitalizados']=df['hospitalizados'].rolling(21,center=True, min_periods=7).mean()

    return df

def prepare_not_vac(df_77, df_pop ):
    """
    get the not vaccinated population

    Parameters
    ----------
    df_77 : Dataframe
        columns: ['Grupo de edad', 'start_date', 'Laboratorio', 'Dosis',
               'vaccinated_today', 'accumulated_vaccinated', 'Region'].
    df_pop : Dataframe
        columns: ['Grupo de edad', 'Personas'].

    Returns
    -------
    df_not_vacc: Dataframe
        columns: ['Grupo de edad', 'start_date','not_vaccinated'].

    """
    df_not_vacc = df_77.copy()
    df_not_vacc = df_not_vacc [(df_not_vacc.Region=='Total')&(df_not_vacc['Dosis']=='Primera')]
    df_not_vacc = df_not_vacc[['Grupo de edad', 'start_date',
      'accumulated_vaccinated']]
    df_not_vacc =df_not_vacc.groupby(['Grupo de edad', 'start_date'])['accumulated_vaccinated'].sum().reset_index()
    df_not_vacc = pd.merge(df_not_vacc, df_pop, how='left', on=['Grupo de edad'])
    df_not_vacc['not_vaccinated']= df_not_vacc['Personas']-df_not_vacc['accumulated_vaccinated']
    
    return df_not_vacc[['Grupo de edad', 'start_date','not_vaccinated']]
    

def prepare_population():
    """
    get the population by age groups

    Returns
    -------
    df : Dataframe
        columns: ['Grupo de edad', 'Personas'].

    """
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/poblacion/'
    path=path_projet+path_data
    df = pd.read_csv(path+'estimacion_poblacion_ine_2020.csv',sep=';')
    
    df['Grupo de edad']=None #np.nan
    df['Grupo de edad']=np.where((df['edad']>14)&(df['edad']<=39),'<=39',df['Grupo de edad'] )
    df['Grupo de edad']=np.where((df['edad']>39)&(df['edad']<=49),'40-49',df['Grupo de edad'] )
    df['Grupo de edad']=np.where((df['edad']>49)&(df['edad']<=59),'50-59',df['Grupo de edad'] )
    df['Grupo de edad']=np.where((df['edad']>59)&(df['edad']<=69), '60-69',df['Grupo de edad'] )
    df['Grupo de edad']=np.where(df['edad']>69,'>=70',df['Grupo de edad'] )
    
    df.dropna(inplace=True)
    df.drop(columns=['edad'], inplace= True)
    df = df.groupby(['Grupo de edad']).agg('sum').reset_index()
    
    return df    

    

    
    
    