# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 12:38:29 2022

@author: ignasi

Data vaccination 

5 people take part in a vaccination program, to be given a vaccine that requires 2 doses to be effective against the disease.

Rebecca has received 2 doses, then a 3rd (booster) dose;
Thomas has received 2 doses;
James has received 1 dose;
Lauren has not received any dose.
In our data:

The total number of doses administered (total_vaccinations) will be equal to 6 (3 + 2 + 1);
The total number of people vaccinated (people_vaccinated) will be equal to 3 (Rebecca, James, Thomas);
The total number of people with a complete initial protocol (people_fully_vaccinated) will be equal to 2 (Rebecca, Thomas);
The total number of boosters administered (total_boosters) will be equal to 1 (Rebecca).

"""
import sys
import os
import warnings
import numpy as np
import pandas as pd
import itertools
import time
# OWN MODULES


path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
path_world='Code/World'

module_path=path_projet+path_world
if module_path not in sys.path:
    sys.path.append(module_path)
    
from owid_data_by_country import call_data_by_country_owid_data
from covariants_data import covariants_severidad




import matplotlib.pyplot as plt 
from matplotlib.dates import date2num
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

def read_data(country="Argentina",
              columns= [ "location", 
                        "date", 
                        'total_cases',
                        "new_cases",
                        "new_cases_per_million",
                        'total_deaths', 
                        'new_deaths',
                        'icu_patients',
                        'icu_patients_per_million',
                        'total_vaccinations',
                        'people_vaccinated', 
                        'people_fully_vaccinated',
                        'total_boosters',
                        'new_vaccinations'],
              update_data=False,
              vaccine='general'):
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
    
    
    df=call_data_by_country_owid_data()
    
    print("Change columns name , add a Grupo de edad as a columns")
    df["Grupo de edad"]="Total"
    df['Laboratorio']='general'
    df.rename(columns={'date': 'start_date'},inplace=True)
    
    
    df_9 = prepare_producto_9(df)
    df_10 = prepare_producto_10(df)
    df_16 = prepare_producto_16(df)
    df_pop = prepare_population()
    if vaccine=='general':
        df_77=prepare_vac(df, df_pop)
        lab = np.sort(df_77["Laboratorio"].unique())
        lab_to_labID = {lab:labID for labID, lab in enumerate(lab)}

    else:
        pass
        #df_77=Avance_vacunacion()
        # Get {dosis:dosisID} dictionary from vaccinated
        #lab = np.sort(df_77["Laboratorio"].unique())
        #lab_to_labID = {lab:labID for labID, lab in enumerate(lab)}
    
    #save date start second dosis
    date_firt_dosis = df_77[(df_77.Dosis=="Primera")&(df_77.accumulated_vaccinated>0)]['start_date'].unique()[0]
    date_second_dosis = df_77[(df_77.Dosis=="Segunda")&(df_77.accumulated_vaccinated>0)]['start_date'].unique()[0]
    
    
    # Get {date:dateID} dictionary from infections data
    dates = np.sort(np.intersect1d(df_16["start_date"].unique(),df_9["start_date"].unique()))
    #dates = np.sort(df_16["start_date"].unique())
    date_to_dateID = {date:dateID for dateID, date in enumerate(dates)}
    
    dateID_firt_dosis = date_to_dateID[date_firt_dosis]
    dateID_second_dosis = date_to_dateID[date_second_dosis]
    
    # Get {group:groupID} dictionary from uci beds
    groups = np.sort(df_9["Grupo de edad"].unique())
    group_to_groupID = {group:groupID for groupID, group in enumerate(groups)}
    
    # Get {dosis:dosisID} dictionary from vaccinated
    dosis = np.sort(["Primera", 'Segunda'])
    dosis_to_dosisID = {dosis:dosisID for dosisID, dosis in enumerate(dosis)}

    # Filter for our data range 
    df_9 = df_9[df_9["start_date"].isin(date_to_dateID)]
    df_10= df_10[df_10["start_date"].isin(date_to_dateID)]
    df_16 = df_16[df_16["start_date"].isin(date_to_dateID)]
    df_77 = df_77[df_77["start_date"].isin(date_to_dateID)]
    
    #prepare new dataframe
    df_not_vacc = prepare_not_vac(df_77, df_pop)
    df_not_vacc =  df_not_vacc[df_not_vacc["start_date"].isin(date_to_dateID)]
    
    
    # Get data matrices
    pop = get_pop(df_pop)
    uci = get_uci(df_9)
    dead = get_dead(df_10)
    inf = get_inf(df_16,sintomatic_value=False)
    vac,vac_acc = get_vac(df_77, date_to_dateID, group_to_groupID, dosis_to_dosisID,lab_to_labID,lab=False)
    not_vac = get_not_vac(df_not_vacc,pop, date_to_dateID, group_to_groupID)
    
    
    # Useful dictionaries for external analysis
    dateID_to_date = {dateID:date for dateID, date in enumerate(dates)}
    groupID_to_group = {groupID:group for groupID, group in enumerate(groups)}
    
    data = {'vac':vac, 'inf':inf, 'uci':uci,'dead':dead, 'pop':pop, 
            'not_vac': not_vac,'vac_acc': vac_acc,
            'dateID_firt_dosis' : dateID_firt_dosis,
            'dateID_second_dosis' : dateID_second_dosis,
            'groupID_to_group':groupID_to_group, 
            'dateID_to_date':dateID_to_date, 
            'date_to_dateID':date_to_dateID,
            'dosis_to_dosisID':dosis_to_dosisID,
            'lab_to_labID':lab_to_labID}
    
    
    
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
  "https://github.com/owid/covid-19-data/blob/master/public/data/archived/ecdc/locations.csv"
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
    
    #inf[:,48:52]=np.expand_dims((inf[:,47]+inf[:,52])/2,axis=(-1))
    
    
    return inf

def get_vac(df_77, date_to_dateID, group_to_groupID, dosis_to_dosisID,lab_to_labID, lab=False):
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
    #df_vac_copy.fillna(0,inplace=True)
    if lab==False:
        L = len(lab_to_labID)
        D = len(dosis_to_dosisID)
        N = len(group_to_groupID)
        T = len(date_to_dateID)
        vac = np.zeros((L,D, N, T))
        vac_acc= np.zeros((L,D, N, T))
        
        edge_list = zip(df_vac_copy["Laboratorio"],df_vac_copy["Dosis"],df_vac_copy["Grupo de edad"],df_vac_copy["start_date"],df_vac_copy["vaccinated_today"],df_vac_copy["accumulated_vaccinated"])
        for edge_entry in edge_list:
            lab,dosis, grupo_edad, date_t, vac_ijt,vac_acc_ijk = edge_entry
            l = lab_to_labID[lab]
            i = dosis_to_dosisID[dosis]
            j = group_to_groupID[grupo_edad]
            t = date_to_dateID[date_t.to_datetime64()]
            vac[l,i,j,t] += vac_ijt
            vac_acc[l,i,j,t] += vac_acc_ijk
        
            
    else:
        pass
        """
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
        """
    
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
    df_not_vacc_copy.dropna(inplace=True)
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
    
    
def prepare_producto_9(df):
    """

    transform the data
    for some dates, data is incomplete, apply a linear interpolation 

    Parameters
    ----------
    df : Dataframe
        DESCRIPTION.
    
    Returns
    -------
    df : Dataframe
        columns: 'Grupo de edad', 'start_date', 'icu_patients'].


    """
    start_time = time.time()
    df=df[["start_date","Grupo de edad","icu_patients"]].copy()
    df.rename(columns={'icu_patients': 'uci_beds'},inplace=True)
    df['uci_beds']=df.groupby(
       ['Grupo de edad'])['uci_beds'].apply(lambda group: group.interpolate(limit=10)).apply(np.floor).fillna(method="bfill")
    
    end_time = round(time.time()-start_time,2)
    print("Execution time:" + str(end_time) +" s.")
    print("Porducto 9 ready: ICU patients")
    print("="*40)
    return  df.dropna()

    
    
    
def prepare_producto_10(df):
    """
    prepare dead time series
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

    
    df= df[['Grupo de edad','start_date','new_deaths','total_deaths']].copy()
    df.rename(columns={'new_deaths':'dead_today', 
                       'total_deaths':'accumulated_dead'},
              inplace=True)

    """
    Doble check: 
        1. check the columns it is always incremental
        2. sometimes the data is not continuous ( daily ) we want to make sure that the deaths are daily
        3. Apply hampel function 
        4. Rolling time series for 7 perios, centered on value , get the mean 
    """
    df['accumulated_dead']=df.groupby(
            ["Grupo de edad"])['accumulated_dead'].cummax()

    df['mean_dead_hampel']=df.groupby(["Grupo de edad"])['dead_today'].apply(hampel)
    f1 = lambda x:  x.rolling(7,center=True, min_periods=7).mean()
    df['mean_dead_hampel']=df.groupby(["Grupo de edad"])['mean_dead_hampel'].apply(f1)
    
    
    end_time = round(time.time()-start_time,2)
    print("Execution time:" + str(end_time) +" s.")
    print("Porducto 10 ready: Dead people ")
    print("="*40)
    return df.dropna()
    



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


def prepare_producto_16(df):
    """
    transform the data
    for some dates, data is incomplete, apply a linear interpolation 

    Returns
    -------
    df : Dataframe
        columns: ['Grupo de edad','start_date','accumulated_infected',
               'infected_today'].

    """
    start_time = time.time()
    df = df[['Grupo de edad','start_date','total_cases','new_cases','total_cases']].copy()
    df.rename(columns={'total_cases':'accumulated_infected',
                       'new_cases':'infected_today'}, inplace= True)
    
    df['infected_today']=df.groupby(
       ['Grupo de edad'])['infected_today'].apply(lambda group: group.interpolate(limit=10)).apply(np.floor)
    
    
    df =df[['Grupo de edad','start_date','accumulated_infected',
           'infected_today']]
    
    end_time = round(time.time()-start_time,2)
    print("Execution time:" + str(end_time) +" s.")
    print("Porducto 16 ready")
    print("="*40)
    return df.dropna()



def prepare_vac(df, df_pop):
    """
    get the  vaccinated and not vaccinated population


    Parameters
    ----------
    df : Dataframe
        DESCRIPTION.
    df_pop :  Dataframe
        columns: ['Grupo de edad', 'Personas'].

    Returns
    -------
    df_not_vacc: Dataframe
        columns: ['Grupo de edad', 'start_date','not_vaccinated'].
    df_vacc : Dataframe
        columns: ['Grupo de edad','start_date','new_vaccinations','newVaccinated_second']

    """
    
    #Check if list of dates is complete
    print("Check if list of dates is complete!") 
    print(df.start_date.diff().value_counts())
    df_vacc=df[['Grupo de edad', 'Laboratorio','start_date','people_vaccinated']].copy()
    #df_vacc.rename(columns={'people_vaccinated':'vaccinated_today'},inplace=True)
    #df_vacc.drop(inplace=True)
    # create new columns
    df_vacc['1° Dosis'] = df_vacc.groupby(['Grupo de edad', 'Laboratorio'])['people_vaccinated'].diff().fillna(df['people_vaccinated'])
    df_vacc['2° Dosis'] =  df_vacc.groupby(['Grupo de edad', 'Laboratorio'])['1° Dosis'].shift(periods=28, fill_value=0)

    index_frist_dosis=df_vacc.loc[~df_vacc['1° Dosis'].isnull()].index[0]
    index_second_dosis=df_vacc.loc[~df_vacc['2° Dosis'].isnull()].index[0]
    print("Frist dosis was: {}".format(df.iloc[[index_frist_dosis]].start_date))
    print("Second dosis was: {}".format(df.iloc[[index_second_dosis]].start_date))
    df_vacc.fillna(0,inplace=True)
    df_vacc.drop(columns=['people_vaccinated'],inplace=True)
    df_vacc.columns=['Grupo de edad', 'Laboratorio', 'start_date', 'Primera', 'Segunda']
    aux_df1=pd.melt(df_vacc, id_vars=['Grupo de edad', 'start_date','Laboratorio'], value_vars=['Primera', 'Segunda'],var_name='Dosis', value_name='vaccinated_today')
    aux_df1['accumulated_vaccinated']=aux_df1.groupby(['Grupo de edad', 'Laboratorio','Dosis'])['vaccinated_today'].cumsum()
    
    return aux_df1


def prepare_not_vac(df_77, df_pop):
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
    df_not_vacc = df_not_vacc [(df_not_vacc['Dosis']=='Primera')]
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
    
    url_dp1 = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/archived/ecdc/locations.csv"
    dp1 = pd.read_csv(url_dp1, error_bad_lines=False)
    df=dp1[dp1.location=="Argentina"][['population']].copy()
    df.rename(columns={'population':'Personas'},inplace=True)
    df=df.astype({'Personas':'int'})
    df['Grupo de edad']="Total"
    


    
    return df[['Grupo de edad', 'Personas']]

    

    
 
if __name__ == "__main__":   
    read_data(country="Argentina",
                  columns= [ "location", 
                            "date", 
                            'total_cases',
                            "new_cases",
                            "new_cases_per_million",
                            'total_deaths', 
                            'new_deaths',
                            'icu_patients',
                            'icu_patients_per_million',
                            'total_vaccinations',
                            'people_vaccinated', 
                            'people_fully_vaccinated',
                            'total_boosters',
                            'new_vaccinations'],
                  update_data=False,
                  vaccine='general')