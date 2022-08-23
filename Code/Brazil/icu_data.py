#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 14 11:39:49 2022

@author: ineira

Registro de Ocupação Hospitalar COVID-19 - 2020

https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/LEITOS/2022-07-12/esus-vepi.LeitoOcupacao_2020.csv
 
--> check data from h
https://transparenciacovid19.ok.org.br/files/ESPECIAL_Transparencia-Covid19_OcupacaoLeitos_01.pdf
https://github.com/okfn-brasil/transparencia-leitos-covid-analise/blob/main/docs/manual-de-utilizacao-da-api-da-leitos.pdf

cnes: estabelecimentos-->https://raw.githubusercontent.com/okfn-brasil/transparencia-leitos-covid-analise/main/exports/hospitais_cnes.json
zip((icu_2020.columns.tolist(),["" for item in range(len(icu_2020.columns))]))

{'Unnamed: 0': '',
 '_id': '',
 'dataNotificacao': 'dia de la notifcacion',
 'cnes': 'estabelecimentos',
 'ocupacaoSuspeitoCli': '',
 'ocupacaoSuspeitoUti': '',
 'ocupacaoConfirmadoCli': '',
 'ocupacaoConfirmadoUti': '',
 'ocupacaoCovidUti': '',
 'ocupacaoCovidCli': '',
 'ocupacaoHospitalarUti': '',
 'ocupacaoHospitalarCli': '',
 'saidaSuspeitaObitos': '',
 'saidaSuspeitaAltas': '',
 'saidaConfirmadaObitos': '',
 'saidaConfirmadaAltas': '',
 'origem': '',
 '_p_usuario': '',
 'estadoNotificacao': estado,
 'municipioNotificacao': municipio ej Joaçaba,
 'estado': estado,
 'municipio': municipio,
 'excluido': '',
 'validado': '',
 '_created_at': dia que se creo la data de la notificacion,
 '_updated_at': dia que se creo la data que se actualizo}


"""

import requests
import io
import os

import pandas as pd
import  numpy as np

import json
from urllib.request import urlretrieve #open url


from datetime import datetime,timedelta
pd.set_option('display.float_format', lambda x: '%.9f' % x)


from matplotlib import cm
cmap = cm.get_cmap('Spectral')
import matplotlib.pyplot as plt # plotting data 
import matplotlib.dates as mdates
from matplotlib.dates import date2num


def download_uci_data():
    """
    Registro de Ocupação Hospitalar COVID-19 - 2020

    https://s3.sa-east-1.amazonaws.com/ckan.saude.gov.br/LEITOS/2022-07-12/esus-vepi.LeitoOcupacao_2020.csv
    
    Returns
    -------
    frames : TYPE
        DESCRIPTION.

    """


    icu_csv_file_2020="esus-vepi.LeitoOcupacao_2020.csv"
    icu_csv_file_2021="esus-vepi.LeitoOcupacao_2021.csv"
    icu_csv_file_2022="esus-vepi.LeitoOcupacao_2022.csv" #
    
    
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/Brazil/'
    
    
    data_files=[icu_csv_file_2020,icu_csv_file_2021]

    
    frames = []
    for f in data_files[:]:
        print(f)
        df=pd.read_csv(path_projet+path_data+f)
        print(df.dataNotificacao.min(),
              df.dataNotificacao.max()
            )
        
        df_hb = preprocess_data(df)
        frames.append(df_hb)
    
    
    
    return frames
    
def preprocess_data(df, update_cnes_data=False):
    """
    1. keep columns
    2. clean up acronyms, many are in lower case, preserving data

    Parameters
    ----------
    df : TYPE
        DESCRIPTION.
    updated_days : TYPE
        DESCRIPTION.
    TS_RUN : TYPE
        DESCRIPTION.
    get_cnes_data : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    None.

    """

    
    keep_columns=['Unnamed: 0',
     '_id',
     'dataNotificacao',
     'cnes',
     'ocupacaoSuspeitoCli',
     'ocupacaoSuspeitoUti',
     'ocupacaoConfirmadoCli',
     'ocupacaoConfirmadoUti',
     'ocupacaoCovidUti',
     'ocupacaoCovidCli',
     'ocupacaoHospitalarUti',
     'ocupacaoHospitalarCli',
     'saidaSuspeitaObitos',
     'saidaSuspeitaAltas',
     'saidaConfirmadaObitos',
     'saidaConfirmadaAltas',
     'origem',
     '_p_usuario',
     #'estadoNotificacao',
     #'municipioNotificacao',
     'estado',
     'municipio',
     'excluido',
     'validado',
     '_created_at',
     '_updated_at']
    
    ## limpiar los acrónimos, muchos están en minúsculas, preservando los datos
    df=df[keep_columns].copy()
    df['estado_original'] = df['estado']
    df['estado'] = df['estado'].str.upper()
    df['start_date']=pd.to_datetime(df.dataNotificacao, format="%Y-%m-%d").dt.strftime('%Y-%m-%d')
    df['_updated_at']=pd.to_datetime(df._updated_at)
    # fill nans with zeros
    df = df.fillna({
            'ocupacaoSuspeitoCli':0, # supuesta
            'ocupacaoSuspeitoUti': 0, # supuesta
           'ocupacaoConfirmadoCli': 0, # confirmada
           'ocupacaoConfirmadoUti': 0, # confirmada
           'ocupacaoCovidUti': 0,  # coupacion covid--->este es el producto
           'ocupacaoCovidCli': 0,
           'ocupacaoHospitalarUti': 0,
           'ocupacaoHospitalarCli': 0,
           'saidaSuspeitaObitos': 0, #salidas supuestas por muerte
           'saidaSuspeitaAltas': 0, #salidas supuestas por altas
           'saidaConfirmadaObitos': 0, # salidas confirmadas por puerte
           'saidaConfirmadaAltas': 0, # salidas confirmadas por alta
       })
    # considering offer without occupancy, can not check offer
    
    df['totalOcupCli'] = df['ocupacaoHospitalarCli'] + df['ocupacaoCovidCli'] #check ocupacaoConfirmadoCli
    df['totalOcupUti'] = df['ocupacaoHospitalarUti'] + df['ocupacaoCovidUti']
    
    df['has_uti_proxy'] = np.where(df['totalOcupUti'] > 0, True, False)
    
    df['cnes'] = df.cnes.astype('category')
    print(f"Duplicated values by start_date and cnes: { df.duplicated(subset=['start_date','cnes']).sum()}")
    print(f"Aprox {round(df.duplicated(subset=['start_date','cnes']).sum()/len(df)*100,5)}% of the data")
    df.sort_values(by=['cnes','start_date','_updated_at'], 
                   ascending=[True, False,True]).drop_duplicates(subset=['start_date','cnes'], keep='first')
    
    #beds cnes (leitos= beds)
    
    df_hosp =  get_all_hospital_data(update_data=update_cnes_data)
    # merge dados api e dados cnes
    df_h = df.merge(df_hosp, on='cnes', suffixes=('', '_cnes'), how='left')
   
    # leitos cnes
    all_beds = []
    for index, row in df_h[df_h['beds'].notna()].iterrows():
        beds = row['beds']
        for bed in beds:
            bed['cnes'] = row['cnes']
            all_beds.append(bed)
    
    df_beds = pd.DataFrame(all_beds)
    df_beds['total_beds'] = df_beds['qtExistente'].astype(float)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df_beds[df_beds.dsLeito.str[0:3] == 'UTI'].groupby(['dsLeito', 'dsAtributo']).total_beds.sum().sort_values(ascending=False))
     # leitos tipo UTI
    df_beds['uti'] = (np.where(
        df_beds.dsLeito.str[0:3] == 'UTI',
            np.where(df_beds.dsLeito.str.contains('NEONATAL|QUEIMADOS') == False, True, False)
        , False))
    # sum beds cnes uti
    cnes_uti = df_beds[df_beds['uti'] == True].groupby('cnes').agg({'total_beds': 'sum'})
    cnes_uti.reset_index(inplace=True)
    cnes_uti.rename(columns={'total_beds': 'uti_beds_via_cnes'}, inplace=True)
    
    
    df_hb = df_h.merge(cnes_uti, on='cnes', how='left')
    
    # beds occupation errors?
    
    
    return df_hb




def get_all_hospital_data(update_data=False):
    cnes_csv_input_file= "https://raw.githubusercontent.com/okfn-brasil/transparencia-leitos-covid-analise/main/exports/hospitais_cnes.json"
    cnes_csv_file="hospitais_cnes.json"
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/Brazil/'
    
    cnes_csv_save_file = path_projet+path_data+cnes_csv_file
    os.makedirs(os.path.dirname(cnes_csv_save_file), exist_ok=True)
    
    last_update_file=path_projet+path_data+"last_update_file_cnes_data.json"
    os.makedirs(os.path.dirname(last_update_file), exist_ok=True)
    if update_data:
    
        urlretrieve(cnes_csv_input_file, cnes_csv_save_file)
        today= datetime.now().strftime('%Y-%m-%d')
        print(f"Update the file: {today}")
        
        with open(last_update_file, "w") as out:
            json.dump({'last_update':today},out)
    else:
        
        with open(last_update_file) as f:
            last_update=json.load(f)
        print(f"The last Update was: {last_update['last_update']}")
        
        # ler resultado
    dtypes = {
        'cnes': str  # has left padding zeros
    }
    df_hosp = pd.read_json(cnes_csv_save_file, dtype=dtypes)
    return df_hosp



"""

frames=download_uci_data()
df_hb=frames[1]
df_tmp = df_hb[(df_hb.has_uti_proxy == True)]
df_tmp = df_hb.copy()

agg = {
    '_id': 'count',
    'uti_beds_via_cnes': 'sum',
    'ocupacaoSuspeitoUti': 'sum',
    'ocupacaoConfirmadoUti': 'sum',
    'ocupacaoCovidUti': 'sum',
}

by_state = df_tmp.groupby(['estado','start_date']).agg(agg)
country = df_tmp.groupby('start_date').agg(agg)
country['estadoSigla'] = 'Brasil'
"""
