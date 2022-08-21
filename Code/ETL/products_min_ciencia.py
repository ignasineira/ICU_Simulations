# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 20:05:31 2021

@author: ignas

Main idea: Download data directly from the github of the Ministry of Science.
"""
import os
import warnings
warnings.filterwarnings("ignore")
import urllib
import  json

import numpy as np
import pandas as pd


import seaborn as sns 
from datetime import  datetime
sns.set(color_codes=True)
import itertools
import scipy.stats as st
import time

def producto_5():
    """
    Descargar data directamente del github del ministerio de MinCiencia
    Data Product 5 - Totales Nacionales Diarios: Set de archivos con cifras a nivel nacional.
    El primero de ellos (TotalesNacionales.csv) incluye los casos nuevos confirmados, 
    totales o acumulados, recuperados, fallecidos a nivel nacional y activos según fecha de 
    diagnóstico y fecha de inicio de síntomas, reportados diariamente por el Ministerio de
    Salud desde el 03-03-2020.

    Returns
    -------
    df : DataFrame
        prodcuto 5: 
            ['start_date', 'Casos nuevos con sintomas', 'Casos totales',
             'Casos nuevos sin sintomas','Casos nuevos totales']

    """
    url_dp1 = "https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto5/TotalesNacionales_T.csv"
    dp1 = pd.read_csv(url_dp1, error_bad_lines=False)
    dp1.rename({'Fecha': 'start_date'}, axis='columns',inplace=True)
 
    dp1['start_date']=dp1.start_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    dp1['start_date']=pd.to_datetime(dp1['start_date'])
    
    
    df=dp1[['start_date', 'Casos nuevos con sintomas', 'Casos totales','Casos nuevos sin sintomas','Casos nuevos totales']].copy()
    df['Casos nuevos sin sintomas'] = df['Casos nuevos sin sintomas'].fillna(0)
    
    #check values
    df['Casos nuevos totales check']=df['Casos nuevos sin sintomas']+df['Casos nuevos con sintomas']
    
    df['Casos nuevos totales'].equals(df['Casos nuevos totales check'])
    df['diff Casos nuevos totales']=df['Casos nuevos totales']-df['Casos nuevos totales check']
    df['proc diff Casos nuevos totales']=df['diff Casos nuevos totales']/df['Casos nuevos totales']
    return df


def producto_7_8():
    """
    Descargar producto 7 y 8  directamente del github del ministerio de MinCiencia
    
    Data Product 7 - Exámenes PCR por región: 
    Set de archivos que dan cuenta del número de exámenes PCR realizados por región 
    reportados diariamente por el Ministerio de Salud, desde el 09-04-2020. 
    El archivo PCR.csv contiene las columnas ‘Región’, ‘Código Región’ y ‘Población’, 
    seguidas por columnas correspondientes a ‘[Fecha]’. Estas últimas columnas, 
    ‘[Fecha]’ indican el número de exámenes realizados por región.
    
    Data Product 8 - Pacientes COVID-19 en UCI por región: 
    Set de 2 archivos que dan cuenta del número de pacientes en UCI, y que son casos
    confirmados por COVID-19, por región reportados diariamente por el Ministerio de Salud,
    desde el 01-04-2020. El archivo UCI.csv contiene las columnas
    ‘Región’, ‘Código Región’ y ‘Población’, y múltiples columnas correspondientes a ‘[Fecha]’.
    Estas últimas columnas, ‘[Fecha]’ indican el número de pacientes en UCI por región,
    desde el 01-04-2020 hasta la fecha. Incluye versión con serie de tiempo. Ver más.
    Returns
    -------
    df_7 : DataFrame
        producto 7: ['Region', 'start_date', 'infected_today']
        
    df_8 : DataFrame
        producto 8: ['Region', 'start_date', 'UCI']

    """
    url_dp1 = "https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto7/PCR.csv"
    dp1 = pd.read_csv(url_dp1, error_bad_lines=False)
    df = []
    name_columns = list(dp1.columns)
    row, col = dp1.shape
    for i in range(row):  # numero de filas
   		for j in range(3, col):  # numero columna numero
   			aux = []
   			aux.append(dp1.iloc[i, 1])  # guardo numero de region
   			aux.append(str(name_columns[j]))  # guardo el dia
   			aux.append(dp1.iloc[i, j])  # numero de casos notificados
   			df.append(aux)
   	# Create the pandas DataFrame
    df_7 = pd.DataFrame(df, columns=[name_columns[0], 'start_date', 'infected_today'])
    url_dp2 = 'https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto8/UCI_std.csv'
    df_8 = pd.read_csv(url_dp2, error_bad_lines=False)
    df_8 = df_8[['Codigo region', 'fecha', 'numero']]
    df_8.columns = ['Region', 'start_date', 'UCI']

    return df_7,df_8


def producto_9(max_date_check='2021-07-01'):
    """
    Descargar producto 9 directamente del github del ministerio de MinCiencia
    
    Data Product 9 - Pacientes COVID-19 en UCI por grupo de edad: 
    Set de 2 archivos que dan cuenta del número de pacientes en UCI por grupos etarios
    (<=39; 40-49; 50-59; 60-69; y >=70) y que son casos confirmados por COVID-19, 
    reportados diariamente por el Ministerio de Salud, desde el 01-04-2020.
    El archivo HospitalizadosUCIEtario.csv contiene la columna ‘Grupo de edad’, 
    seguida por columnas correspondientes a ‘[Fecha]’. Estas últimas columnas, ‘[Fecha]’, 
    indican el número de pacientes en UCI por grupo etario, desde el 01-04-2020 hasta la fecha. 
    Incluye versión con serie de tiempo. 

    Returns
    -------
    df :  DataFrame
        producto 9: ['Grupo de edad', 'start_date', 'uci_beds']

    """
    url_dp1 = "https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto9/HospitalizadosUCIEtario.csv"
    dp1 = pd.read_csv(url_dp1, error_bad_lines=False)
    df = []
    name_columns = list(dp1.columns)
    row, col = dp1.shape
    for i in range(row):  # numero de filas
        for j in range(1, col):  # numero columna numero
            aux = []
            aux.append(dp1.iloc[i, 0])  # guardo grupo de edad
            aux.append(str(name_columns[j]))  # guardo el dia
            aux.append(dp1.iloc[i, j])  # numero de camas uci uso
            df.append(aux)
    
    df = pd.DataFrame(df, columns=[name_columns[0], 'start_date', 'uci_beds'])
 
    df['start_date']=df.start_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df['start_date']=pd.to_datetime(df['start_date'])
    df=df[df.start_date<max_date_check]
    return df

def producto_10():
    """
    Descargar producto 10 directamente del github del ministerio de MinCiencia
    
    Data Product 10 - Fallecidos con COVID-19 por grupo de edad:
    Set de 2 archivos que dan cuenta del número de personas fallecidas con COVID-19
    por grupos etarios (<=39; 40-49; 50-59; 60-69; 70-79; 80-89; y >=90) 
    reportados diariamente por el Ministerio de Salud, desde el 09-04-2020. 
    El archivo FallecidosEtario.csv contiene la columna ‘Grupo de edad’, seguida
    por columnas correspondientes a ‘[Fecha]’. Estas últimas columnas, ‘[Fecha]’,
    indican el número de personas fallecidas por grupo etario, desde el 09-04-2020
    hasta la fecha. Incluye versión con serie de tiempo.
    

    Returns
    -------
    df_10 : DataFrame
        producto 10: ['Grupo de edad', 'start_date', 'accumulated_dead', 'dead_today'].

    """
    url_dp1 = "https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto10/FallecidosEtario.csv"
    dp1 = pd.read_csv(url_dp1, error_bad_lines=False)
    df = []
    name_columns = list(dp1.columns)
    row, col = dp1.shape
    for i in range(row):  # numero de filas
        for j in range(1, col):  # numero columna numero
            aux = []
            aux.append(dp1.iloc[i, 0])  # guardo grupo de edad
            aux.append(str(name_columns[j]))  # guardo el dia
            aux.append(dp1.iloc[i, j])  # numero de casos fallecidos
            df.append(aux)
    
    df_10 = pd.DataFrame(df, columns=[name_columns[0], 'start_date', 'accumulated_dead'])
 
    df_10['start_date']=df_10.start_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df_10['start_date']=pd.to_datetime(df_10['start_date'])
    df_10 = df_10.sort_values(by=[name_columns[0], 'start_date'])
    df_10['dead_today'] = df_10.groupby([name_columns[0]])['accumulated_dead'].diff()/(df_10.groupby([name_columns[0]])['start_date'].diff()/ np.timedelta64(1, 'D'))
    
    return df_10

def producto_16():
    """
    Descargar producto 16 directamente del github del ministerio de MinCiencia
    
    Data Product 16 - Casos por genero y grupo de edad: 
    Archivo que da cuenta del número acumulado de casos confirmados distribuidos por género
    y grupo de edad, para cada fecha reportada. Este concatena la historia de los informes 
    epidemiológicos e informe de situación COVID-19 publicados por el Ministerio de Salud del país.
    Incluye versión con serie de tiempo.

    Returns
    -------
    df_16 : DataFrame
       producto 16: 'Grupo de edad', 'Sexo', 'start_date', 'accumulated_infected'].

    """
    url_dp1 = "https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto16/CasosGeneroEtario.csv"
    dp1 = pd.read_csv(url_dp1, error_bad_lines=False)
    df = []
    name_columns = list(dp1.columns)
    row, col = dp1.shape
    for i in range(row):  # numero de filas
        for j in range(2, col):  # numero columna numero
            aux = []
            aux.append(dp1.iloc[i, 0])  # guardo grupo de edad
            aux.append(dp1.iloc[i, 1]) #guardo sexo
            aux.append(str(name_columns[j]))  # guardo el dia
            aux.append(dp1.iloc[i, j])  # numero de casos notificados
            df.append(aux)
    
    df_16 = pd.DataFrame(df, columns=[name_columns[0],name_columns[1], 'start_date', 'accumulated_infected'])
 
    df_16['start_date']=df_16.start_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df_16['start_date']=pd.to_datetime(df_16['start_date'])
    #df_16 = df_16.sort_values(by=[name_columns[0],name_columns[1], 'start_date'])
    #df_16['infected_today'] = df_16.groupby([name_columns[0],name_columns[1]])['accumulated_infected'].diff()/(df_16.groupby([name_columns[0],name_columns[1]])['start_date'].diff()/ np.timedelta64(1, 'D'))
    return df_16


def producto_20():
    """
    Descargar producto 20 directamente del github del ministerio de MinCiencia
    
    Data Product 20 - Camas Críticas Disponbles a nivel nacional:
    Este producto da cuenta del número total de Camas Críticas en el Sistema Integrado Covid 19, 
    el número de camas disponibles y el número de camas ocupadas, para cada fecha reportada. 
    Se concatena la historia de los reportes diarios publicados por el Ministerio de Salud del país.
    Incluye versión con serie de tiempo.
    
    Agregar columnas que desplazan la serie de tiempo 7 y 14 en el futuro,
    para chequear serie de tiempo. 
    
    Eliminamos las datos que tienen Na, es decir los primeros 14
    
    Returns
    -------
    df_20 : DataFrame
       producto 20: ['total', 'disponibles', 'ocupados', 'start_date', 'total_7', 'total_14',
              'dif_7', 'dif_14']

    """
    url_dp1 ='https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto20/NumeroVentiladores.csv'
    dp1 = pd.read_csv(url_dp1, error_bad_lines=False)
    
    
    df = []
    date_columns = list(dp1.columns)
    name_columns =list(dp1.Ventiladores)
    row, col = dp1.shape
    for j in range(2, col):  # numero columna numero
        aux = []
        aux.append(dp1.iloc[0, j])  # guardo total
        aux.append(dp1.iloc[1, j]) #guardo disponibles
        aux.append(dp1.iloc[2, j])  # numero de casos notificados
        aux.append(str(date_columns[j]))  # guardo el dia
        df.append(aux)
    
    df_20 = pd.DataFrame(df, columns=[name_columns[0],name_columns[1],name_columns[2], 'start_date'])
 
    df_20['start_date']=df_20.start_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df_20['start_date']=pd.to_datetime(df_20['start_date'])
    df_20['total_7']=df_20['total'].shift(7, axis = 0)
    df_20['total_14']=df_20['total'].shift(14, axis = 0)
    df_20['dif_7']=(df_20['ocupados']-df_20['total_7'])
    df_20['dif_14']=(df_20['ocupados']-df_20['total_14'])
    df_20.dropna(inplace=True) 
    
    return df_20


def producto_21():
    """
    Descargar producto 21 directamente del github del ministerio de MinCiencia
    
    Data Product 21 - Sintomas por Casos Confirmados e informado en el último día: 
    Este producto da cuenta de la sintomatología reportada por los casos confirmados. 
    También da cuenta de la sintomatología reportada por casos confirmados que han 
    requerido hospitalización. Se concatena la historia de los informes de Situación 
    Epidemiológica publicados por el Ministerio de Salud del país. 
    Los archivos SintomasCasosConfirmados.csv y SintomasHospitalizados.csv 
    tienen una columna 'Síntomas' y una serie de columnas '[Fechas]', 
    donde por cada síntoma en una fila, se reporta el número acumulado a cada fecha 
    de casos confirmados con dicho síntoma (entre casos confirmados y hospitalizados, respectivamente). Incluye versiones con series de tiempo.

    Returns
    -------
    df :  DataFrame
       producto 21: ['Sintomas', 'Hospitalización', 'start_date', 'accumulated_infected']

    """
    url_dp1 ='https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto21/Sintomas.csv'
    dp1 = pd.read_csv(url_dp1, error_bad_lines=False)
    df_21 = dp1.melt(id_vars=['Sintomas', 'Hospitalización'],var_name="start_date",value_name='accumulated_infected')
    df_21['start_date']=df_21.start_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df_21['start_date']=pd.to_datetime(df_21['start_date'])
    return df_21

def producto_22():
    """
    Descargar producto 22 directamente del github del ministerio de MinCiencia
    
    Data Product 22 - Pacientes COVID-19 hospitalizados por grupo de edad:
    Este producto, que consiste de varios archivos, da cuenta del número acumulado
    del total de pacientes hospitalizados con diagnóstico COVID-19 por rango de edad
    y género. También da cuenta del número acumulado de pacientes internados con 
    diagnóstico COVID-19 en la Unidad de Cuidados Intensivos (UCI) por rango de edad.
    Se concatena la historia de los informes de Situación Epidemiológica publicados por
    el Ministerio de Salud del país. Los archivos presentan varias versiones de rangos etareos.
    Incluye versiones con series de tiempo.

    Returns
    -------
    df :  DataFrame
       producto 22: ['Grupo de edad', 'Sexo', 'start_date', 'accumulated_uci_beds'],

    """
    url_dp1 ='https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto22/HospitalizadosEtario_Acumulado.csv'
    dp1 = pd.read_csv(url_dp1, error_bad_lines=False)
    df = []
    name_columns = list(dp1.columns)
    row, col = dp1.shape
    for i in range(12,row):  # numero de filas
        for j in range(2, col):  # numero columna numero
            aux = []
            aux.append(dp1.iloc[i, 0])  # guardo grupo de edad
            aux.append(dp1.iloc[i, 1])  # guardo grupo de sexo
            aux.append(str(name_columns[j]))  # guardo el dia
            aux.append(dp1.iloc[i, j])  # numero de camas uci uso
            df.append(aux)
    
    df_22 = pd.DataFrame(df, columns=[name_columns[0],name_columns[1], 'start_date', 'accumulated_uci_beds'])
 
    df_22['start_date']=df_22.start_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df_22['start_date']=pd.to_datetime(df_22['start_date'])
    df_22=df_22[df_22.start_date>='2020-04-22'].copy()
    
    return df_22
    

def producto_24():
    """
    Descargar producto 24 directamente del github del ministerio de MinCiencia
    
    Data Product 24 - Hospitalización de pacientes COVID-19 en sistema integrado:
    Este producto da cuenta del número de pacientes en hospitalización con diagnóstico COVID-19
    según el tipo de cama que ocupan: Básica, Media, UTI y UCI. Se concatena la historia de 
    reportes diarios publicados por el Ministerio de Salud del país. 
    El archivo CamasHospital_Diario.csv, corresponde al reporte diario de la cantidad de
    pacientes en camas (Básica, Media, UCI o en UTI). Contiene las columnas 'Tipo de Cama', 
    y una serie de columnas '[Fecha]', donde estas contiene el número de ocupación por día, 
    por tipo de cama. Incluye versión con serie de tiempo.
    
    Se suman todas las camas que se tiene en el sistema, luego calcula la proporcion
    de camas utilizadas por tipo de cama y filtramos solo por camas UCI

    Returns
    -------
    df :  DataFrame
       producto 24:['start_date','proporcion']

    """
    url_dp1 ='https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto24/CamasHospital_Diario.csv'
    dp1 = pd.read_csv(url_dp1, error_bad_lines=False)
    df = []
    name_columns = list(dp1.columns)
    row, col = dp1.shape
    for i in range(0,row):  # numero de filas
        for j in range(1, col):  # numero columna numero
            aux = []
            aux.append(dp1.iloc[i, 0])  # guardo Tipo de cama
            aux.append(str(name_columns[j]))  # guardo el dia
            aux.append(dp1.iloc[i, j])  # numero de camas uci uso
            df.append(aux)
    
    df = pd.DataFrame(df, columns=[name_columns[0], 'start_date', 'accumulated_beds'])
 
    df['start_date']=df.start_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df['start_date']=pd.to_datetime(df['start_date'])
    #df=df[df.start_date>='2020-04-22'].copy()
    aux_24 = df[['start_date', 'accumulated_beds']].copy()
    aux_24 = aux_24.groupby(['start_date']).agg('sum').reset_index()
    df = pd.merge(df,aux_24, how='left', on=['start_date'])
    df['proporcion']=df.accumulated_beds_x/df.accumulated_beds_y
    
    df=df[df["Tipo de cama"]=='UCI'][['start_date','proporcion']].copy()
    return df


def producto_26():
    """
    Descargar producto 26 y 27 directamente del github del ministerio de MinCiencia
    
    Data Product 26 - Casos nuevos con síntomas por región: 
    Set de 3 archivos que dan cuenta del número de casos nuevos por día según resultado
    del diagnóstico que presentan síntomas, por región de residencia, reportados por
    el Ministerio de Salud. El archivo CasosNuevosConSintomas.csv contiene la
    columna ‘Región’, seguida por columnas correspondientes a ‘[Fecha]’. 
    Estas últimas columnas, ‘[Fecha]’, indican el número de casos nuevos con síntomas,
    por región, desde el 29-04-2020 hasta la fecha. Incluye versión con serie de tiempo. 

    Data Product 27 - Casos nuevos sin síntomas por región: 
    
    Set de 3 archivos que dan cuenta del número de casos nuevos por día según resultado
    del diagnóstico que son asintomáticos, por región de residencia, 
    reportados por el Ministerio de Salud desde el 29-04-2020. 
    El archivo CasosNuevosSinSintomas.csv contiene la columna ‘Región’, 
    seguida por columnas correspondientes a ‘[Fecha]’. Estas últimas columnas,
    ‘[Fecha]’, indican el número de casos nuevos sin síntomas, por región, 
    desde el 29-04-2020 hasta la fecha. Incluye versión con serie de tiempo.

    Returns
    -------
    df :  DataFrame
       producto 26+27:['Region', 'start_date', 'sintomatic_today', 'asintomatic_today']

    """

    url_dp1 ='https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto26/CasosNuevosConSintomas.csv'
    
    dp1 = pd.read_csv(url_dp1, error_bad_lines=False)
    df = []
    name_columns = list(dp1.columns)
    row, col = dp1.shape
    for i in range(0,row):  # numero de filas
        for j in range(1, col):  # numero columna numero
            aux = []
            aux.append(dp1.iloc[i, 0])  # guardo Region
            aux.append(str(name_columns[j]))  # guardo el dia
            aux.append(dp1.iloc[i, j])  # numero de camas uci uso
            df.append(aux)
    
    df = pd.DataFrame(df, columns=[name_columns[0], 'start_date', 'sintomatic_today'])
 
    df['start_date']=df.start_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df['start_date']=pd.to_datetime(df['start_date'])
    
    url_dp2 = 'https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto27/CasosNuevosSinSintomas.csv'
    dp2 = pd.read_csv(url_dp2, error_bad_lines=False)
    df2 = []
    name_columns = list(dp2.columns)
    row, col = dp2.shape
    for i in range(0,row):  # numero de filas
        for j in range(1, col):  # numero columna numero
            aux = []
            aux.append(dp2.iloc[i, 0])  # guardo Region
            aux.append(str(name_columns[j]))  # guardo el dia
            aux.append(dp2.iloc[i, j])  # numero de camas uci uso
            df2.append(aux)
    
    df2 = pd.DataFrame(df2, columns=[name_columns[0], 'start_date', 'asintomatic_today'])
    df2['start_date']=df2.start_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df2['start_date']=pd.to_datetime(df2['start_date'])
    
    
    
    df= pd.merge(df,df2,how='left', on=['start_date','Region'])
    df.fillna(0, inplace=True)
    return df

def producto_39():
    """
    Descargar producto 39 directamente del github del ministerio de MinCiencia
    
    Data Product 39 - Casos confirmados de COVID-19 según fecha de inicio de síntomas y notificación: 
    Set de 3 archivos que dan cuenta de los casos confirmados de COVID-19 
    según la fecha de inicio de síntomas y fecha de notificación a nivel nacional. 
    Refleja la información del último informe epidemiológico publicado por el Ministerio de Salud del país.



    Returns
    -------
    df :  DataFrame
       producto 39:['Categoria', 'Serie', 'start_date', 'infected']

    """
    url_dp1 ='https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto39/NotificacionInicioSintomas.csv'
    dp1 = pd.read_csv(url_dp1, error_bad_lines=False)
    df_39 = dp1.melt(id_vars=['Categoria', 'Serie'],var_name="start_date",value_name='infected')
    df_39['start_date']=df_39.start_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df_39['start_date']=pd.to_datetime(df_39['start_date'])
    return df_39


def producto_47():
    """
    Descargar producto 47 directamente del github del ministerio de MinCiencia
    
    Data Product 47 - Media Movil de Casos Nuevos por 100,000Hab.: 
        Este producto da cuenta de la media movil semanal de casos nuevos confirmados 
        por region, normalizado por cada 100,000 habitantes. 
    Nos quedamos solo con el total nacional
    Returns
    -------
    df_47 : DataFrame
       producto 47:['start_date', 'media_movil_casos_nuevos']

    """
    url_dp1 ='https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/master/output/producto47/MediaMovil_std.csv'
    df_47 = pd.read_csv(url_dp1, error_bad_lines=False)
    df_47=df_47[df_47.Region=='Total'].copy()
    df_47.rename(columns={'Fecha':'start_date', 'Media movil':'media_movil_casos_nuevos'},inplace=True)
    df_47['start_date']=df_47.start_date.apply(lambda x: datetime.strptime(x, '%Y-%m-%d'))
    df_47['start_date']=pd.to_datetime(df_47['start_date'])
    df_47.fillna(0, inplace=True)
    return df_47[['start_date', 'media_movil_casos_nuevos']]


def producto_77():
    """
    Descargar producto 77 directamente del github del ministerio de MinCiencia
    
    Data Product 77 - Avance por rango etario y región en Campaña de Vacunación COVID-19: 
    Este producto da cuenta del avance en la campaña de vacunación contra Sars-Cov-2 por rango etario y región. Ver más
    
    El problema de este producto es que es una tabla que se sobreescribe con cierta periocidad, 
    para solucionar este problema generamos una función previa que busca todas las actulizaciones que han exitido en es producto
    luego guardamos las llaves y la data correspondiente a esa llave.
    En caso contrario avisamos que no existe esa data, para esto usamos las funciones: call_commit_product_77,read_csv_file_producto_77
    
    luego arreglar datos faltantes , y chaquear que la serie de tiempo sea creciciente 
    
    
    Returns
    -------
    df :  DataFrame
       producto 47:['start_date', 'media_movil_casos_nuevos']
       
    
    old code: 
        
    df['accumulated_vaccinated']=df.groupby(
        df_date_columns[:-1])['accumulated_vaccinated'].transform(
            lambda v: v.ffill())
            
    df['accumulated_vaccinated']=df.groupby(
        df_date_columns[:-1])['accumulated_vaccinated'].cummax()
    
    try to change the values by value before
    f1 = lambda x:  x.rolling(1, min_periods=1).sum().shift(periods=1, fill_value=0)
    df['accumulated_vaccinated_2']=df.groupby(
        df_date_columns[:-1])['accumulated_vaccinated_1'].apply(f1)
    
    df['accumulated_vaccinated_2']=df[[
    "accumulated_vaccinated_2","accumulated_vaccinated_1"]].max(axis=1)


    """
    
    start_time = time.time()
    
    shap_date_arr = call_commit_product_77() #colum 1:date ; colum2: shap
    
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/'
    save_date_arr = np.loadtxt(path_projet+path_data+'scrapping_file/test_2.txt',dtype='str')
    save_folder=path_projet+path_data+"scrapping_file/"
    df_list=[]
    for i in range(shap_date_arr.shape[0]):
        shap = shap_date_arr[i,1]
        date = shap_date_arr[i,0]
        try:
            if np.any(save_date_arr[:] == str(date)):
                df=pd.read_csv(save_folder+"df_"+str(date)+".csv")
            else:
                df =  read_csv_file_producto_77(shap,date)
                save_date_arr=np.append(save_date_arr,str(date))   
                np.savetxt(save_folder+'test_2.txt',save_date_arr, fmt='%s')
                
            df_list.append(df)
        except Exception:
            print( str(date)+' : This is empty or they use other messages in commit')
    
    df = pd.concat(df_list, ignore_index=True)
    
    df['start_date']=pd.to_datetime(df['start_date'])

    #arreglar datos faltantes ...poco eficiente
    df_date, df_date_columns=get_df_date_77(df, missing_values=True)
    df= pd.concat([ df, df_date], ignore_index=True)
    df = df.sort_values(by=df_date_columns).reset_index(drop=True)
    # si el valor es na, tomo el promedio entre el anterior y posterior 
    # si el valor anterior es mayor reemplazo el valor anterior
    #shift me desplazo un periodo
    
    start_date='2021-02-02' #day before the vaccination schedule began
    end_date='2021-02-21' #day before start the data
    
    df_date, df_date_columns=get_df_date_77(df, missing_values=False,start_date=start_date, end_date=end_date)
    df= pd.concat([ df, df_date], ignore_index=True)
    df = df.sort_values(by=df_date_columns).reset_index(drop=True)
    #set some values :
    #firt dose day of vaccine 
    df['accumulated_vaccinated']=df['accumulated_vaccinated'].where(df.start_date!=df.start_date.unique()[0],0)
    
    
    #2do dose day of vaccine 
    mask1=(df.start_date>=np.datetime64(start_date)+np.timedelta64(1, 'D')-np.timedelta64(9, 'D'))&(
        df.start_date<=np.datetime64(start_date)+np.timedelta64(1, 'D')+np.timedelta64(12, 'D'))&(
            df['Dosis']=='Primera')
    mask2 =(df.start_date>=np.datetime64(end_date)+np.timedelta64(1, 'D'))&(
        df.start_date<=np.datetime64(end_date)+np.timedelta64(1, 'D')+21)&(
            df['Dosis']=='Segunda')
    mask3 =(df.start_date==np.datetime64(start_date))&(df['Dosis']=='Segunda')
    aux =df.loc[mask2,'accumulated_vaccinated'].tolist()
    df.loc[mask1,'accumulated_vaccinated']= aux
    df.loc[mask3,'accumulated_vaccinated']= 0
    
    #df['accumulated_vaccinated']=df['accumulated_vaccinated'].where(
        #(df.start_date!=np.datetime64(start_date))&(df['Dosis']=='Segunda'),0)
    

        
    
    df['accumulated_vaccinated']=df.groupby(
        df_date_columns[:-1])['accumulated_vaccinated'].apply(lambda group: group.interpolate(limit=30,method='polynomial', order=3))#.apply(np.floor)
    
    df['accumulated_vaccinated'] =  np.floor(df['accumulated_vaccinated'])
    df['accumulated_vaccinated']=df.groupby(
        df_date_columns[:-1])['accumulated_vaccinated'].cummax()
    
    df['vaccinated_today'] =(
        df.groupby(['Region', 'Dosis', 'edad'])['accumulated_vaccinated'].diff()
        /(df.groupby(['Region', 'Dosis', 'edad'])['start_date'].diff()/ np.timedelta64(1, 'D'))
        )
    
    
    
    df['Grupo de edad']=np.nan
    df['Grupo de edad']=np.where(df['edad']<=39,'<=39',df['Grupo de edad'] )
    df['Grupo de edad']=np.where((df['edad']>39)&(df['edad']<=49),'40-49',df['Grupo de edad'] )
    df['Grupo de edad']=np.where((df['edad']>49)&(df['edad']<=59),'50-59',df['Grupo de edad'] )
    df['Grupo de edad']=np.where((df['edad']>59)&(df['edad']<=69), '60-69',df['Grupo de edad'] )
    df['Grupo de edad']=np.where(df['edad']>69,'>=70',df['Grupo de edad'] )
    
    
    df.drop(columns=['edad'], inplace= True)
    df = df.groupby(['Region', 'Dosis', 'Grupo de edad','start_date']).agg('sum').reset_index()
    
    
    end_time = round(time.time()-start_time,2)
    print("Execution time:" + str(end_time) +" s.")
    print("Porducto 77 ready")
    print("="*40)
    return  df


def call_commit_product_77(last_date_check='2021-06-13'):
    """
    Main idea:
    search in all the commit made for product 77
    
    
    For each date consulted: 
        check if you have saved the commit in a project folder,
        if not, look for it in the minciencia github. 
        
       check if there is data for the product 77 for the commit,
       if there is data we save it, otherwise it alerts the 
       user that there is no data
        
    
    *by default we use the first date on 2021-02-22,
    since there is no vaccination data before that.
    
    Parameters
    ----------
    last_date_check : str datetime, optional
        Last date check in github historial commit. The default is '2021-06-13'.

    Returns
    -------
    shap_date_arr: numpy array shape (Len(Time),2 )
        for each row is a tuple of date time and .

    """

    url = "https://api.github.com/repos/MinCiencia/Datos-COVID19/commits?page=1&sha=master"
    #date_arr = np.arange(np.datetime64('2021-02-22'),np.datetime64('today', 'D')+ np.timedelta64(1, 'D'))
    if last_date_check==None:
        date_arr = np.arange(np.datetime64('2021-02-22'),np.datetime64('today', 'D')+ np.timedelta64(1, 'D'))
    else:
        date_arr = np.arange(np.datetime64('2021-02-22'),
                         np.datetime64('2021-06-13'))
    
    
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/'
    
    save_date_arr = np.loadtxt(path_projet+path_data+'scrapping_file/test.txt',dtype='str')
    save_folder=path_projet+path_data+"scrapping_file/"
    shap_date_arr =[]
    for i in range(0,len(date_arr)-1):
        #print(str(date_arr[i]))
        if np.any(save_date_arr[:] == str(date_arr[i])):
            with open(save_folder+str(date_arr[i])+'.txt') as json_file:
                data = json.load(json_file)
        else:
            since="&since="+str(date_arr[i])+"T00:00:00Z"
            until="&until="+str(date_arr[i+1])+"T00:00:00Z"
            response = urllib.request.urlopen(url+since+until)
            data = json.loads(response.read())
            with open(save_folder+str(date_arr[i])+'.txt', 'w') as outfile:
                json.dump(data, outfile)
                
            save_date_arr=np.append(save_date_arr,str(date_arr[i]))   
            np.savetxt(save_folder+'test.txt',save_date_arr, fmt='%s')
        try:
            message_commit='Added data from campaña de vacunacion'
            data_list=[item for item in data 
                       if item['commit']['message']==message_commit]
            shap_list=[item['sha'] for item in data_list]
            date_list=[np.datetime64(item['commit']['author']['date'][:-1]) 
                       for item in data_list]
            index_last_shape=np.argmax(date_list)
            shap_date_arr=np.append(
                shap_date_arr,
                np.array([date_arr[i],shap_list[index_last_shape]]),
                axis=0)
        except Exception:
            print(str(date_arr[i])+' : This is empty or they use other messages in commit')
            pass
    print("ID commit is save!")
    print("="*40)
    return shap_date_arr.reshape((-1,2))






def read_csv_file_producto_77(commit,date):
    """
    Main idea:
       load  data for that commit and day, I save the data 

    Parameters
    ----------
    commit : string
        github commit.
    date : str date
        DESCRIPTION.

    Returns
    -------
    df : Dataframe 
        df: region, dosis, edad,accumulated_vaccinated .

    """
    save_folder="scrapping_file/"
    url = "https://raw.githubusercontent.com/MinCiencia/Datos-COVID19/"+commit+"/output/producto77/total_vacunados_region_edad.csv"
    dp1 = pd.read_csv(url, error_bad_lines=False)
    df = []
    name_columns = list(dp1.columns)
    row, col = dp1.shape
    for i in range(row):  # numero de filas
        for j in range(2, col):  # numero columna numero
            aux = []
            aux.append(dp1.iloc[i, 0])  # guardo grupo de region
            aux.append(dp1.iloc[i, 1]) #guardo dosis
            aux.append(name_columns[j])  #edad
            aux.append(dp1.iloc[i, j])  # numero de casos notificados
            df.append(aux)
    df = pd.DataFrame(df, columns=[name_columns[0],name_columns[1], 'edad', 'accumulated_vaccinated'])
    df['start_date']=date
    
    df.to_csv(save_folder+"df_"+str(date)+".csv",index=False)
    return df

def get_df_date_77(df, missing_values=True,start_date='2021-02-02', end_date='2021-02-21'):
    """
    Main idea: 
        generate all the combinations for the set region, dose, age and dates

    Parameters
    ----------
    df : Dataframe
        df from producto 77.
    missing_values : bolean, optional
        if we are interested in all possible dates True, otherwise False ,
        combinations are generated from start_date and end_date dates. 
        The default is True.
    start_date : string datetime, optional
        Suppose the day starts before the vaccine starts. The default is '2021-02-02'.
    end_date : string datetime, optional
        include that day . The default is '2021-02-21'.

    Returns
    -------
    df_date : Dataframe
        DESCRIPTION.
    df_date_columns : list of string
        ['Region', 'Dosis', 'edad', 'start_date'].
    
    """

    if missing_values:
        date_arr=np.setdiff1d(
            pd.date_range(
                df.start_date.unique()[0],
                df.start_date.unique()[-1],
                freq='1d'),
            df.start_date.unique())
    else: 
        date_arr=pd.date_range(
            np.datetime64(start_date)- 19 ,#- (np.datetime64(end_date)-np.datetime64(start_date))
            np.datetime64(end_date),
            freq='1d')
    
    df_date=pd.DataFrame(
            list(itertools.product(
                    df['Region'].unique(),
                    df['Dosis'].unique(),
                    df['edad'].unique(),
                    date_arr)))
    df_date_columns = ['Region', 'Dosis', 'edad', 'start_date']
    df_date.columns=df_date_columns
    df_date['accumulated_vaccinated']= np.nan
    
    return df_date, df_date_columns
