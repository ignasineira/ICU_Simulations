#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 19:06:04 2022

@author: ineira

Data on SARS-CoV-2 variants by Our World in Data

Our data on SARS-CoV-2 sequencing and variants is sourced from GISAID, 
a global science initiative that provides open-access to genomic data
 of SARS-CoV-2. We recognize the work of the authors and laboratories 
 responsible for producing this data and sharing it via the GISAID 
 initiative.

https://covariants.org/


https://github.com/owid/covid-19-data/tree/master/public/data/archived/variants
-->
https://covariants.org/
-->
https://github.com/hodcroftlab/covariants




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



from owid_data import download_owid_data,get_owid_dataframe

dir_mame = os.path.dirname(os.path.realpath('__file__')) #os.getcwd()


    
def download_covariants_data(update_data=False):
    """
    

    Parameters
    ----------
    update_data : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    country_csv_save_file : TYPE
        DESCRIPTION.
    casecounts_csv_save_file : TYPE
        DESCRIPTION.
        
        
    perCountryData has 3 different type of sourse check per_country_intro_content/region
    
    tip:
        with open(country_csv_save_file) as f:
            perCountryData = json.load(f)
        
        world_data = perCountryData["regions"][0]["distributions"]
        per_country_intro_content = perCountryData["regions"][0]["per_country_intro_content"]
        max_date = perCountryData["regions"][0]["max_date"]
        min_date = perCountryData["regions"][0]["min_date"]
        cluster_names = perCountryData["regions"][0]["cluster_names"]
        

    """
    country_csv_filename="perCountryData.json"
    casecounts_csv_filename="perCountryDataCaseCounts.json"
    
    path_data_covariants="https://raw.githubusercontent.com/hodcroftlab/covariants/master/web/data/"
    country_csv_path = os.path.join(path_data_covariants, country_csv_filename)
    casecounts_csv_path = os.path.join(path_data_covariants, casecounts_csv_filename)
    
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/World/'
    
    country_csv_save_file = path_projet+path_data+country_csv_filename
    casecounts_csv_save_file = path_projet+path_data+casecounts_csv_filename
    os.makedirs(os.path.dirname(country_csv_save_file), exist_ok=True)
    os.makedirs(os.path.dirname(casecounts_csv_save_file), exist_ok=True)
    
    last_update_file=path_projet+path_data+"last_update_file_covariants_data.json"
    os.makedirs(os.path.dirname(last_update_file), exist_ok=True)
    if update_data:
        urlretrieve(country_csv_path, country_csv_save_file)
        urlretrieve(casecounts_csv_path, casecounts_csv_save_file)
        today= datetime.now().strftime('%Y-%m-%d')
        print(f"Update the file: {today}")
        
        with open(last_update_file, "w") as out:
            json.dump({'last_update':today},out)
    
    else:
        
        with open(last_update_file) as f:
            last_update=json.load(f)
        print(f"The last Update was: {last_update['last_update']}")
    
    return country_csv_save_file,casecounts_csv_save_file
    
    
    
def case_counts_update(countries_list=['Brazil','Chile','Argentina'], update_data=False, threshold=0.02,period_pass=0.1):
    """
    Main idea:
        1.group the data in a week


    Parameters
    ----------
    countrys : TYPE, optional
        DESCRIPTION. The default is ['Brazil'].
    update_data : TYPE, optional
        DESCRIPTION. The default is False.
    threshold : float, optional
        0.03 is 3%. The default is 0.02.
    period_pass : float, optional
        0.5 is 50%. The default is 0.4.

    Returns
    -------
    None.
    
    
    check data:
    
    world_data_counts[-1]["distribution"][0].keys()
    week: 
    total_sequences: total sequences
    stand_total_cases: how many cases per million do we find for a date and country
    stand_estimated_cases: percent_counts*stand_total_cases     **percent_counts= float(n) / total_sequences
    percent_total_cases: total_sequences / total_cases
    

    """

    # key: country names in covariants
    # value: country name in owid
    alernative_country_names = {
        "USA" : "United States",
        "Czech Republic"  : "Czechia",
        "Côte d'Ivoire": "Cote d'Ivoire",
        "Democratic Republic of the Congo": "Democratic Republic of Congo",
        "Sint Maarten": "Sint Maarten (Dutch part)",
    }
    #owid data
    owid =get_owid_dataframe(update_data=update_data)
    if countries_list !=None:
        print("Check if the list of countrys is in owid data")
        print(countries_list)
        check_list=[]
        owid_countries= owid.location.unique().tolist()
        for item in countries_list:
            if item in owid_countries:
                check_list.append(item)
        
        owid=owid[owid.location.isin(check_list)].copy()
    
    
    owid["date_formatted"] = owid["date"].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
    #group the data in a week
    owid["date_2weeks"] = owid["date_formatted"].apply(to2week)
    owid_grouped = owid.groupby(["date_2weeks", "location"])[["new_cases_per_million", "new_cases"]].sum().reset_index()
    
    #covariants data
    country_csv_save_file,casecounts_csv_save_file=download_covariants_data(update_data=update_data)
    with open(country_csv_save_file) as f:
        perCountryData = json.load(f)
    
    world_data = perCountryData["regions"][0]["distributions"]
    per_country_intro_content = perCountryData["regions"][0]["per_country_intro_content"]   
    max_date = perCountryData["regions"][0]["max_date"]
    min_date = perCountryData["regions"][0]["min_date"]
    cluster_names = perCountryData["regions"][0]["cluster_names"]
     
    world_data_counts = []
    weeks = []
    countries = []

    
    for i in range(len(world_data)):
        country = world_data[i]["country"]
        if countries_list !=None:
            if country not in countries_list:
                continue
        
        if country not in owid_grouped["location"].values and country not in alernative_country_names:
            print("Attention! Country not found in owid data: " + country)
            continue
        print(f"this country meets both criteria: {country}")
        
        country_owid = country
        if country in alernative_country_names:
            country_owid = alernative_country_names[country]
        countries.append(country)
        #save info on dict
        world_data_counts.append({"country": country, "distribution": []})
        for j in world_data[i]["distribution"]:
            """
            1.calcular el porcentaje por variante 
            2.cuandtos casos por millon encontramos para una fecha y pais
            3. cuanto en total para una fecha y pais
            
            5. calcular la cantidad de test de variante vs el total de test
            
            """
            cluster_counts = j["cluster_counts"]
            total_sequences = j["total_sequences"]
            week = j["week"]
            
            if (week not in weeks) and ("2020" not in week):# Ignore 2020
                weeks.append(week)
            
            #calcular el porcentaje por variante 
            percent_counts = {c : float(n) / total_sequences for c, n in cluster_counts.items()}
            
            #cuandtos casos por millon encontramos para una fecha y pais
            stand_total_cases = owid_grouped.loc[(owid_grouped.date_2weeks == week) & (owid_grouped.location == country_owid)]["new_cases_per_million"]
            total_cases = owid_grouped.loc[(owid_grouped.date_2weeks == week) & (owid_grouped.location == country_owid)]["new_cases"]
    
            if len(stand_total_cases) > 0: 
                stand_total_cases = int(stand_total_cases.iloc[0])
            else:  # No count data
                continue  # Skip if no count data
    
            if len(total_cases) > 0:
                total_cases = int(total_cases.iloc[0])
            else:  # No count data
                continue  # Skip if no count data
    
            stand_estimated_cases = {c: round(float(n) * stand_total_cases) for c, n in percent_counts.items()}
            stand_estimated_cases["others"] = max(stand_total_cases - sum(stand_estimated_cases.values()),0) #se asune que la original
            percent_total_cases = total_sequences / total_cases if total_cases != 0 else None
    
            world_data_counts[-1]["distribution"].append({"week": week, "total_sequences": total_sequences, "stand_total_cases" : stand_total_cases, "stand_estimated_cases" : stand_estimated_cases, "percent_total_cases" : percent_total_cases})
                
    df = pd.DataFrame(columns=sorted(weeks), index=sorted(countries))
    
    for i in range(len(world_data_counts)):
        country = world_data_counts[i]["country"]
        #if country not in df.index.tolist():continue
        for j in world_data_counts[i]["distribution"]:
            week = j["week"]
            if "2020" in week: continue  # Ignore 2020
            percent_total_cases = j["percent_total_cases"]
            df[week][country]= percent_total_cases
    

    print("\nAt threshold " + str(threshold*100) + "% (sorted):")
    total_weeks = len(weeks)
    df_threshold = (df>=threshold).sum(axis=1)
    p = False
    for country, row in df_threshold.sort_values(ascending=False).iteritems():
        perc = row/float(total_weeks)
        if perc < period_pass and not p:
            print("-----------------------------------------------")
            p = True
        t = ""
        if len(country) < 7:
            t = "\t"
        print(country + ":\t" + t + "Pass for " + str(row) + "/" + str(total_weeks) + " 2-week-periods (" + str(round(perc*100, 2)) + "%)" )
        
    print("\nAt " + str(threshold*100) + "% and " + str(period_pass*100) + "% period-pass threshold, the following countries would plot:")
    df_period_pass = df_threshold[(df_threshold/float(total_weeks)) >= period_pass]
    for i in df_period_pass.index:
        print(i)
        
    
    countries_pass = df_threshold[(df_threshold/float(total_weeks)) >= period_pass].index
    world_data_counts_cutoff = [x for x in world_data_counts if x["country"] in countries_pass]
    print(f"{len(world_data_counts_cutoff)}/{len(world_data_counts)} countries have passed threshold {threshold} and period_pass {period_pass}")
    
    
    
    df_csv_filename="case_counts_market_share_total_sequences.csv"
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/World/'
    
    df_csv_save_file = path_projet+path_data+df_csv_filename
    os.makedirs(os.path.dirname(country_csv_save_file), exist_ok=True)
    df.to_csv(df_csv_save_file)
    
    
    #save json 
    output_csv_filename = "perCountryDataCaseCounts_severidad.json"
    output_csv_save_file = path_projet+path_data+output_csv_filename
    os.makedirs(os.path.dirname(output_csv_save_file), exist_ok=True)
    
    with open(output_csv_save_file, "w") as out:
        json.dump({"regions": [{"region": "World", "distributions" : world_data_counts_cutoff, "per_country_intro_content": per_country_intro_content, "max_date": max_date, "min_date": min_date, "cluster_names": cluster_names}]}, out, indent=2, sort_keys=True)

    
    
def covariants_severidad(update_data=False,country='Argentina'):
    """
    It is worth interpreting with caution:

    Not all samples are representative - sometimes some samples are more likely to be sequenced than others 
    (for containing a particular mutation, for example)
    The last data point - this often has incomplete data and may change as more sequences come in
    Frequencies that are very 'jagged' - this often indicates low sequencing numbers and so may not be truly representative of the country
    In many places, sampling may not be equal across the region: 
    samples may only cover one area or certain areas. It's important not to assume frequencies shown are necessarily representative.

    Parameters
    ----------
    update_data : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    variant : TYPE
        DESCRIPTION.

    """
    path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]+'ICU_Simulations/'
    path_data='Data/Input/World/'
    input_csv_filename = "perCountryDataCaseCounts_severidad.json"
    input_csv_save_file = path_projet+path_data+input_csv_filename
    if not os.path.exists(input_csv_save_file):
        case_counts_update(update_data=update_data)
        
    with open(input_csv_save_file) as f:
        perCountryData = json.load(f)
    
    world_data = perCountryData["regions"][0]["distributions"]
    max_date = perCountryData["regions"][0]["max_date"]
    min_date = perCountryData["regions"][0]["min_date"]
    cluster_names = perCountryData["regions"][0]["cluster_names"]
    
    
    brazil=search_list_by_country(world_data, country=country)
     
    df=pd.json_normalize(brazil,record_path= ["distribution"], max_level=0)
    df["date"]=pd.to_datetime(df['week'], format= '%Y-%m-%d')
    variant  = pd.json_normalize(df['stand_estimated_cases']).set_index(df.date)
    variant_per = variant.apply(lambda x: x/x.sum(), axis=1)
    
    
    fig,ax = plt.subplots(figsize=(15,9))
    variant_per.plot.area(ax=ax, cmap=cmap)
    plt.ylim(0,1)
    ax.set_yticklabels(['0','20%','40%','60%','80%','100%'])

    #put the lengend outside the plot
    plt.legend(bbox_to_anchor =(1.05,1))
    plt.title(f'Market share for covariants ({country})', fontsize=20)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Percentaje", fontsize=12)
    ax.set_xlim([variant_per.index.min()+pd.DateOffset(-2),variant_per.index.max()+pd.DateOffset(2)])
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # formatter for major axis only
    # Also moves the bottom of the axes up to make room for them.
    fig.autofmt_xdate()
    #plt.gcf().autofmt_xdate()
    
    ax.axvspan(date2num(datetime(2021,5,15)), date2num(datetime(2021,5,16)), 
           label="LAST DATE",color="red", alpha=0.5)
    ax.axvspan(date2num(datetime(2021,1,17)), date2num(datetime(2021,1,18)), 
           label="first dosis",color="red", alpha=0.5)
    
    plt.show()
    
    
    
    chile=search_list_by_country(world_data, country='Chile')
     
    df=pd.json_normalize(chile,record_path= ["distribution"], max_level=0)
    df["date"]=pd.to_datetime(df['week'], format= '%Y-%m-%d')
    variant  = pd.json_normalize(df['stand_estimated_cases']).set_index(df.date)
    df_aux = variant.apply(lambda x: x/x.sum(), axis=1)
    
    
    fig,ax = plt.subplots(figsize=(15,9))
    df_aux.plot.area(ax=ax, cmap=cmap)
    plt.ylim(0,1)
    ax.set_yticklabels(['0','20%','40%','60%','80%','100%'])

    #put the lengend outside the plot
    plt.legend(bbox_to_anchor =(1.05,1))
    plt.title('Market share for covariants (Chile)', fontsize=20)
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Percentaje", fontsize=12)
    ax.set_xlim([df_aux.index.min()+pd.DateOffset(-2),df_aux.index.max()+pd.DateOffset(2)])
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    # formatter for major axis only
    # Also moves the bottom of the axes up to make room for them.
    fig.autofmt_xdate()
    #plt.gcf().autofmt_xdate()
    
    ax.axvspan(date2num(datetime(2021,5,15)), date2num(datetime(2021,5,16)), 
           label="LAST DATE",color="red", alpha=0.5)
    ax.axvspan(date2num(datetime(2021,1,17)), date2num(datetime(2021,1,18)), 
           label="first dosis",color="red", alpha=0.5)
    
    plt.show()
    
    return variant,variant_per
    
def to2week(x):
    iso_y, iso_w, iso_d = x.isocalendar()[:3]
    if iso_w==1:
        prev_week = x - timedelta(days=7)
        iso_y, iso_w, iso_d = prev_week.isocalendar()[:3]

    return datetime.strptime("{}-W{}-1".format(*(iso_y, iso_w // 2 * 2)), "%G-W%V-%u")


def search_list_by_country(world_data_counts, country='Brazil'):
    
    return [element for element in world_data_counts if element['country'] == country][0]

if __name__ == "__main__":
    download_covariants_data(update_data=True)
    get_owid_dataframe(columns= ["continent", "location", "date", "new_cases", "new_cases_per_million"],update_data=True)

    case_counts_update(countries_list=['Brazil','Chile','Argentina'], update_data=False, threshold=0.02,period_pass=0.1)
