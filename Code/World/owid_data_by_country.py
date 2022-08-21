#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 30 12:05:35 2022

@author: ineira
"""
import pandas as pd
import numpy as np

from owid_data import get_owid_dataframe

def call_data_by_country_owid_data(country="Argentina",
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
                                   update_data=False ):
    
    
    df=get_owid_dataframe(columns= columns,update_data=update_data)
    
    
    print(f"You select {country} for the analysis")
    print(f"You select update data: {update_data}")
    
    
    if country in df.location.unique():
        df=df[df.location==country].copy().reset_index()
        # print all available columns
        #print(df.columns)
        print(f"This is data from {country}!")
        print("Dates!")
        print("Nª dates: {}".format(len(df.date.unique())))
        print("Frist and last date: {} to {}".format(df.date.unique()[0],
                                                    df.date.unique()[-1]))
        df['date'] = pd.to_datetime(df['date'], format= '%Y-%m-%d')
        print("Check if list of dates is complete! ") 
        print(df.date.diff().value_counts())
        print(" If you see a value different than 1 days, is a problem with the data")
        
        print("======"*10)
        print("Inspect data")
        
        for column in columns:
            if column not in ['location',"date"]:
                index=df.loc[~df[column].isnull()].index[0]
                print("Frist {} was: {}".format(column,
                                                np.datetime_as_string(df.iloc[[index]].date.values[0], unit='D')))
        
        
    
    return df


if __name__ == "__main__":

    df_country=call_data_by_country_owid_data()