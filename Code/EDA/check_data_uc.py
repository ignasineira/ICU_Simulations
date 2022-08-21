#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 26 18:46:00 2022

@author: ineira
"""


df1['Grupo de edad']=np.nan
df1['Grupo de edad']=np.where(df1['edad']<=39,'<=39',df1['Grupo de edad'] )
df1['Grupo de edad']=np.where((df1['edad']>39)&(df1['edad']<=49),'40-49',df1['Grupo de edad'] )
df1['Grupo de edad']=np.where((df1['edad']>49)&(df1['edad']<=59),'50-59',df1['Grupo de edad'] )
df1['Grupo de edad']=np.where((df1['edad']>59)&(df1['edad']<=69), '60-69',df1['Grupo de edad'] )
df1['Grupo de edad']=np.where(df1['edad']>69,'>=70',df1['Grupo de edad'] )
df1['condicion_name']=df1['condicion al 15 mayo'].map({1:'vivo',2: 'fallecido', 3:'hospitalizado'})
df1['Variante']=df1['resultado_variantec'].map({1:'P1',2: 'P2', 3:'UK'})
df1['Variante']= df1['Variante'].replace(np.nan, 'Original')

df1['Variante'].value_counts()
df1['condicion al 15 mayo'].value_counts()
df1.groupby(['Variante','condicion_name']).size().unstack(fill_value=0)
df1.groupby(['Grupo de edad','condicion_name']).size().unstack(fill_value=0)

dict_factor=dict()
dict_factor['dict_factor']=[0.011799503728593824,
 0.035921368014992974,
 0.07184934248279035,
 0.003324302819280508,
 0.083]

index=[0, 1, 2, 3, 4]
dict_factor['Grupo de edad']=['40-49', '50-59', '60-69', '<=39', '>=70']

df_factor=pd.DataFrame.from_dict(dict_factor,)

df_16=prepare_producto_16()
df_10=producto_10()

start_date='2020-07-01';end_date='2020-12-31'
df=df_10[(df_10['start_date']>=start_date)&(df_10['start_date']<=end_date)]
df=pd.merge(df_10,df_16, how='inner', on=['Grupo de edad','start_date'])
df=pd.merge(df,df_factor, how='inner', on=['Grupo de edad'])

df['fatality']=df['accumulated_dead']/df['accumulated_infected']
df['fatality_icu']=df['accumulated_dead']/(df['accumulated_infected']*df['dict_factor'])

df=df[(df['start_date']>=start_date)&(df['start_date']<=end_date)]
df[df.start_date=='2020-12-31'].accumulated_infected

df[df.start_date=='2020-12-31'][['Grupo de edad','accumulated_infected','accumulated_dead','fatality']]


plt.subplots( figsize=(15, 9))
ax=sns.lineplot(
    data=df,
    x='start_date', y ='fatality'
    , palette= 'bright'#'Set3'
    ,legend=True,hue='Grupo de edad'
)

ax.axes.set_title("Timeseries for fatality " ,fontsize=15)
degrees = 70
plt.xticks(rotation=degrees)
plt.ylabel('Fatality', fontsize = 15) # x-axis label with fontsize 15 
plt.xlabel('Time', fontsize = 15)
plt.tight_layout()
plt.show()


plt.subplots( figsize=(15, 9))
ax=sns.lineplot(
    data=df,
    x='start_date', y ='fatality_icu'
    , palette= 'bright'#'Set3'
    ,legend=True,hue='Grupo de edad'
)

ax.axes.set_title("Timeseries for fatality uci " ,fontsize=15)
degrees = 70
plt.xticks(rotation=degrees)
plt.ylabel('Fatality', fontsize = 15) # x-axis label with fontsize 15 
plt.xlabel('Time', fontsize = 15)
plt.tight_layout()
plt.show()