# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 11:34:02 2021

@author: ignas

this file generates simulations given a set of parameters



"""
import os 
import sys


import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.dates import date2num
import matplotlib.dates as mdates
from scipy import stats
from scipy.stats import poisson, geom, bernoulli
from tqdm import tqdm
import scipy.sparse as sps


from datetime import  datetime


path_projet=os.getcwd().rsplit('ICU_Simulations',1)[0]
if path_projet[-1]!='/':
    path_projet+='/'
path_projet+='ICU_Simulations/'
path_ETL='Code/ETL'

module_path=path_projet+path_ETL
if module_path not in sys.path:
    sys.path.append(module_path)
    
from data_processing import read_data
from fit_probability_to_icu import find_probability_icu


np.random.seed(seed=2020)


'''
Funciones utilies
'''

def intervalo_confianza(data,inf_today, confidence=0.95, print_to_console=0, print_pres=4):
    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    s = np.std(data, ddof=1)
    h = (s / np.sqrt(n)) * stats.t.ppf((1 + confidence) / 2., n-1)
    if (print_to_console==1):
    	print("\t-------------------------")
    	print("\tUCI_real \tMedia\tDesv.t\tIntervalo")
    	print("\t-------------------------")
    	print("\t{}\t{}\t{}\t[{}, {}]".format(str(inf_today),str(round(m, print_pres)),
    									str(round(s, print_pres)),
    									str(round(m-h, print_pres)),
    									str(round(m+h, print_pres))
    								))
    	print("\t-------------------------")
    return m, s, m-h, m+h



class Camas_UCI():
    def __init__(self,data,ID_grupo_etario,probability_not_vac, p_to_uci,n_to_uci,p_in_uci,n_in_uci):
        """
        Not vacc include
        Input:
            data: dictionary with very useful information
            ID_grupo_etario: id of the age group to simulate
            probability_not_vac: probability of going to ICU
            p_to_uci: p is the probability of success  of the NB distribution for the time of going to ICU
            n_to_uci: n is the  the number of successes  of the NB distribution for the time of going to ICU
            p_in_uci: p is the probability of success  of the NB distribution for the time of in ICU
            n_in_uci: n is the  the number of successes  of the NB distribution for the time in ICU
        """
        self.data = data
        self.ID_grupo_etario= ID_grupo_etario
        self.probability_not_vac=probability_not_vac
        self.p_to_uci=p_to_uci 
        self.n_to_uci=n_to_uci
        self.p_in_uci=p_in_uci 
        self.n_in_uci=n_in_uci
        
    def people_going_to_uci_by_day(self,replicas,inf_today):
        """
        Parameters
        ----------
        replicas : int
            number of replicas/simulations to perform
        inf_today : int
            number of infected for a specific day

        Returns
        -------
        Sparce Matrix (replicas, inf_today)
            view result plt.spy(A,markersize=1)
            usar sparce matrix en realidad no tira numeros aleatorios por matriz
            #sps.random(replicas, inf_today, density=self.probability_not_vac, data_rvs=np.ones)
        """
        M=np.random.binomial(size=(replicas,inf_today),n=1,p=self.probability_not_vac)
        rows = np.where(M==1)[0].tolist()
        cols = np.where(M==1)[1].tolist()
        ones = np.ones(len(rows), np.uint32)
        S = sps.coo_matrix((ones, (rows, cols)), shape=(replicas,inf_today))#.tocsr()

        return S
    
    def people_time_go_to_uci_by_day(self,replicas,inf_today,window_slide=3,max_days_go_uci=30):
        """
        Parameters
        ----------
        replicas : int
            number of replicas/simulations to perform
        inf_today : int
            number of infected for a specific day

        **ponderaciones_to_uci: probability that distribution negative binomial

        Returns
        -------
        Sparce Matrix (replicas, inf_today)
            view result plt.spy(A,markersize=1)
            
        
        """
        #np.random.binomial(size=(replicas,inf_today),n=1,p=self.probability_not_vac)
        
        nb_matrix=np.random.negative_binomial(size=(replicas,inf_today),n=self.n_to_uci,p=self.p_to_uci)
               
        while len(np.where(nb_matrix>= max_days_go_uci)[0])!=0:
            rows,cols=np.where(nb_matrix>= max_days_go_uci)
            nums =np.random.negative_binomial(size=len(rows),n=self.n_to_uci,p=self.p_to_uci)
            for i,j,num in zip(rows,cols,nums):
                nb_matrix[i,j]=num
                
        return nb_matrix
    
    def people_time_in_uci_by_day(self,replicas,inf_today,max_days_in_uci= 100):
        """
        Parameters
        ----------
        replicas : int
            number of replicas/simulations to perform
        inf_today : int
            number of infected for a specific day

        **ponderaciones_to_uci: probability that distribution negative binomial

        Returns
        -------
        Sparce Matrix (replicas, inf_today)
            view result plt.spy(A,markersize=1)
        """
        
        #S = np.random.poisson(size=(replicas,inf_today),lam=self.mu_in_uci)
        nb_matrix=np.random.negative_binomial(size=(replicas,inf_today),n=self.n_in_uci,p=self.p_in_uci)
               
        while len(np.where(nb_matrix>= max_days_in_uci)[0])!=0:
            rows,cols=np.where(nb_matrix>= max_days_in_uci)
            nums =np.random.negative_binomial(size=len(rows),n=self.n_in_uci,p=self.p_in_uci)
            for i,j,num in zip(rows,cols,nums):
                nb_matrix[i,j]=num

        return nb_matrix
    
    #@classmethod
    def ICU_Simulations_camas_not_vac(self,replicas,confidence,error, start_date='2020-07-1',end_date='2020-12-01',max_days_go_uci=30,
                  max_days_in_uci= 100):
        
        
        inf=self.data['inf'][self.ID_grupo_etario,:].copy()
        shif_curve=[8, 7, 8, 7, 4]
        item=shif_curve[self.ID_grupo_etario]
        inf[0:-(item-1)]=self.data['inf'][self.ID_grupo_etario,item-1:]
        
        uci_real=self.data['uci'][self.ID_grupo_etario,:]
        uci_beds= np.zeros((replicas,inf.shape[0]))
        for t in range(inf.shape[0]):
            inf_today= int(inf[t])
            people_going_to_uci=self.people_going_to_uci_by_day(replicas,inf_today)
            time_go_to_uci=self.people_time_go_to_uci_by_day(replicas,inf_today,max_days_go_uci=max_days_go_uci) 
            time_in_uci=self.people_time_in_uci_by_day(replicas,inf_today,max_days_in_uci= max_days_in_uci)
            for i,j,v in zip(people_going_to_uci.row, people_going_to_uci.col, people_going_to_uci.data):
                go_to_uci=int(time_go_to_uci[i,j])
                in_uci=int(time_in_uci[i,j])
                if (in_uci>0)&(t+go_to_uci<inf.shape[0]):
                    uci_beds[i,t+go_to_uci:min(t+go_to_uci+in_uci,inf.shape[0])]+=1
                
        self.uci_beds = uci_beds
        """
        people_going_to_uci =np.array([self.people_going_to_uci_by_day(replicas,inf_today) for inf_today in inf])
        time_go_to_uci=np.array([self.people_time_go_to_uci_by_day(replicas,inf_today) for inf_today in inf])
        time_in_uci=np.array([self.people_time_in_uci_by_day(replicas,inf_today) for inf_today in inf])
        """
        #aux
        dateID_to_date = self.data['dateID_to_date']
        date_to_dateID = self.data["date_to_dateID"]
        dateID_start_date = self.data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
        dateID_end_date = self.data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
        #Calculamos n
        z_score = stats.t.ppf((1 + confidence) / 2., len(self.uci_beds)-1)
        s = np.std(self.uci_beds, ddof=1,axis=0)
        n = np.max(s*(z_score/error)**2)
        #Reportamos resultados parciales
        print("Resultados tras {} replicas es:".format(replicas))
        for date in range(dateID_start_date,dateID_end_date):
            intervalo_confianza(uci_beds[:,date],uci_real[date], confidence=confidence, print_to_console=1)
        
        print("En total se necesitan {} replicas. Ejecutando:".format(int(n)))
        #run the results that are needed
        if (int(n) - replicas)>0:
            replicas=int(n) - replicas
            uci_beds_2= np.zeros((replicas,inf.shape[0]))
            for t in range(inf.shape[0]):
                inf_today= int(inf[t])
                people_going_to_uci=self.people_going_to_uci_by_day(replicas,inf_today)
                time_go_to_uci=self.people_time_go_to_uci_by_day(replicas,inf_today,max_days_go_uci=max_days_go_uci) 
                time_in_uci=self.people_time_in_uci_by_day(replicas,inf_today,max_days_in_uci= max_days_in_uci)
                for i,j,v in zip(people_going_to_uci.row, people_going_to_uci.col, people_going_to_uci.data):
                    go_to_uci=int(time_go_to_uci[i,j])
                    in_uci=int(time_in_uci[i,j])
                    if (in_uci>0)&(t+go_to_uci<inf.shape[0]):
                        uci_beds_2[i,t+go_to_uci:min(t+go_to_uci+in_uci,inf.shape[0])]+=1
            self.uci_beds = np.concatenate((uci_beds,uci_beds_2),axis=0)
        #Reportamos resultados
        mean_reasult=[]
        for date in range(dateID_start_date,dateID_end_date):
             prom, desv, li, ls =intervalo_confianza(self.uci_beds[:,date],uci_real[date], confidence=confidence, print_to_console=1)
             mean_reasult.append([prom, desv, li, ls])
        
        self.mean_reasult=mean_reasult
        return mean_reasult
    
    
    
    
    
    def plot_uci_pred_2020(self,data, uci, W=29, pond=1, start_date='2020-07-20',end_date='2020-12-01', infected=False):
        groupID_to_group = data['groupID_to_group']
        dateID_to_date = data['dateID_to_date']
        date_to_dateID = data["date_to_dateID"]
        #start_date='2020-10-01'  
        uci_pred_05=np.percentile(self.uci_beds, 5,axis=0)
        uci_pred_95=np.percentile(self.uci_beds, 95,axis=0)
        uci_pred_50=np.percentile(self.uci_beds, 50,axis=0)
        uci_pred_mean=np.mean(self.uci_beds, axis=0,dtype=np.float64)
        dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
        dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
        cols =['Grupo de edad', 'start_date','inf', 'uci_real', 'uci_pred','uci_pred_05','uci_pred_95','uci_pred_50','uci_pred_mean']
        lst = []
        g=self.ID_grupo_etario
        for date in range(dateID_start_date,dateID_end_date+1): 
            info = [groupID_to_group[g],dateID_to_date[date],data['inf'][g,date]*pond,data['uci'][g,date]]
            info.append(uci[g,date-(W-1)])
            info.append(uci_pred_05[date])
            info.append(uci_pred_95[date])
            info.append(uci_pred_50[date])
            info.append(uci_pred_mean[date])
            lst.append(info)
        df_res = pd.DataFrame(lst, columns=cols)
        df_res["start_date"] = pd.to_datetime(df_res.start_date)
        for i, (group_name, group_df) in enumerate(df_res.groupby(["Grupo de edad"])):
        #for i in range(2):
        
            #plt.subplots(figsize=(15, 9))
            # plot the group on these axes
            if group_name=='>=70':
                ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred','inf','uci_pred_05','uci_pred_95'])
                
            else:
                if infected:
                    ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred','inf','uci_pred_05','uci_pred_95'])
                else:
                    ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred','uci_pred_05','uci_pred_95'])
            # set the title
            ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
            ax.grid()
            
            ax.xaxis_date()
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            
            ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,3)), 
                   label="older adults",color="red", alpha=0.3)
            
            ax.axes.set_title("Timeseries UCI for "+group_name ,fontsize=15)
            ax.set_ylabel('N° beds', fontsize=10)
            ax.set_xlabel('Date', fontsize=10)
            """
            try:
                plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name) +'.png')
            except:
                plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name)[2:] +'.png')
            """
            plt.tight_layout()
            plt.gcf().autofmt_xdate()
            #plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
            plt.show()
            
            
            ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred'])
            ax.fill_between( x=group_df["start_date"].values,y1=group_df['uci_pred_05'],y2=group_df['uci_pred_95'], facecolor='yellow', alpha=0.35)
            #ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='yellow', alpha=0.5)
            ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
            ax.grid()
            
            ax.xaxis_date()
            #ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='yellow', alpha=0.5)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            
            ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,3)), 
                   label="older adults",color="red", alpha=0.3)
            
            
            
            
            ax.axes.set_title("Timeseries UCI for "+group_name ,fontsize=15)
            ax.set_ylabel('N° beds', fontsize=10)
            ax.set_xlabel('Date', fontsize=10)
            
            plt.tight_layout()
            plt.gcf().autofmt_xdate()
            #plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
            plt.show()
            
            ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred'])
            ax.fill_between( x=group_df["start_date"].values,y1=group_df['uci_pred_05'],y2=group_df['uci_pred_95'], facecolor='blue', alpha=0.2)
            #ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='yellow', alpha=0.5)
            ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
            ax.grid()
            
            ax.xaxis_date()
            #ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='yellow', alpha=0.5)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            
            ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,3)), 
                   label="older adults",color="red", alpha=0.3)
            
            
            
            
            ax.axes.set_title("Timeseries UCI for "+group_name ,fontsize=15)
            ax.set_ylabel('N° beds', fontsize=10)
            ax.set_xlabel('Date', fontsize=10)
            
            plt.tight_layout()
            plt.gcf().autofmt_xdate()
            #plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
            plt.show()
            
    
    
class Camas_UCI_new_variant():
    def __init__(self,data,ID_grupo_etario,
                  increase_vac1_Pfizer,increase_vac2_Pfizer,
                  increase_vac1_Sinovac,increase_vac2_Sinovac,
                  prob_uci_vac1_Pfizer,prob_uci_vac2_Pfizer,
                  prob_uci_vac1_Sinovac,prob_uci_vac2_Sinovac,
                  prob_uci_vac1_b117_Pfizer,prob_uci_vac2_b117_Pfizer,
                  prob_uci_vac1_b117_Sinovac,prob_uci_vac2_b117_Sinovac,
                  prob_uci_vac1_new_variant_Pfizer,prob_uci_vac2_new_variant_Pfizer,
                  prob_uci_vac1_new_variant_Sinovac,prob_uci_vac2_new_variant_Sinovac,
                  p_shares_new_variant,
                  ID_list_day_new_variant,
                  p_shares_b117,
                  ID_list_day_b117,
                  probability_not_vac,
                  probability_not_vac_b117,
                  probability_not_vac_new_variant,
                  p_to_uci,
                  n_to_uci,
                  p_to_uci_2021,
                  n_to_uci_2021,
                  ID_day_change_time,
                  p_in_uci,
                  n_in_uci):
        """
        Input:
            data: dictionary with very useful information
            ID_grupo_etario: id of the age group to simulate
            probability_not_vac: probability of going to ICU
            p_to_uci: p is the probability of success  of the NB distribution for the time of going to ICU
            n_to_uci: n is the  the number of successes  of the NB distribution for the time of going to ICU
            p_in_uci: p is the probability of success  of the NB distribution for the time of in ICU
            n_in_uci: n is the  the number of successes  of the NB distribution for the time in ICU
        """
        self.data = data
        self.ID_grupo_etario= ID_grupo_etario
        
        self.increase_vac1_Pfizer=np.array(increase_vac1_Pfizer)
        self.increase_vac2_Pfizer=np.array(increase_vac2_Pfizer)
        self.increase_vac1_Sinovac=np.array(increase_vac1_Sinovac)
        self.increase_vac2_Sinovac=np.array(increase_vac2_Sinovac)
        
        self.prob_uci_vac1_Pfizer= np.array(prob_uci_vac1_Pfizer)
        self.prob_uci_vac2_Pfizer= np.array(prob_uci_vac2_Pfizer)
        
        self.prob_uci_vac1_Sinovac= np.array(prob_uci_vac1_Sinovac)
        self.prob_uci_vac2_Sinovac= np.array(prob_uci_vac2_Sinovac)
        
        self.prob_uci_vac1_b117_Pfizer=np.array(prob_uci_vac1_b117_Pfizer)
        self.prob_uci_vac2_b117_Pfizer=np.array(prob_uci_vac2_b117_Pfizer)
        
        self.prob_uci_vac1_b117_Sinovac= np.array(prob_uci_vac1_b117_Sinovac)
        self.prob_uci_vac2_b117_Sinovac= np.array(prob_uci_vac2_b117_Sinovac)
        
        
        self.prob_uci_vac1_new_variant_Pfizer= np.array(prob_uci_vac1_new_variant_Pfizer)
        self.prob_uci_vac2_new_variant_Pfizer= np.array(prob_uci_vac2_new_variant_Pfizer)
        
        self.prob_uci_vac1_new_variant_Sinovac= np.array(prob_uci_vac1_new_variant_Sinovac)
        self.prob_uci_vac2_new_variant_Sinovac= np.array(prob_uci_vac2_new_variant_Sinovac)
        
        self.probability_not_vac=probability_not_vac
        self.p_to_uci=p_to_uci 
        self.n_to_uci=n_to_uci
        
        self.p_to_uci_2021=p_to_uci_2021
        self.n_to_uci_2021=n_to_uci_2021
        self.ID_day_change_time=ID_day_change_time
        
        
        self.p_in_uci=p_in_uci 
        self.n_in_uci=n_in_uci
        self.probability_not_vac_b117=probability_not_vac_b117
        self.probability_not_vac_new_variant=probability_not_vac_new_variant
        
        
        self.p_shares_new_variant=p_shares_new_variant
        self.ID_list_day_new_variant=ID_list_day_new_variant
        self.p_shares_b117=p_shares_b117
        self.ID_list_day_b117=ID_list_day_b117
        
        self.firts_day_variant=min(ID_list_day_new_variant[0],ID_list_day_b117[0])
        inf = data['inf'].copy()
        ID_list_day_new_variant1=ID_list_day_new_variant.copy()
        ID_list_day_new_variant1.append(inf.shape[1]) #add the last day of inf
        ID_list_day_new_variant1.insert(0,0)
        p_shares_new_variant1=p_shares_new_variant.copy()
        p_shares_new_variant1.append(p_shares_new_variant1[-1])
        p_shares_new_variant1.insert(0,0)
        ID_list_day_new_variant1 = np.array(ID_list_day_new_variant1)
        p_shares_new_variant1 = np.array(p_shares_new_variant1)
        #ID_firts_day_new_variant=ID_list_day_new_variant1[0]
        day_step=np.diff(ID_list_day_new_variant1)
    
        ID_list_day_b1171=ID_list_day_b117.copy()
        ID_list_day_b1171.append(inf.shape[1]) #add the last day of inf
        ID_list_day_b1171.insert(0,0)
        p_shares_b1171=p_shares_b117.copy()
        p_shares_b1171.append(p_shares_b1171[-1])
        p_shares_b1171.insert(0,0)
        ID_list_day_b1171 = np.array(ID_list_day_b1171)
        p_shares_b1171 = np.array(p_shares_b1171)
        #ID_firts_day_b117=ID_list_day_b1171[0]
        day_step_b117=np.diff(ID_list_day_b1171)
        self.prob_new_variant=np.concatenate([np.linspace(p_shares_new_variant1[i],p_shares_new_variant1[i+1],day_step[i],endpoint=False) for i in range(day_step.shape[0])])
        self.prob_B117=np.concatenate([np.linspace(p_shares_b1171[i],p_shares_b1171[i+1],day_step[i],endpoint=False) for i in range(day_step.shape[0])])
        
        
        

    def prob_vacc(self,
                  increase_vac1_Pfizer,increase_vac2_Pfizer,
                  increase_vac1_Sinovac,increase_vac2_Sinovac,
                  prob_uci_vac1_Pfizer,prob_uci_vac2_Pfizer,
                  prob_uci_vac1_Sinovac,prob_uci_vac2_Sinovac,
                  probability_not_vac,date_to_dateID,T,window):
        """
        #prob_inf_by_vacc.shape == (T-W-1, 1+2(W))
        Probability of an infected person given the number of days he/she has been vaccinated. 

        prob_uci_by_vacc: ( 1+2(W))
        Probability of going to the ICU given the number of days vaccinated. 
        """
        sl_window = sliding_window_view(self.data['vac'][:,:,self.ID_grupo_etario,:],window, axis=-1).copy()
        #chage the las value t-w for acc vac in t-w
        
        print(sl_window.shape,sl_window[:,1,:,0].shape)
        vacc_acc=self.data['vac_acc'][:,:,self.ID_grupo_etario,:T-(window-1)]
        sl_window[:,1,:,0]=vacc_acc[:,1,:]
        
        #chage number of 1 dosis depending on the 2do dosis
        lab_to_labID=self.data['lab_to_labID']
        L= len(lab_to_labID)
        dateID_second_dosis = self.data['dateID_second_dosis']
        """
        for lab in range(L):
            for t in range(dateID_second_dosis,T):
                replace_value_for_1_vacc(self.data['vac_acc'][lab,1,self.ID_grupo_etario,t], sl_window[lab,0,t-(window-1),:], i=0)
                        
        """
        window_porc_vac_pop = np.multiply(sl_window,1/self.data['pop'][self.ID_grupo_etario])
        
        day_step=np.array([7,7,7,7,window-28])
        
        grilla_vac1_Pfizer=np.concatenate([
         np.linspace(increase_vac1_Pfizer[i],increase_vac1_Pfizer[i],day_step[i],endpoint=False) for i in range(increase_vac1_Pfizer.shape[0]-1)],
         axis=None).tolist()[::-1]
        grilla_vac2_Pfizer=np.concatenate([
         np.linspace(increase_vac2_Pfizer[i],increase_vac2_Pfizer[i],day_step[i],endpoint=False) for i in range(increase_vac2_Pfizer.shape[0]-1)],
         axis=None).tolist()[::-1]
        
        grilla_vac1_Sinovac=np.concatenate([
         np.linspace(increase_vac1_Sinovac[i],increase_vac1_Sinovac[i],day_step[i],endpoint=False) for i in range(increase_vac1_Sinovac.shape[0]-1)],
         axis=None).tolist()[::-1]
        grilla_vac2_Sinovac=np.concatenate([
         np.linspace(increase_vac2_Sinovac[i],increase_vac2_Sinovac[i],day_step[i],endpoint=False) for i in range(increase_vac2_Sinovac.shape[0]-1)],
         axis=None).tolist()[::-1]
        #dado que W=0 es el t-W se deven dar vuelta los valores
        grilla = np.array([[grilla_vac1_Pfizer,grilla_vac2_Pfizer],[grilla_vac1_Sinovac,grilla_vac2_Sinovac]]) #(L,D,W)
        
        numerador= np.multiply(window_porc_vac_pop,np.expand_dims(1-grilla,axis=(-2)))#(L,D,T-(W-1),W) *(L,D,1,W)-->(L,D,T-(W-1),W)
        #primero la curva Pfizer, despues Snovac
        aux = np.concatenate((numerador[0,0],numerador[0,1],numerador[1,0],numerador[1,1]),axis=-1) #flat dosis dimensions(L,D,T-(W-1),W)-->(T-(W-1),W*L*D)
        
        #call porc not vac
        porc_not_vac_pop = np.expand_dims(np.multiply(self.data['not_vac'],1/np.expand_dims(self.data['pop'],axis=(-1)))[self.ID_grupo_etario,window-1:],axis=(-1))
        numerador=np.concatenate((porc_not_vac_pop,aux),axis=1)#(T-(W-1),1)+(T-(W-1),W*L*D)  ---->(T-(W-1),W*L*D+1)
        denominador= np.sum(numerador,axis=(-1))#(T-(W-1),W*L*D+1) ---->(T-(W-1))
        prob_inf_by_vacc = np.multiply(numerador, np.expand_dims(1/denominador,axis=(-1)))#(T-(W-1),W*L*D+1)/(T-(W-1),1) ---->(T-(W-1),W*L*D+1)
        #print(denominador.shape)
        #print(prob_inf_by_vacc.shape)
        
        grilla_vac1_Pfizer=np.concatenate([
                 np.linspace(prob_uci_vac1_Pfizer[i],prob_uci_vac1_Pfizer[i],day_step[i],endpoint=False) for i in
                  range(prob_uci_vac1_Pfizer.shape[0]-1)],axis=None).tolist()[::-1]
        grilla_vac2_Pfizer=np.concatenate([
                 np.linspace(prob_uci_vac2_Pfizer[i],prob_uci_vac2_Pfizer[i],day_step[i],endpoint=False) for i in
                  range(prob_uci_vac2_Pfizer.shape[0]-1)],axis=None).tolist()[::-1]
        grilla_vac1_Sinovac=np.concatenate([
                 np.linspace(prob_uci_vac1_Sinovac[i],prob_uci_vac1_Sinovac[i],day_step[i],endpoint=False) for i in
                  range(prob_uci_vac1_Sinovac.shape[0]-1)],axis=None).tolist()[::-1]
        grilla_vac2_Sinovac=np.concatenate([
                 np.linspace(prob_uci_vac2_Sinovac[i],prob_uci_vac2_Sinovac[i],day_step[i],endpoint=False) for i in
                  range(prob_uci_vac2_Sinovac.shape[0]-1)],axis=None).tolist()[::-1]
        probability_not_vac=np.array([probability_not_vac])
        
        self.grilla_vac1_Pfizer=grilla_vac1_Pfizer
        self.grilla_vac2_Pfizer=grilla_vac2_Pfizer
        self.grilla_vac1_Sinovac=grilla_vac1_Sinovac
        self.grilla_vac2_Sinovac=grilla_vac2_Sinovac
        
        aux = np.concatenate((grilla_vac1_Pfizer,grilla_vac2_Pfizer,grilla_vac1_Sinovac,grilla_vac2_Sinovac),axis=-1)
        prob_uci_by_vacc= np.concatenate((probability_not_vac,aux),axis=-1)
        
        
        return prob_inf_by_vacc, prob_uci_by_vacc

    def prob_new_variant_by_day(self,replicas,inf_today,prob_new_variant ):
        
        """
        Parameters
        ----------
        replicas : int
            number of replicas/simulations to perform
        inf_today : int
            number of infected for a specific day

        Returns
        -------
        Sparce Matrix (replicas, inf_today)
            view result plt.spy(A,markersize=1)
            usar sparce matrix en realidad no tira numeros aleatorios por matriz
            #sps.random(replicas, inf_today, density=self.probability_not_vac, data_rvs=np.ones)
        """
        M=np.random.binomial(size=(replicas,inf_today),n=1,p=prob_new_variant)
        
        return M
    def prob_new_variant_by_day_for_vacc(self,replicas,inf_today,list_prob_day ):
        
        """
        Parameters
        ----------
        replicas : int
            number of replicas/simulations to perform
        inf_today : int
            number of infected for a specific day

        Returns
        -------
        Sparce Matrix (replicas, inf_today)
            view result plt.spy(A,markersize=1)
            usar sparce matrix en realidad no tira numeros aleatorios por matriz
            #sps.random(replicas, inf_today, density=self.probability_not_vac, data_rvs=np.ones)
        """
        
        xk = np.arange(3)
        pk=list_prob_day
        custm = stats.rv_discrete(name='custm', values=(xk, pk))
        aux_M= custm.rvs(size=(replicas,inf_today))
        
        inf_without_variant=np.where(aux_M==0,1,0)
        inf_new_variant=np.where(aux_M==1,1,0)
        inf_B117=np.where(aux_M==2,1,0)
        #print(list_prob_day)
        if bernoulli.rvs(p=0.5, size=1)[0]==1:
            print("For a random day: ")
            print([round(num,3) for num in list_prob_day]
                  ,np.histogram(aux_M, bins=np.arange(4),density=True)[0].round(3))
            #print(replicas*inf_today,inf_without_variant.sum(),inf_new_variant.sum(), inf_B117.sum())
        
        
        
        return inf_without_variant,inf_new_variant,inf_B117
    
        
    def people_going_to_uci_by_day(self,replicas,inf_today,prob_inf_by_vacc_and_day, prob_uci_by_vacc,infected_by_variant,case="base_case"):
        """
        Parameters
        ----------
        replicas : int
            number of replicas/simulations to perform
        inf_today : int
            number of infected for a specific day
        prob_inf_by_vacc_and_day: array
            Probability of an infected person given the number of days he/she has been vaccinated.
        
        prob_uci_by_vacc: array
            Probability of going to the ICU given the number of days vaccinated.
        infected_by_variant: matrix 
          if those infected today are infected by this variant 
            
        
        Returns
        -------
        Sparce Matrix (replicas, inf_today)
            view result plt.spy(A,markersize=1)
        """
        
        if case == "base_case":
            M=np.random.binomial(size=(replicas,inf_today),n=1,p=self.probability_not_vac)
            rows = np.where(M==1)[0].tolist()
            cols = np.where(M==1)[1].tolist()
            ones = np.ones(len(rows), np.uint32)
            S = sps.coo_matrix((ones, (rows, cols)), shape=(replicas,inf_today))
            return S

        if case == "vacc_variant_case":
            
            xk = np.arange(prob_uci_by_vacc.shape[0])
            pk=prob_inf_by_vacc_and_day
            #print(prob_inf_by_vacc_and_day,sum(pk))
            custm = stats.rv_discrete(name='custm', values=(xk, pk))
            aux_M= custm.rvs(size=(replicas,inf_today))
            
            aux1_M=np.array([[np.random.binomial(size=1,n=1,p=prob_uci_by_vacc[aux_M[i,j]])[0] for j in range(inf_today)] for i in range(replicas)])
            #aux1_M=np.array([[ bernoulli.rvs(size=1,p=prob_uci_by_vacc[aux_M[i,j]])[0] for j in range(inf_today)] for i in range(replicas)])
            M=np.multiply(aux1_M,infected_by_variant)
            
            rows = np.where(M==1)[0].tolist()
            cols = np.where(M==1)[1].tolist()
            ones = np.ones(len(rows), np.uint32)
            S = sps.coo_matrix((ones, (rows, cols)), shape=(replicas,inf_today))#.tocsr()
    
            return S
        
    
    def people_time_go_to_uci_by_day(self,replicas,inf_today,p_to_uci,n_to_uci,max_days_go_uci=30):
        """
        Parameters
        ----------
        replicas : int
            number of replicas/simulations to perform
        inf_today : int
            number of infected for a specific day

        **ponderaciones_to_uci: probability that distribution is geometric

        Returns
        -------
        Sparce Matrix (replicas, inf_today)
            view result plt.spy(A,markersize=1)
            
        """
        #np.random.binomial(size=(replicas,inf_today),n=1,p=self.probability_not_vac)
        
        nb_matrix=np.random.negative_binomial(size=(replicas,inf_today),n=n_to_uci,p=p_to_uci)
               
        while len(np.where(nb_matrix>= max_days_go_uci)[0])!=0:
            rows,cols=np.where(nb_matrix>= max_days_go_uci)
            nums =np.random.negative_binomial(size=len(rows),n=n_to_uci,p=p_to_uci)
            for i,j,num in zip(rows,cols,nums):
                nb_matrix[i,j]=num
                
        return nb_matrix
        
    
    def people_time_in_uci_by_day(self,replicas,inf_today,p_in_uci,n_in_uci,max_days_in_uci= 100):
        """
        Parameters
        ----------
        replicas : int
            number of replicas/simulations to perform
        inf_today : int
            number of infected for a specific day

        **ponderaciones_to_uci: probability that distribution is geometric

        Returns
        -------
        Sparce Matrix (replicas, inf_today)
            view result plt.spy(A,markersize=1)
        """
        
        nb_matrix=np.random.negative_binomial(size=(replicas,inf_today),n=n_in_uci,p=p_in_uci)
               
        while len(np.where(nb_matrix>= max_days_in_uci)[0])!=0:
            rows,cols=np.where(nb_matrix>= max_days_in_uci)
            nums =np.random.negative_binomial(size=len(rows),n=n_in_uci,p=p_in_uci)
            for i,j,num in zip(rows,cols,nums):
                nb_matrix[i,j]=num

        return nb_matrix
    
    #@classmethod
    def ICU_Simulations_camas_vac(self,replicas,confidence,error, start_date='2020-07-20',end_date='2020-04-01',max_days_go_uci=30,
                  max_days_in_uci= 100,max_days_go_uci_b117=30,
                  max_days_in_uci_b117= 100,max_days_go_uci_new_variant=30,
                  max_days_in_uci_new_variant= 100,window=29, shift= False):

        #aux
        dateID_to_date = self.data['dateID_to_date']
        date_to_dateID = self.data["date_to_dateID"]
        dateID_start_date = self.data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]-(window-1)
        dateID_end_date = self.data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]-(window-1)
        T = len(date_to_dateID)
        
        #call probability matrix
        prob_inf_by_vacc, prob_uci_by_vacc=self.prob_vacc(self.increase_vac1_Pfizer,self.increase_vac2_Pfizer,
                                                          self.increase_vac1_Sinovac,self.increase_vac2_Sinovac,
                                                          self.prob_uci_vac1_Pfizer,self.prob_uci_vac2_Pfizer,
                                                          self.prob_uci_vac1_Sinovac,self.prob_uci_vac2_Sinovac,
                                                          self.probability_not_vac,date_to_dateID,T,window)
        
        prob_inf_by_vacc_new_variant, prob_uci_by_vacc_new_variant=self.prob_vacc(self.increase_vac1_Pfizer,self.increase_vac2_Pfizer,
                                                                                  self.increase_vac1_Sinovac,self.increase_vac2_Sinovac,
                                                                                  self.prob_uci_vac1_new_variant_Pfizer,self.prob_uci_vac2_new_variant_Pfizer,
                                                                                  self.prob_uci_vac1_new_variant_Sinovac,self.prob_uci_vac2_new_variant_Sinovac,
                                                                                  self.probability_not_vac_new_variant,date_to_dateID,T,window)
        
        prob_inf_by_vacc_b117, prob_uci_by_vacc_b117=self.prob_vacc(self.increase_vac1_Pfizer,self.increase_vac2_Pfizer,
                                                                    self.increase_vac1_Sinovac,self.increase_vac2_Sinovac,
                                                                    self.prob_uci_vac1_b117_Pfizer,self.prob_uci_vac2_b117_Pfizer,
                                                                    self.prob_uci_vac1_b117_Sinovac,self.prob_uci_vac2_b117_Sinovac,
                                                                    self.probability_not_vac_b117,date_to_dateID,T,window)
        
        
        self.prob_inf_by_vacc=prob_inf_by_vacc
        self.prob_inf_by_vacc_new_variant=prob_inf_by_vacc_new_variant
        self.prob_inf_by_vacc_b117=prob_inf_by_vacc_b117
        
        self.prob_uci_by_vacc=prob_uci_by_vacc
        self.prob_uci_by_vacc_new_variant=prob_uci_by_vacc_new_variant
        self.prob_uci_by_vacc_b117=prob_uci_by_vacc_b117
        
        
        inf=self.data['inf'][self.ID_grupo_etario,:].copy()
        
        if shift:
            
            shif_curve=[8, 7, 8, 7, 4]
            item=shif_curve[self.ID_grupo_etario]
            inf[0:-(item-1)]=self.data['inf'][self.ID_grupo_etario,item-1:]
        inf=inf[window-1:]
        
        
        
        
        uci_real=self.data['uci'][self.ID_grupo_etario,window-1:]
        uci_beds= np.zeros((replicas,inf.shape[0]))
        prob_new_variant= self.prob_new_variant[window-1:]
        prob_B117= self.prob_B117[window-1:]
        firts_day_variant=self.firts_day_variant-(window-1)
        print(firts_day_variant)
        
        ratio_b117_list=[]
        ms_b117_list=[]
        
        for t in range(inf.shape[0]):
            inf_today= int(inf[t])
            
            #time to uci 2020
            if t<self.ID_day_change_time-(window-1):
                p_to_uci,n_to_uci=self.p_to_uci,self.n_to_uci
            
            
            elif t>=self.ID_day_change_time-(window-1):
                p_to_uci,n_to_uci=self.p_to_uci_2021,self.n_to_uci_2021
            
            
            
            
            if t<firts_day_variant:
                people_going_to_uci_without_variant=self.people_going_to_uci_by_day(replicas,inf_today,None, None,None,case="base_case")
                #print(inf_today,self.p_to_uci,self.n_to_uci)
                time_go_to_uci=self.people_time_go_to_uci_by_day(replicas,inf_today,p_to_uci,n_to_uci,max_days_go_uci=max_days_go_uci) 
                time_in_uci=self.people_time_in_uci_by_day(replicas,inf_today,self.p_in_uci,self.n_in_uci,max_days_in_uci= max_days_in_uci)
                for i,j,v in zip(people_going_to_uci_without_variant.row, people_going_to_uci_without_variant.col, people_going_to_uci_without_variant.data):
                    go_to_uci=int(time_go_to_uci[i,j])
                    in_uci=int(time_in_uci[i,j])
                    if (in_uci>0)&(t+go_to_uci<inf.shape[0]):
                        uci_beds[i,t+go_to_uci:min(t+go_to_uci+in_uci,inf.shape[0])]+=1
                        
                        
            if t>=firts_day_variant: 
                print("Day {} of {}".format(t,T-(window-1)))
                inf_without_variant,inf_new_variant,inf_B117= self.prob_new_variant_by_day_for_vacc(replicas,inf_today,[1-prob_new_variant[t]-prob_B117[t],prob_new_variant[t],prob_B117[t]])
                
                
                print("N° of firts rep not variant {}, variant {}, b117 {}".format(inf_without_variant[0].sum(),inf_new_variant[0].sum(),inf_B117[0].sum()))
                
                
                people_going_to_uci_without_variant=self.people_going_to_uci_by_day(replicas,inf_today,prob_inf_by_vacc[t], prob_uci_by_vacc,inf_without_variant,case="vacc_variant_case")
                
                k=0
                for i,j,v in zip(people_going_to_uci_without_variant.row, people_going_to_uci_without_variant.col, people_going_to_uci_without_variant.data):
                    #go_to_uci=int(time_go_to_uci_without_variant[i,j])
                    #in_uci=int(time_in_uci_without_variant[i,j])
                    go_to_uci=int(self.people_time_go_to_uci_by_day(1,1,p_to_uci,n_to_uci,max_days_go_uci=max_days_go_uci))
                    in_uci=int(self.people_time_in_uci_by_day(1,1,self.p_in_uci,self.n_in_uci,max_days_in_uci=max_days_in_uci))
                    
                    k+=1
                    if (in_uci>0)&(t+go_to_uci<inf.shape[0]):
                        uci_beds[i,t+go_to_uci:min(t+go_to_uci+in_uci,inf.shape[0])]+=1
                print("For not variant we have {} infections , infections today {}, porc {} %".format(k,inf_today,round(k/(inf_today*replicas)*100,3)))
                
                
                ponderador=k
            
                
                if prob_new_variant[t]>0:
                    people_going_to_uci_new_variant=self.people_going_to_uci_by_day(replicas,inf_today,prob_inf_by_vacc_new_variant[t], prob_uci_by_vacc_new_variant,inf_new_variant,case="vacc_variant_case")
                    try:
                        k=0
                        for i,j,v in zip(people_going_to_uci_new_variant.row, people_going_to_uci_new_variant.col, people_going_to_uci_new_variant.data):
                            go_to_uci=int(self.people_time_go_to_uci_by_day(1,1,p_to_uci,n_to_uci,max_days_go_uci=max_days_go_uci_new_variant))
                            in_uci=int(self.people_time_in_uci_by_day(1,1,self.p_in_uci,self.n_in_uci,max_days_in_uci=max_days_in_uci_new_variant))
                            
                            k+=1
                            if (in_uci>0)&(t+go_to_uci<inf.shape[0]):
                                uci_beds[i,t+go_to_uci:min(t+go_to_uci+in_uci,inf.shape[0])]+=1
                        print("For new variant we have {} infections , infections today {}, porc {} %".format(k,inf_today,round(k/(inf_today*replicas)*100,3)))
                        if (1-prob_new_variant[t]-prob_B117[t])/prob_new_variant[t]>0:
                        
                        
                            print("Ratio new variant: {}".format(round(k*(1-prob_new_variant[t]-prob_B117[t])/(prob_new_variant[t]*ponderador),4)))
                    except:
                        pass
                if prob_B117[t]>0:
                    people_going_to_uci_B117=self.people_going_to_uci_by_day(replicas,inf_today,prob_inf_by_vacc_b117[t], prob_uci_by_vacc_b117,inf_B117,case="vacc_variant_case")
                    try:
                        k=0
                        for i,j,v in zip(people_going_to_uci_B117.row, people_going_to_uci_B117.col, people_going_to_uci_B117.data):
                            go_to_uci=int(self.people_time_go_to_uci_by_day(1,1,p_to_uci, n_to_uci,max_days_go_uci=max_days_go_uci_b117))
                            in_uci=int(self.people_time_in_uci_by_day(1,1,self.p_in_uci,self.n_in_uci,max_days_in_uci=max_days_in_uci_b117))
                            k+=1
                            if (in_uci>0)&(t+go_to_uci<inf.shape[0]):
                                uci_beds[i,t+go_to_uci:min(t+go_to_uci+in_uci,inf.shape[0])]+=1
                        print("For B117 we have {} infections , infections today {}, porc {} %".format(k,inf_today,round(k/(inf_today*replicas)*100,3)))
                        if ((1-prob_new_variant[t]-prob_B117[t])/prob_new_variant[t])>0:
                        
                        
                            print("Ratio b117: {}".format(round(k*(1-prob_new_variant[t]-prob_B117[t])/(prob_B117[t]*ponderador),4)))
                            
                            ratio_b117_list.append(round(k*(1-prob_new_variant[t]-prob_B117[t])/(prob_B117[t]*ponderador),4))
                            ms_b117_list.append([round(k*(1-prob_new_variant[t]-prob_B117[t])/(2.03*ponderador),4),round(prob_B117[t],4)])
                            
                            
                            
                    except:
                        pass
                
        self.ratio_b117_list=np.array(ratio_b117_list)
        self.ms_b117_list=np.array(ms_b117_list)
        self.uci_beds = uci_beds
        """
        people_going_to_uci =np.array([self.people_going_to_uci_by_day(replicas,inf_today) for inf_today in inf])
        time_go_to_uci=np.array([self.people_time_go_to_uci_by_day(replicas,inf_today) for inf_today in inf])
        time_in_uci=np.array([self.people_time_in_uci_by_day(replicas,inf_today) for inf_today in inf])
        """
        
        #Calculamos n
        z_score = stats.t.ppf((1 + confidence) / 2., len(self.uci_beds)-1)
        s = np.std(self.uci_beds, ddof=1,axis=0)
        n = np.max(s*(z_score/error)**2)
        #Reportamos resultados parciales
        print("Resultados tras {} replicas es:".format(replicas))
        """
        for date in range(dateID_start_date,dateID_end_date):
            intervalo_confianza(uci_beds[:,date],uci_real[date], confidence=confidence, print_to_console=1)
        """
        print("En total se necesitan {} replicas. Ejecutando:".format(int(n)))
        #run the results that are needed
        """
        
        """
        """
        if (int(n)-replicas)>0:
            replicas=int(n) 
            uci_beds_2= np.zeros((replicas,inf.shape[0]))
            for t in range(inf.shape[0]):
                
                #time to uci 2020
                if t<self.ID_day_change_time-(window-1):
                    p_to_uci,n_to_uci=self.p_to_uci,self.n_to_uci
                
                
                elif t>=self.ID_day_change_time-(window-1):
                    p_to_uci,n_to_uci=self.p_to_uci_2021,self.n_to_uci_2021
                    
                
                
                inf_today= int(inf[t])
                if t<firts_day_variant:
                    people_going_to_uci_without_variant=self.people_going_to_uci_by_day(replicas,inf_today,None, None,None,case="base_case")
                    time_go_to_uci=self.people_time_go_to_uci_by_day(replicas,inf_today,p_to_uci,n_to_uci,max_days_go_uci=max_days_go_uci) 

                    time_in_uci=self.people_time_in_uci_by_day(replicas,inf_today,self.p_in_uci,self.n_in_uci,max_days_in_uci= max_days_in_uci)
                    for i,j,v in zip(people_going_to_uci_without_variant.row, people_going_to_uci_without_variant.col, people_going_to_uci_without_variant.data):
                        go_to_uci=int(time_go_to_uci[i,j])
                        in_uci=int(time_in_uci[i,j])
                        if (in_uci>0)&(t+go_to_uci<inf.shape[0]):
                            uci_beds_2[i,t+go_to_uci:min(t+go_to_uci+in_uci,inf.shape[0])]+=1
                            
                if t>=firts_day_variant: 
                    
                    inf_without_variant= self.prob_new_variant_by_day(replicas,inf_today,1-prob_new_variant[t]-prob_B117[t])
                    people_going_to_uci_without_variant=self.people_going_to_uci_by_day(replicas,inf_today,prob_inf_by_vacc[t], prob_uci_by_vacc,inf_without_variant,case="vacc_variant_case")
                    
                    k=0
                    for i,j,v in zip(people_going_to_uci_without_variant.row, people_going_to_uci_without_variant.col, people_going_to_uci_without_variant.data):
                        #go_to_uci=int(time_go_to_uci_without_variant[i,j])
                        #in_uci=int(time_in_uci_without_variant[i,j])
                        go_to_uci=int(self.people_time_go_to_uci_by_day(1,1,p_to_uci,n_to_uci,max_days_go_uci=max_days_go_uci))
                        in_uci=int(self.people_time_in_uci_by_day(1,1,self.p_in_uci,self.n_in_uci,max_days_in_uci=max_days_in_uci_new_variant))
                        
                        k+=1
                        if (in_uci>0)&(t+go_to_uci<inf.shape[0]):
                            uci_beds_2[i,t+go_to_uci:min(t+go_to_uci+in_uci,inf.shape[0])]+=1
                    print(k,inf_today,k/(inf_today*replicas))

                    inf_new_variant=self.prob_new_variant_by_day(replicas,inf_today,prob_new_variant[t])
                    people_going_to_uci_new_variant=self.people_going_to_uci_by_day(replicas,inf_today,prob_inf_by_vacc_new_variant[t], prob_uci_by_vacc_new_variant,inf_new_variant,case="vacc_variant_case")
                    try:
                        for i,j,v in zip(people_going_to_uci_new_variant.row, people_going_to_uci_new_variant.col, people_going_to_uci_new_variant.data):
                            go_to_uci=int(self.people_time_go_to_uci_by_day(1,1,p_to_uci,n_to_uci,max_days_go_uci=max_days_go_uci_b117))
                            in_uci=int(self.people_time_in_uci_by_day(1,1,self.p_in_uci,self.n_in_uci,max_days_in_uci=max_days_in_uci_b117))
                            if (in_uci>0)&(t+go_to_uci<inf.shape[0]):
                                uci_beds_2[i,t+go_to_uci:min(t+go_to_uci+in_uci,inf.shape[0])]+=1
                    except:
                        pass
                    
                    inf_B117=self.prob_new_variant_by_day(replicas,inf_today,prob_B117[t])
                    people_going_to_uci_B117=self.people_going_to_uci_by_day(replicas,inf_today,prob_inf_by_vacc_b117[t], prob_uci_by_vacc_b117,inf_B117,case="vacc_variant_case")
                    try:
                        for i,j,v in zip(people_going_to_uci_B117.row, people_going_to_uci_B117.col, people_going_to_uci_B117.data):
                            go_to_uci=int(self.people_time_go_to_uci_by_day(1,1,p_to_uci,n_to_uci,max_days_go_uci=max_days_go_uci_b117))
                            in_uci=int(self.people_time_in_uci_by_day(1,1,self.p_in_uci,self.n_in_uci,max_days_in_uci=max_days_in_uci_b117))
                            if (in_uci>0)&(t+go_to_uci<inf.shape[0]):
                                uci_beds_2[i,t+go_to_uci:min(t+go_to_uci+in_uci,inf.shape[0])]+=1
                    except:
                        pass
                    
            
            self.uci_beds = np.concatenate((uci_beds,uci_beds_2),axis=0)
            """
        #Reportamos resultados
        mean_reasult=[]
        for date in range(dateID_start_date,dateID_end_date):
             prom, desv, li, ls =intervalo_confianza(self.uci_beds[:,date],uci_real[date], confidence=confidence, print_to_console=0)
             mean_reasult.append([prom, desv, li, ls])
        
        self.mean_reasult=mean_reasult
        return mean_reasult
    
    
    
    
    
    def plot_uci_pred_2020(self,data, uci, W=29, pond=1, start_date='2020-07-20',end_date='2020-12-01', infected=False):
        groupID_to_group = data['groupID_to_group']
        dateID_to_date = data['dateID_to_date']
        date_to_dateID = data["date_to_dateID"]
        #start_date='2020-10-01'  
        uci_pred_05=np.percentile(self.uci_beds, 5,axis=0)
        uci_pred_95=np.percentile(self.uci_beds, 95,axis=0)
        uci_pred_50=np.percentile(self.uci_beds, 50,axis=0)
        uci_pred_mean=np.mean(self.uci_beds, axis=0,dtype=np.float64)
        dateID_start_date = data['date_to_dateID'][np.datetime64(start_date+"T00:00:00.000000000")]
        dateID_end_date = data['date_to_dateID'][np.datetime64(end_date+"T00:00:00.000000000")]
        cols =['Grupo de edad', 'start_date','inf', 'uci_real', 'uci_pred','uci_pred_05','uci_pred_95','uci_pred_50','uci_pred_mean']
        lst = []
        g=self.ID_grupo_etario
        for date in range(dateID_start_date,dateID_end_date): 
            info = [groupID_to_group[g],dateID_to_date[date],data['inf'][g,date]*pond,data['uci'][g,date]]
            info.append(uci[g,date-(W-1)])
            info.append(uci_pred_05[date-(W-1)])
            info.append(uci_pred_95[date-(W-1)])
            info.append(uci_pred_50[date-(W-1)])
            info.append(uci_pred_mean[date-(W-1)])
            lst.append(info)
        df_res = pd.DataFrame(lst, columns=cols)
        df_res["start_date"] = pd.to_datetime(df_res.start_date)
        for i, (group_name, group_df) in enumerate(df_res.groupby(["Grupo de edad"])):
        #for i in range(2):
        
            #plt.subplots(figsize=(15, 9))
            # plot the group on these axes
            if group_name=='>=70':
                ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred','inf','uci_pred_05','uci_pred_95'])
                
            else:
                if infected:
                    ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred','inf','uci_pred_05','uci_pred_95'])
                else:
                    ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred','uci_pred_05','uci_pred_95'])
            # set the title
            ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
            ax.grid()
            
            ax.xaxis_date()
            """
            tks = 30
            #locator = mdates.AutoDateLocator(minticks=tks, maxticks=tks)
            locator = mdates.DateFormatter('%Y-%m-%d')
            formatter = mdates.ConciseDateFormatter(locator)
            #ax.xaxis.set_major_locator(locator)
            #ax.xaxis.set_major_formatter(formatter)
            """
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            
            ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,3)), 
                   label="older adults",color="red", alpha=0.3)        
            ax.axes.set_title("Timeseries UCI for "+group_name ,fontsize=15)
            ax.set_ylabel('N° beds', fontsize=10)
            ax.set_xlabel('Date', fontsize=10)
            """
            try:
                plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name) +'.png')
            except:
                plt.savefig('image/UCI/Timeseries_UCI_'+str(group_name)[2:] +'.png')
            """
            plt.tight_layout()
            plt.gcf().autofmt_xdate()
            #plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
            plt.show()
            
            
            ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred'])
            ax.fill_between( x=group_df["start_date"].values,y1=group_df['uci_pred_05'],y2=group_df['uci_pred_95'], facecolor='yellow', alpha=0.35)
            #ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='yellow', alpha=0.5)
            ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
            ax.grid()
            
            ax.xaxis_date()
            #ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='yellow', alpha=0.5)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            
            ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,3)), 
                   label="older adults",color="red", alpha=0.3)
            
            
            
            
            ax.axes.set_title("Timeseries UCI for "+group_name ,fontsize=15)
            ax.set_ylabel('N° beds', fontsize=10)
            ax.set_xlabel('Date', fontsize=10)
            
            plt.tight_layout()
            plt.gcf().autofmt_xdate()
            #plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
            plt.show()
            
            ax=group_df.plot( kind='line', x="start_date", y=['uci_real', 'uci_pred'])
            ax.fill_between( x=group_df["start_date"].values,y1=group_df['uci_pred_05'],y2=group_df['uci_pred_95'], facecolor='blue', alpha=0.2)
            #ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='yellow', alpha=0.5)
            ax.set_xlim([group_df["start_date"].min()+pd.DateOffset(-2),group_df["start_date"].max()+pd.DateOffset(2)])
            ax.grid()
            
            ax.xaxis_date()
            #ax.fill_between(t, mu2+sigma2, mu2-sigma2, facecolor='yellow', alpha=0.5)
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            
            
            ax.axvspan(date2num(datetime(2021,1,1)), date2num(datetime(2021,1,3)), 
                   label="older adults",color="red", alpha=0.3)
            
            
            
            
            ax.axes.set_title("Timeseries UCI for "+group_name ,fontsize=15)
            ax.set_ylabel('N° beds', fontsize=10)
            ax.set_xlabel('Date', fontsize=10)
            
            plt.tight_layout()
            plt.gcf().autofmt_xdate()
            #plt.setp(ax.get_xticklabels(), rotation=90, horizontalalignment='left')
            plt.show()
        


def replace_value_for_1_vacc(vac_acc_2jt, split_array, i=0):
    
    """
    
    
    """
    diff = split_array[i]-vac_acc_2jt
    if diff >=0:
        split_array[i]=diff
        pass
    
    else:
        split_array[i]=0
        replace_value_for_1_vacc(abs(diff), split_array, i=i+1)
"""
if __name__== "__main__" :
    
    #data = read_data()



    list_p_to_uci=[0.463, 0.2139, 0.1494, 0.3976, 0.2466]
    list_n_to_uci= [8.8889, 2.6667, 2.2222, 5.1111, 3.6667]
    list_p_to_uci_2021=[0.699, 0.402, 0.365, 0.508, 0.7  ]
    list_n_to_uci_2021= [21.798 ,  6.2727,  6.5657,  9.202 , 22.0909]
    ID_day_change_time=275
    
    list_p_in_uci=[0.0699, 0.0619, 0.0515, 0.0739, 0.0412]
    list_n_in_uci=[1.9495, 1.4646, 1.4444, 2.0101, 1.303]

    p_shares_new_variant=[[0,0, 0.0, 0.269, 0.5, 0.739, 0.643],
                        [0,0, 0.0, 0.357, 0.517, 0.522, 0.833],
                        [0,0, 0.0, 0.462, 0.55, 0.818, 0.818],
                        [0,0, 0.057, 0.418, 0.465, 0.636, 0.62],
                        [0,0, 0.0, 0.333, 0.333, 0.571, 0.75]]
    ID_list_day_new_variant=[260,289,320,348,379,409,440]
    p_shares_b117=[[0,0, 0.062, 0.115, 0.096, 0.0, 0.0],
                  [0,0, 0.0, 0.0, 0.069, 0.0, 0.0],
                  [0,0, 0.0, 0.308, 0.1, 0.0, 0.0],
                  [0,0, 0.057, 0.114, 0.088, 0.029, 0.014],
                  [0,0, 0.0, 0.0, 0.167, 0.0, 0.0]]
    ID_list_day_b117=[260,289,320,348,379,409,440]
  
    
    
    CONFIDENCE = 0.95     
    ERROR = 5
    N_INICIAL = 100          
    
    print("Nivel de confianza	: {}".format(CONFIDENCE))
    print("Error			: {}".format(ERROR))
    print("Replicas Iniciales	: {}".format(N_INICIAL))			 
    print("--------------------")
    ID_grupo_etario=0;
    p_to_uci=list_p_to_uci[ID_grupo_etario];
    n_to_uci=list_n_to_uci[ID_grupo_etario];
    p_to_uci_2021=list_p_to_uci_2021[ID_grupo_etario];
    n_to_uci_2021=list_n_to_uci_2021[ID_grupo_etario];
    ID_day_change_time;
    p_in_uci=list_p_in_uci[ID_grupo_etario];
    n_in_uci=list_n_in_uci[ID_grupo_etario];
    shift=True; 
    ID_list_day_new_variant=[260,289,320,348,379,409,440];
    ID_list_day_b117=[260,289,320,348,379,409,440];
    camas_uci_group_0=  Camas_UCI_new_variant(data, 
                                  ID_grupo_etario, 
                                  increase_vac1_Pfizer[ID_grupo_etario],increase_vac2_Pfizer[ID_grupo_etario],
                                  increase_vac1_Sinovac[ID_grupo_etario],increase_vac2_Sinovac[ID_grupo_etario],
                                  prob_uci_vac1_Pfizer[ID_grupo_etario],prob_uci_vac2_Pfizer[ID_grupo_etario],
                                  prob_uci_vac1_Sinovac[ID_grupo_etario],prob_uci_vac2_Sinovac[ID_grupo_etario],
                                  prob_uci_vac1_b117_Pfizer[ID_grupo_etario],prob_uci_vac2_b117_Pfizer[ID_grupo_etario],
                                  prob_uci_vac1_b117_Sinovac[ID_grupo_etario],prob_uci_vac2_b117_Sinovac[ID_grupo_etario],
                                  prob_uci_vac1_new_variant_Pfizer[ID_grupo_etario],prob_uci_vac2_new_variant_Pfizer[ID_grupo_etario],
                                  prob_uci_vac1_new_variant_Sinovac[ID_grupo_etario],prob_uci_vac2_new_variant_Sinovac[ID_grupo_etario],
                                  p_shares_new_variant[ID_grupo_etario],
                                  ID_list_day_new_variant,
                                  p_shares_b117[ID_grupo_etario],
                                  ID_list_day_b117,
                                  probability_not_vac[ID_grupo_etario],
                                  probability_not_vac_b117[ID_grupo_etario],
                                  probability_not_vac_new_variant[ID_grupo_etario],
                                  p_to_uci,
                                  n_to_uci,
                                  p_to_uci_2021,
                                  n_to_uci_2021,
                                  ID_day_change_time,
                                  p_in_uci,
                                  n_in_uci)
    mean_reasult_group_0=camas_uci_group_0.ICU_Simulations_camas_vac(N_INICIAL,
                                                                      CONFIDENCE,
                                                                      ERROR,
                                                                      start_date='2020-07-20',
                                                                      end_date='2021-04-19',shift= shift)
    camas_uci_group_0.plot_uci_pred_2020(data, uci, W=29, pond=1, start_date='2020-07-20',end_date='2021-05-15', infected=False)
    
    
    
    ID_grupo_etario=1;

    p_to_uci=list_p_to_uci[ID_grupo_etario];
    n_to_uci=list_n_to_uci[ID_grupo_etario];
    p_to_uci_2021=list_p_to_uci_2021[ID_grupo_etario];
    n_to_uci_2021=list_n_to_uci_2021[ID_grupo_etario];
    ID_day_change_time=275;
    p_in_uci=list_p_in_uci[ID_grupo_etario];
    n_in_uci=list_n_in_uci[ID_grupo_etario];
    shift=True; 
    ID_list_day_new_variant=[260,289,320,348,379,409,440];
    ID_list_day_b117=[260,289,320,348,379,409,440];
    camas_uci_group_1=  Camas_UCI_new_variant(data, 
                                  ID_grupo_etario, 
                                  increase_vac1_Pfizer[ID_grupo_etario],increase_vac2_Pfizer[ID_grupo_etario],
                                  increase_vac1_Sinovac[ID_grupo_etario],increase_vac2_Sinovac[ID_grupo_etario],
                                  prob_uci_vac1_Pfizer[ID_grupo_etario],prob_uci_vac2_Pfizer[ID_grupo_etario],
                                  prob_uci_vac1_Sinovac[ID_grupo_etario],prob_uci_vac2_Sinovac[ID_grupo_etario],
                                  prob_uci_vac1_b117_Pfizer[ID_grupo_etario],prob_uci_vac2_b117_Pfizer[ID_grupo_etario],
                                  prob_uci_vac1_b117_Sinovac[ID_grupo_etario],prob_uci_vac2_b117_Sinovac[ID_grupo_etario],
                                  prob_uci_vac1_new_variant_Pfizer[ID_grupo_etario],prob_uci_vac2_new_variant_Pfizer[ID_grupo_etario],
                                  prob_uci_vac1_new_variant_Sinovac[ID_grupo_etario],prob_uci_vac2_new_variant_Sinovac[ID_grupo_etario],
                                  p_shares_new_variant[ID_grupo_etario],
                                  ID_list_day_new_variant,
                                  p_shares_b117[ID_grupo_etario],
                                  ID_list_day_b117,
                                  probability_not_vac[ID_grupo_etario],
                                  probability_not_vac_b117[ID_grupo_etario],
                                  probability_not_vac_new_variant[ID_grupo_etario],
                                  p_to_uci,
                                  n_to_uci,
                                  p_to_uci_2021,
                                  n_to_uci_2021,
                                  ID_day_change_time,
                                  p_in_uci,
                                  n_in_uci)
    mean_reasult_group_1=camas_uci_group_1.ICU_Simulations_camas_vac(N_INICIAL,
                                                                      CONFIDENCE,
                                                                      ERROR,
                                                                      start_date='2020-07-20',
                                                                      end_date='2021-04-19',shift= shift)
    camas_uci_group_1.plot_uci_pred_2020(data, uci, W=29, pond=1, start_date='2020-07-20',end_date='2021-05-15', infected=False)
    
    ID_grupo_etario=2;

    p_to_uci=list_p_to_uci[ID_grupo_etario];
    n_to_uci=list_n_to_uci[ID_grupo_etario];
    p_to_uci_2021=list_p_to_uci_2021[ID_grupo_etario];
    n_to_uci_2021=list_n_to_uci_2021[ID_grupo_etario];
    ID_day_change_time=275;
    p_in_uci=list_p_in_uci[ID_grupo_etario];
    n_in_uci=list_n_in_uci[ID_grupo_etario];
    shift=True;
    ID_list_day_new_variant=[260,289,320,348,379,409,440];
    ID_list_day_b117=[260,289,320,348,379,409,440];
    camas_uci_group_2=   Camas_UCI_new_variant(data, 
                                  ID_grupo_etario, 
                                  increase_vac1_Pfizer[ID_grupo_etario],increase_vac2_Pfizer[ID_grupo_etario],
                                  increase_vac1_Sinovac[ID_grupo_etario],increase_vac2_Sinovac[ID_grupo_etario],
                                  prob_uci_vac1_Pfizer[ID_grupo_etario],prob_uci_vac2_Pfizer[ID_grupo_etario],
                                  prob_uci_vac1_Sinovac[ID_grupo_etario],prob_uci_vac2_Sinovac[ID_grupo_etario],
                                  prob_uci_vac1_b117_Pfizer[ID_grupo_etario],prob_uci_vac2_b117_Pfizer[ID_grupo_etario],
                                  prob_uci_vac1_b117_Sinovac[ID_grupo_etario],prob_uci_vac2_b117_Sinovac[ID_grupo_etario],
                                  prob_uci_vac1_new_variant_Pfizer[ID_grupo_etario],prob_uci_vac2_new_variant_Pfizer[ID_grupo_etario],
                                  prob_uci_vac1_new_variant_Sinovac[ID_grupo_etario],prob_uci_vac2_new_variant_Sinovac[ID_grupo_etario],
                                  p_shares_new_variant[ID_grupo_etario],
                                  ID_list_day_new_variant,
                                  p_shares_b117[ID_grupo_etario],
                                  ID_list_day_b117,
                                  probability_not_vac[ID_grupo_etario],
                                  probability_not_vac_b117[ID_grupo_etario],
                                  probability_not_vac_new_variant[ID_grupo_etario],
                                  p_to_uci,
                                  n_to_uci,
                                  p_to_uci_2021,
                                  n_to_uci_2021,
                                  ID_day_change_time,
                                  p_in_uci,
                                  n_in_uci)
    mean_reasult_group_2=camas_uci_group_2.ICU_Simulations_camas_vac(N_INICIAL,
                                                                      CONFIDENCE,
                                                                      ERROR,
                                                                      start_date='2020-07-20',
                                                                      end_date='2021-04-19',shift= shift)
    camas_uci_group_2.plot_uci_pred_2020(data, uci, W=29, pond=1, start_date='2020-07-20',end_date='2021-05-15', infected=False)
    
    
    
    
    ID_grupo_etario=3;

    p_to_uci=list_p_to_uci[ID_grupo_etario];
    n_to_uci=list_n_to_uci[ID_grupo_etario];
    p_to_uci_2021=list_p_to_uci_2021[ID_grupo_etario];
    n_to_uci_2021=list_n_to_uci_2021[ID_grupo_etario];
    ID_day_change_time=275;
    p_in_uci=list_p_in_uci[ID_grupo_etario];
    n_in_uci=list_n_in_uci[ID_grupo_etario];
    shift=True; 
    ID_list_day_new_variant=[260,289,320,348,379,409,440];
    ID_list_day_b117=[260,289,320,348,379,409,440];
    camas_uci_group_3=  Camas_UCI_new_variant(data, 
                                  ID_grupo_etario, 
                                  increase_vac1_Pfizer[ID_grupo_etario],increase_vac2_Pfizer[ID_grupo_etario],
                                  increase_vac1_Sinovac[ID_grupo_etario],increase_vac2_Sinovac[ID_grupo_etario],
                                  prob_uci_vac1_Pfizer[ID_grupo_etario],prob_uci_vac2_Pfizer[ID_grupo_etario],
                                  prob_uci_vac1_Sinovac[ID_grupo_etario],prob_uci_vac2_Sinovac[ID_grupo_etario],
                                  prob_uci_vac1_b117_Pfizer[ID_grupo_etario],prob_uci_vac2_b117_Pfizer[ID_grupo_etario],
                                  prob_uci_vac1_b117_Sinovac[ID_grupo_etario],prob_uci_vac2_b117_Sinovac[ID_grupo_etario],
                                  prob_uci_vac1_new_variant_Pfizer[ID_grupo_etario],prob_uci_vac2_new_variant_Pfizer[ID_grupo_etario],
                                  prob_uci_vac1_new_variant_Sinovac[ID_grupo_etario],prob_uci_vac2_new_variant_Sinovac[ID_grupo_etario],
                                  p_shares_new_variant[ID_grupo_etario],
                                  ID_list_day_new_variant,
                                  p_shares_b117[ID_grupo_etario],
                                  ID_list_day_b117,
                                  probability_not_vac[ID_grupo_etario],
                                  probability_not_vac_b117[ID_grupo_etario],
                                  probability_not_vac_new_variant[ID_grupo_etario],
                                  p_to_uci,
                                  n_to_uci,
                                  p_to_uci_2021,
                                  n_to_uci_2021,
                                  ID_day_change_time,
                                  p_in_uci,
                                  n_in_uci)
    mean_reasult_group_3=camas_uci_group_3.ICU_Simulations_camas_vac(N_INICIAL,
                                                                      CONFIDENCE,
                                                                      ERROR,
                                                                      start_date='2020-07-20',
                                                                      end_date='2021-04-19',shift= shift)
    camas_uci_group_3.plot_uci_pred_2020(data, uci, W=29, pond=1, start_date='2020-07-20',end_date='2021-05-15', infected=False)
    
    
    ID_grupo_etario=4;
    p_to_uci=list_p_to_uci[ID_grupo_etario];
    n_to_uci=list_n_to_uci[ID_grupo_etario];
    p_to_uci_2021=list_p_to_uci_2021[ID_grupo_etario];
    n_to_uci_2021=list_n_to_uci_2021[ID_grupo_etario];
    ID_day_change_time;
    p_in_uci=list_p_in_uci[ID_grupo_etario];
    n_in_uci=list_n_in_uci[ID_grupo_etario];
    shift=True; 
    ID_list_day_new_variant=[260,289,320,348,379,409,440];
    ID_list_day_b117=[260,289,320,348,379,409,440];
    
    camas_uci_group_4=Camas_UCI_new_variant(data, 
                                  ID_grupo_etario, 
                                  increase_vac1_Pfizer[ID_grupo_etario],increase_vac2_Pfizer[ID_grupo_etario],
                                  increase_vac1_Sinovac[ID_grupo_etario],increase_vac2_Sinovac[ID_grupo_etario],
                                  prob_uci_vac1_Pfizer[ID_grupo_etario],prob_uci_vac2_Pfizer[ID_grupo_etario],
                                  prob_uci_vac1_Sinovac[ID_grupo_etario],prob_uci_vac2_Sinovac[ID_grupo_etario],
                                  prob_uci_vac1_b117_Pfizer[ID_grupo_etario],prob_uci_vac2_b117_Pfizer[ID_grupo_etario],
                                  prob_uci_vac1_b117_Sinovac[ID_grupo_etario],prob_uci_vac2_b117_Sinovac[ID_grupo_etario],
                                  prob_uci_vac1_new_variant_Pfizer[ID_grupo_etario],prob_uci_vac2_new_variant_Pfizer[ID_grupo_etario],
                                  prob_uci_vac1_new_variant_Sinovac[ID_grupo_etario],prob_uci_vac2_new_variant_Sinovac[ID_grupo_etario],
                                  p_shares_new_variant[ID_grupo_etario],
                                  ID_list_day_new_variant,
                                  p_shares_b117[ID_grupo_etario],
                                  ID_list_day_b117,
                                  probability_not_vac[ID_grupo_etario],
                                  probability_not_vac_b117[ID_grupo_etario],
                                  probability_not_vac_new_variant[ID_grupo_etario],
                                  p_to_uci,
                                  n_to_uci,
                                  p_to_uci_2021,
                                  n_to_uci_2021,
                                  ID_day_change_time,
                                  p_in_uci,
                                  n_in_uci)
    
    mean_reasult_group_4=camas_uci_group_4.ICU_Simulations_camas_vac(N_INICIAL,
                                                                      CONFIDENCE,
                                                                      ERROR,
                                                                      start_date='2020-07-20',
                                                                      end_date='2021-04-19',shift= shift)
    camas_uci_group_4.plot_uci_pred_2020(data, uci, W=29, pond=1, start_date='2020-07-20',end_date='2021-05-15', infected=False)
    
    
    for i,att in enumerate(dir(camas_uci_group_4)):
    if i not in list(range(4,31 )):
        print (att, getattr(camas_uci_group_4,att))
    
    day=334;len_uci=camas_uci_group_2.uci_beds[:,day-28:].shape[1];np.array([np.percentile(camas_uci_group_2.uci_beds[:,day-28:], 5,axis=0).T,np.percentile(camas_uci_group_2.uci_beds[:,day-28:], 95,axis=0).T,np.percentile(camas_uci_group_2.uci_beds[:,day-28:], 50,axis=0).T,uci[2,day-28:day-28+len_uci].round(2)]).T

    
"""      

