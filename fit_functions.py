import numpy as np
import random as random
from numpy.random import seed
from numpy.random import rand
import scipy
#using simpleguitk for display, is not needed for computer game
import simpleguitk as simplegui
import time
#new ones
import pandas as pd
#ml methods
from xgboost import XGBRegressor
from xgboost import XGBClassifier
#logistic regression
#from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LinearRegression
import os
#skyjo game classes and functions 
from skyjo_functions4 import *
#own functions for machine learning
from ml_functions2 import *
#for plotting 
#from matplotlib import pyplot as plt
#from matplotlib.pyplot import figure
#import seaborn as sns

#for confidence intervalls
from scipy.stats import beta
#for fitting of x y data 
from scipy.optimize import curve_fit
#for splitting
from sklearn.model_selection import train_test_split
#confusing matrix
#from sklearn.metrics import confusion_matrix
#for saving and loading of stranger object
import pickle


#gets new parameters
#and step sizes
def get_new_parameters(result):
    new_par=np.zeros(19)
    new_steps=np.zeros(19)
    for i in range(1,20):
        err=100*np.sqrt(np.sum(result[38,i,:])/np.sum(result[38,i,:])**2+np.sum(result[38,0,:])/np.sum(result[38,0,:])**2)
        diff=np.mean(result[40,i,:])-np.mean(result[40,0,:])
        #if really bad results or if significance less than 1 just use current base values 
        if np.mean(result[40,i,:])>75 or abs(diff/err)<1:
            new_par[i-1]=result[18+i,0,0]
            if abs(diff/err)<1:
                #direction cannot be known in thisc case increase it and change sign randomly
                new_steps[i-1]=3*(result[18+i,i,0]-result[18+i,0,0])*np.sign(random.random()-0.5)
            elif np.mean(result[40,i,:])<87:
                #reverse direction and less
                new_steps[i-1]=-1/3*(result[18+i,i,0]-result[18+i,0,0])
            elif np.mean(result[40,i,:])<94:
                new_steps[i-1]=-1/5*(result[18+i,i,0]-result[18+i,0,0])                 
            elif np.mean(result[40,i,:])<97:
                new_steps[i-1]=-1/9*(result[18+i,i,0]-result[18+i,0,0])             
            else:
                new_steps[i-1]=-1/17*(result[18+i,i,0]-result[18+i,0,0])               
        #else use the tried step
        else:
            new_par[i-1]=result[18+i,0,0]+(-result[18+i,i,0]+result[18+i,0,0])*np.sign(diff)
            if abs(diff/err)<3:
                #somewhat smaller in this case, diff/err gives direction needed when diff positive 
                new_steps[i-1]=-3/(diff/err)*(result[18+i,i,0]-result[18+i,0,0])
            else:    
                new_steps[i-1]=-1/np.sign(diff/err)*(result[18+i,i,0]-result[18+i,0,0])                
    return new_par,new_steps       

#means changes within the function do not have an effect outside of it , copy it to the top functions does not change anything 
def first_gradient_step(open_steps,discard_steps,value_steps,realizations):
    n_it=realizations
    results=np.zeros((40,19))
    for j in range(results.shape[1]):
        print(f"doing case {j}")
        level20_open_variable=np.zeros((6))
        level20_discard_variable=np.zeros((6))
        level20_value_variable=np.zeros((7))
        level21_open_variable=np.zeros((6))
        level21_discard_variable=np.zeros((6))
        level21_value_variable=np.zeros((7))
        if j<6:
            level21_open_variable[j]=open_steps[j]
        elif j<12:
            level21_discard_variable[j-6]=discard_steps[j-6]
        else:
            level21_value_variable[j-12]=value_steps[j-12]    
        print(level21_open_variable,level21_discard_variable,level21_value_variable)
        results[0:6,j]=level20_open_variable
        results[6:12,j]=level20_discard_variable
        results[12:19,j]=level20_value_variable    
        results[19:25,j]=level21_open_variable
        results[25:31,j]=level21_discard_variable
        results[31:38,j]=level21_value_variable         
        win20=0
        start_time=time.time()
        for i in range(n_it):
            names=['alpha','beta']
            nature=['computer','computer']
            levels=[20,21]
            winner=skyjo_game(names,nature,levels,0,True,False,level20_open_variable=level20_open_variable,level21_open_variable=level21_open_variable,level20_discard_variable=level20_discard_variable,level21_discard_variable=level21_discard_variable,level20_value_variable=level20_value_variable,level21_value_variable=level21_value_variable)
            if winner[0]==1:
                win20+=1
        results[38,j]=n_it
        results[39,j]=win20
        stop_time=time.time()
        print(f"{n_it} games need {np.round(stop_time-start_time,3)} seconds")
        print(f"level 20 won to {np.round(win20*100/n_it,1)} %")
    return results     

#x0 is fixed on 0 because that is the case floor added
def logistic_adapted(x,a,b,c):
    return a/(1+np.exp(-b*x))+c


#does Monte carlo against a number of cases
#open_vars and co and and are lists of the variables to be used 
#open_ranges and discard_ranges are 2,6 np arrays, value_ranges is 2,7 np array, 
#reliazations are number of models tried in one, trials are number of MC models for level 21
def montecarlo_trials(open_vars,discard_vars,value_vars,open_ranges,discard_ranges,value_ranges,realizations,trials):
    n_it=realizations
    results=np.zeros((40,trials,len(open_vars)))
    for j in range(trials):
        print(f"doing Monte Carlo {j} for level 21")
        #create random values for level 21 within the ranges, save are used for all level 20 models
        level21_open_variable=np.zeros((6))
        level21_discard_variable=np.zeros((6))
        level21_value_variable=np.zeros((7))
        for k in range(7):
            level21_value_variable[k]=value_ranges[0,k]+random.random()*(value_ranges[1,k]-value_ranges[0,k])
            if k<6:
                level21_open_variable[k]=open_ranges[0,k]+random.random()*(open_ranges[1,k]-open_ranges[0,k])
                level21_discard_variable[k]=discard_ranges[0,k]+random.random()*(discard_ranges[1,k]-discard_ranges[0,k])
        for k in range(len(open_vars)):
            print(f"trying case {k} for level 20")
            level20_open_variable=np.array(open_vars[k])
            level20_discard_variable=np.array(discard_vars[k])
            level20_value_variable=np.array(value_vars[k])             
            print(np.round(level21_open_variable,5),np.round(level21_discard_variable,5),np.round(level21_value_variable,5))
            results[0:6,j,k]=level20_open_variable
            results[6:12,j,k]=level20_discard_variable
            results[12:19,j,k]=level20_value_variable    
            results[19:25,j,k]=level21_open_variable
            results[25:31,j,k]=level21_discard_variable
            results[31:38,j,k]=level21_value_variable         
            win20=0
            start_time=time.time()
            for i in range(n_it):
                names=['alpha','beta']
                nature=['computer','computer']
                levels=[20,21]
                winner=skyjo_game(names,nature,levels,0,True,False,level20_open_variable=level20_open_variable,level21_open_variable=level21_open_variable,level20_discard_variable=level20_discard_variable,level21_discard_variable=level21_discard_variable,level20_value_variable=level20_value_variable,level21_value_variable=level21_value_variable)
                if winner[0]==1:
                    win20+=1
            results[38,j,k]=n_it
            results[39,j,k]=win20
            stop_time=time.time()
            print(f"{n_it} games need {np.round(stop_time-start_time,3)} seconds")
            print(f"level 20 won to {np.round(win20*100/n_it,1)} %")
        print(f"level 20 won in average to {np.round(np.mean(results[39,j,:])*100/n_it,1)} %")   
    return results 


#does Monte carlo against a number of cases
#open_vars and co and and are lists of the variables to be used 
#open_ranges and discard_ranges are 2,6 np arrays, value_ranges is 2,7 np array, 
#reliazations are number of models tried in one, trials are number of MC models for level 21
#wfrac when it aborts early
#wfrac2 performance of worst case .
def montecarlo_trials2(open_vars,discard_vars,value_vars,open_ranges,discard_ranges,value_ranges,realizations,trials,wfrac=0.10,wfrac2=0.551):
    n_it=realizations
    results=np.zeros((41,trials,len(open_vars)))
    for j in range(trials):
        print(f"doing Monte Carlo {j} for level 21")
        #create random values for level 21 within the ranges, save are used for all level 20 models
        level21_open_variable=np.zeros((6))
        level21_discard_variable=np.zeros((6))
        level21_value_variable=np.zeros((7))
        for k in range(7):
            level21_value_variable[k]=value_ranges[0,k]+random.random()*(value_ranges[1,k]-value_ranges[0,k])
            if k<6:
                level21_open_variable[k]=open_ranges[0,k]+random.random()*(open_ranges[1,k]-open_ranges[0,k])
                level21_discard_variable[k]=discard_ranges[0,k]+random.random()*(discard_ranges[1,k]-discard_ranges[0,k])
        for k in range(len(open_vars)):
            print(f"trying case {k} for level 20")
            level20_open_variable=np.array(open_vars[k])
            level20_discard_variable=np.array(discard_vars[k])
            level20_value_variable=np.array(value_vars[k])             
            results[0:6,j,k]=level20_open_variable
            results[6:12,j,k]=level20_discard_variable
            results[12:19,j,k]=level20_value_variable    
            results[19:25,j,k]=level21_open_variable
            results[25:31,j,k]=level21_discard_variable
            results[31:38,j,k]=level21_value_variable         
            win20=0
            it_counter=0
            start_time=time.time()
            n_it1=n_it
            if n_it1>22:
                n_it1=22
            #22 always     
            for i in range(n_it1):
                names=['alpha','beta']
                nature=['computer','computer']
                levels=[20,21]
                it_counter+=1
                winner=skyjo_game(names,nature,levels,0,True,False,level20_open_variable=level20_open_variable,level21_open_variable=level21_open_variable,level20_discard_variable=level20_discard_variable,level21_discard_variable=level21_discard_variable,level20_value_variable=level20_value_variable,level21_value_variable=level21_value_variable)
                if winner[0]==1:
                    win20+=1
            #98% ownside win conditions tested here   2.33 sigma   stops early when new model clearly bad or good 
            while it_counter<n_it and abs((win20-it_counter/2)/np.sqrt(it_counter))<2.33:  
                names=['alpha','beta']
                nature=['computer','computer']
                levels=[20,21]
                it_counter+=1
                winner=skyjo_game(names,nature,levels,0,True,False,level20_open_variable=level20_open_variable,level21_open_variable=level21_open_variable,level20_discard_variable=level20_discard_variable,level21_discard_variable=level21_discard_variable,level20_value_variable=level20_value_variable,level21_value_variable=level21_value_variable)
                if winner[0]==1:
                    win20+=1                
            #now checking whether 98% sigficant on bad performance        
            results[38,j,k]=it_counter
            results[39,j,k]=win20
            results[40,j,k]=100*win20/it_counter         
            stop_time=time.time()
            print(f"{it_counter} games need {np.round(stop_time-start_time,3)} seconds")
            print(f"level 20 won to {np.round(results[40,j,k],1)} %")
        print(f"level 20 won in average to {np.round(np.mean(results[40,j,:]),1)} %") 
        if np.mean(results[40,j,:]/100)<wfrac and np.max(results[40,j,:]/100)<wfrac2:
            print("better model found")
            if j==0:
                return True, results[:,j,:], results[:,j,:]
            else:
                #give back fit converged or not, best models, all models 
                return True, results[:,j,:], results[:,0:j+1,:]
    print("no better model found") 
    #second parameter is noen will not be used later
    return False, None, results 

#using input arrays of varaible shape[1] where the largest the the last one 
#only possible if there is one to add
def bestfit_to_array(list_best):
    if len(list_best)>0:
        last=list_best[len(list_best)-1]
        results=np.zeros((last.shape[0],last.shape[1],len(list_best)))
        #set not filled values to -100 to be clearly impossible
        results[:,:,:]=-100
        for i in range(len(list_best)):
            results[0:list_best[i].shape[0],0:list_best[i].shape[1],i]=list_best[i]
        return results  
    else:
        print("List is of length 0, no combination possible.")
        results=np.array([-100.])
        return results
    
#works on list of 3d arrays, also of length 1 not of length 0, but that should not possible 
def allfits_to_array(list_all):
    if len(list_all)>0:
        dims=np.zeros((3,len(list_all)))
        for i in range(len(list_all)):
            dims[:,i]=list_all[i].shape
        results=np.zeros((int(max(dims[0,:])),int(max(dims[1,:])),int(max(dims[2,:])),len(list_all)))
        results[:]=-100  
        for i in range(len(list_all)):
            results[0:list_all[i].shape[0],0:list_all[i].shape[1],0:list_all[i].shape[2],i]=list_all[i]
        return results 
    else:
        print("List is of length 0, no combination possible.")
        results=np.array([-100.])
        return results   

#paarmeters, list of open parameters,  list of discard parameters, list of value parameters,
#open parameter ranges, discard parameter ranges, value parameter ranges,
#maximum number of games palyed for model pair. 
#Maximum number of Monte carlo pairs, 
#maxium number of added models tried,
#wfrac average maximum win fraction  for level 20, #wfrac2 maximum win fraction for single level 20 model
#both need to be smaller
def montecarlo_fit(open_vars,discard_vars,value_vars,open_ranges,discard_ranges,value_ranges,realizations=50,trials=10,max_iter=10,wfrac=0.10,wfrac2=0.5):
    #list to performances of winner models
    list_best_result=[]
    list_all_results=[]
    for i in range(max_iter):
        print(f"iteration {i} using {len(open_vars)} level 20 models")
        boolean,best_result,all_results=montecarlo_trials2(open_vars,discard_vars,value_vars,open_ranges,discard_ranges,value_ranges,realizations,trials,wfrac=wfrac,wfrac2=wfrac2)
        # good model found
        if boolean==True:
            #append this model
            open_vars.append(best_result[19:25,0])
            discard_vars.append(best_result[25:31,0])
            value_vars.append(best_result[31:38,0])
            print("model to append found")
            print(f"best model won to {np.round(100-np.mean(best_result[40,:]),1)} %")
            list_best_result.append(best_result)
            list_all_results.append(all_results)
        else:
            #no better found, still data collected but not ;best model added'      
            print("no model to append found iterations stopped early")
            list_all_results.append(all_results)
            array_best_result=bestfit_to_array(list_best_result)
            array_all_results=allfits_to_array(list_all_results)    
            return open_vars, discard_vars, value_vars,array_best_result,array_all_results 
    print("all iterations done")
    #converting lists to arrays
    array_best_result=bestfit_to_array(list_best_result)
    array_all_results=allfits_to_array(list_all_results)    
    return open_vars, discard_vars, value_vars,array_best_result,array_all_results    
#should also put all model and to save performances   

#gradient gets local gradient compared to input models (can be own but should several to avoid too
#much orienation to a single )
#input models open_vars,discard_vars,value_vars
#single model (can be mean or something else) around which steps are done open_avg,discard_avg,value_avg
#step size vectors to step to explore open_step,discard_step,value_step
#n_games number of games done for the exploring models, base model does more by fixed factor
def gradient_step1(open_vars,discard_vars,value_vars,open_avg,discard_avg,value_avg,open_step,discard_step,value_step,n_games):
    #always 20 because of doing 20 paaremeters separametely plus the base model
    results=np.zeros((41,20,len(open_vars)))
    for j in range(20):
        level21_open_variable=np.array(open_avg)
        level21_discard_variable=np.array(discard_avg)      
        level21_value_variable=np.array(value_avg)           
        if j==0:
            print(f"doing Base model")             
        #start now gradient with open 
        elif j<7:
            print(f"changing open variable {j-1}")  
            level21_open_variable[j-1]+=open_step[j-1]                 
        elif j<13:
            print(f"changing discard variable {j-7}")  
            level21_discard_variable[j-7]+=discard_step[j-7]
        else:
            print(f"changing value variable {j-13}")             
            level21_value_variable[j-13]+=value_step[j-13]        
        for k in range(len(open_vars)):
            print(f"doing case {k} for level 20")
            level20_open_variable=np.array(open_vars[k])
            level20_discard_variable=np.array(discard_vars[k])
            level20_value_variable=np.array(value_vars[k])             
            results[0:6,j,k]=level20_open_variable
            results[6:12,j,k]=level20_discard_variable
            results[12:19,j,k]=level20_value_variable    
            results[19:25,j,k]=level21_open_variable
            results[25:31,j,k]=level21_discard_variable
            results[31:38,j,k]=level21_value_variable         
            win20=0
            start_time=time.time()
            #4 times for often used based model
            if j==0:
                n_games_here=n_games*4
            else:
                n_games_here=n_games   
            for i in range(n_games_here):
                names=['alpha','beta']
                nature=['computer','computer']
                levels=[20,21]
                winner=skyjo_game(names,nature,levels,0,True,False,level20_open_variable=level20_open_variable,level21_open_variable=level21_open_variable,level20_discard_variable=level20_discard_variable,level21_discard_variable=level21_discard_variable,level20_value_variable=level20_value_variable,level21_value_variable=level21_value_variable)
                if winner[0]==1:
                    win20+=1                    
            results[38,j,k]=n_games_here
            results[39,j,k]=win20
            results[40,j,k]=100*win20/n_games_here     
            stop_time=time.time()
            print(f"{n_games_here} games need {np.round(stop_time-start_time,3)} seconds")
            print(f"level 20 won to {np.round(results[40,j,k],1)} %")
        print(f"level 20 won in average to {np.round(np.mean(results[40,j,:]),1)} %") 
    print("all models done") 
    return results 

def get_range(x_plus,y_plus,x_minus, y_minus):
    result=np.zeros((y_plus.shape[0]))
    for i in range(y_plus.shape[0]):
        if np.mean(y_plus[i,:])>np.mean(y_minus[i,:]):
            a=x_plus[np.argmax(y_plus[i,:])]
            if a>0.1:
                a=0.1
            b=compm[0,np.argmin(y_minus[i,:])]
            if b<-0.1:
                b=-0.1
            result[i]=(a+np.abs(b))/2    
        else:
            a=x_plus[np.argmin(y_plus[i,:])]
            if a>0.1:
                a=0.1
            b=compm[0,np.argmax(y_minus[i,:])]
            if b<-0.1:
                b=-0.1
            result[i]=(a+np.abs(b))/2  
    return result            
#get max ranges but not ideal since affected by small random wggles, still o.k. for now 

#parameters,list of open paremeters, discard parameters, value parameters for level 20
# open base, discard base, value base parameters for level 21
# open steps, discard steps, value steps parameters for level 21
def gradient_fit(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,open_step,discard_step,value_step,n_games=100,max_iter=10,output_name="gradient_fit1_it"):
    #to save parameters and steps 
    results=np.zeros((19,max_iter,2))
    for i in range(max_iter):
        print(f"doing iteration {i}")
        start_time=time.time()
        resgrad1=gradient_step1(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,open_step,discard_step,value_step,n_games=n_games)
        np.save(output_name+str(i)+".npy",resgrad1)
        stop_time=time.time()
        print(f"{n_games} games need {np.round(stop_time-start_time,3)} seconds")
        #getting new steps and parameters
        new_par1,new_step1=get_new_parameters(resgrad1)
        results[:,i,0]=new_par1
        results[:,i,1]=new_step1        
        #passing as new base
        base_open=new_par1[0:6]
        base_discard=new_par1[6:12]
        base_value=new_par1[12:19]        
        #passing as new steps
        open_step=new_step1[0:6]
        discard_step=new_step1[6:12]
        value_step=new_step1[12:19]
    return results    


def get_new_parameters2(result):
    new_par=np.zeros(19)
    new_steps=np.zeros(19)
    for i in range(1,20):
        err=100*np.sqrt(np.sum(result[38,i,:])/np.sum(result[38,i,:])**2+np.sum(result[38,0,:])/np.sum(result[38,0,:])**2)
        diff=np.mean(result[40,i,:])-np.mean(result[40,0,:])
        #if really bad results or if significance less than 1 just use current base values 
        if np.mean(result[40,i,:])>75 or abs(diff/err)<1:
            new_par[i-1]=result[18+i,0,0]
            if abs(diff/err)<1:
                #direction cannot be known in thisc case increase it and change sign randomly
                #was 3 before
                new_steps[i-1]=2*(result[18+i,i,0]-result[18+i,0,0])*np.sign(random.random()-0.5)
            elif np.mean(result[40,i,:])<87:
                #reverse direction and less
                new_steps[i-1]=-1/3*(result[18+i,i,0]-result[18+i,0,0])
            elif np.mean(result[40,i,:])<94:
                new_steps[i-1]=-1/5*(result[18+i,i,0]-result[18+i,0,0])                 
            elif np.mean(result[40,i,:])<97:
                new_steps[i-1]=-1/9*(result[18+i,i,0]-result[18+i,0,0])             
            else:
                new_steps[i-1]=-1/17*(result[18+i,i,0]-result[18+i,0,0])               
        #else use the tried step
        else:
            # all of 1.5/abs(diff/err)
            new_par[i-1]=result[18+i,0,0]+(-result[18+i,i,0]+result[18+i,0,0])*np.sign(diff)*1.5/abs(diff/err)
            #before was -3 and was at least the size of the previous step always 
            new_steps[i-1]=-2/(diff/err)*(result[18+i,i,0]-result[18+i,0,0])              
    return new_par,new_steps    

# past fit result,  factor for next base is divided by signifciance
#factor on step when not significant
#factor on step when signifciance divide by signifant
def get_new_parameters3(result,fact_new_par=1.5,fact_new_step_no=2.5,fact_new_step_sig=2.0):
    new_par=np.zeros(19)
    new_steps=np.zeros(19)
    for i in range(1,20):
        err=100*np.sqrt(np.sum(result[38,i,:])/np.sum(result[38,i,:])**2+np.sum(result[38,0,:])/np.sum(result[38,0,:])**2)
        diff=np.mean(result[40,i,:])-np.mean(result[40,0,:])
        #if really bad results or if significance less than 1 just use current base values 
        if np.mean(result[40,i,:])>75 or abs(diff/err)<1:
            new_par[i-1]=result[18+i,0,0]
            if abs(diff/err)<1:
                #direction cannot be known in thisc case increase it and change sign randomly
                #was 3 before
                new_steps[i-1]=fact_new_step_no*(result[18+i,i,0]-result[18+i,0,0])*np.sign(random.random()-0.5)
            elif np.mean(result[40,i,:])<87:
                #reverse direction and less
                new_steps[i-1]=-1/3*(result[18+i,i,0]-result[18+i,0,0])
            elif np.mean(result[40,i,:])<94:
                new_steps[i-1]=-1/5*(result[18+i,i,0]-result[18+i,0,0])                 
            elif np.mean(result[40,i,:])<97:
                new_steps[i-1]=-1/9*(result[18+i,i,0]-result[18+i,0,0])             
            else:
                new_steps[i-1]=-1/17*(result[18+i,i,0]-result[18+i,0,0])               
        #else use the tried step
        else:
            # all of 1.5/abs(diff/err)
            new_par[i-1]=result[18+i,0,0]+(-result[18+i,i,0]+result[18+i,0,0])*np.sign(diff)*fact_new_par/abs(diff/err)
            #before was -3 and was at least the size of the previous step always 
            new_steps[i-1]=-fact_new_step_sig/(diff/err)*(result[18+i,i,0]-result[18+i,0,0])
    return new_par,new_steps    


#parameters,list of open paremeters, discard parameters, value parameters for level 20
# open base, discard base, value base parameters for level 21
# open steps, discard steps, value steps parameters for level 21
#then optional the parameter to adjust step size of get_new_parameters3
# factor for next base is divided by signifciance
#factor on step when not significant
#factor on step when signifciance divide by signifant
def gradient_fit2(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,open_step,discard_step,value_step,n_games=100,max_iter=10,output_name="gradient2_fit1_it",fact_new_par=1.5,fact_new_step_no=2.5,fact_new_step_sig=2.0):
    #to save parameters and steps 
    results=np.zeros((19,max_iter,2))
    for i in range(max_iter):
        print(f"doing iteration {i}")
        start_time=time.time()
        resgrad1=gradient_step1(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,open_step,discard_step,value_step,n_games=n_games)
        np.save(output_name+str(i)+".npy",resgrad1)
        stop_time=time.time()
        print(f"{n_games} games need {np.round(stop_time-start_time,3)} seconds")
        #getting new steps and parameters
        new_par1,new_step1=get_new_parameters3(resgrad1,fact_new_par=fact_new_par,fact_new_step_no=fact_new_step_no,fact_new_step_sig=fact_new_step_sig)
        results[:,i,0]=new_par1
        results[:,i,1]=new_step1        
        #passing as new base
        base_open=new_par1[0:6]
        base_discard=new_par1[6:12]
        base_value=new_par1[12:19]        
        #passing as new steps
        open_step=new_step1[0:6]
        discard_step=new_step1[6:12]
        value_step=new_step1[12:19]
    return results    

#running asignle level 21 against a list of level 20 models 
#tot_score collect of not 
def run_level21(open_vars,discard_vars,value_vars,open_target,discard_target,value_target,n_games,tot_score_collect=False):
    #different size
    if tot_score_collect==False:
       results=np.zeros((41,len(open_vars)))
    else:
       results=np.zeros((42,len(open_vars)))       
    #getting the same level 21 model         
    level21_open_variable=np.array(open_target)
    level21_discard_variable=np.array(discard_target)      
    level21_value_variable=np.array(value_target)                 
    for k in range(len(open_vars)):
        tot_scores=np.zeros((2))
        print(f"doing case {k} for level 20")
        level20_open_variable=np.array(open_vars[k])
        level20_discard_variable=np.array(discard_vars[k])
        level20_value_variable=np.array(value_vars[k])             
        results[0:6,k]=level20_open_variable
        results[6:12,k]=level20_discard_variable
        results[12:19,k]=level20_value_variable    
        results[19:25,k]=level21_open_variable
        results[25:31,k]=level21_discard_variable
        results[31:38,k]=level21_value_variable         
        win20=0
        start_time=time.time()  
        for i in range(n_games):
            names=['alpha','beta']
            nature=['computer','computer']
            levels=[20,21]
            if tot_score_collect==False:
                winner=skyjo_game(names,nature,levels,0,True,tot_score_collect=False,level20_open_variable=level20_open_variable,level21_open_variable=level21_open_variable,level20_discard_variable=level20_discard_variable,level21_discard_variable=level21_discard_variable,level20_value_variable=level20_value_variable,level21_value_variable=level21_value_variable)
            else:
                winner,tot_score=skyjo_game(names,nature,levels,0,True,tot_score_collect=True,level20_open_variable=level20_open_variable,level21_open_variable=level21_open_variable,level20_discard_variable=level20_discard_variable,level21_discard_variable=level21_discard_variable,level20_value_variable=level20_value_variable,level21_value_variable=level21_value_variable)   
                tot_scores+=tot_score              
            if winner[0]==1:
                win20+=1                    
            results[38,k]=n_games
            results[39,k]=win20
            results[40,k]=100*win20/n_games     
            if tot_score_collect==True:
                results[41,k]=(tot_scores[0]-tot_scores[1])/n_games
            stop_time=time.time()
        print(f"{n_games} games need {np.round(stop_time-start_time,3)} seconds")
        print(f"level 20 won to {np.round(results[40,k],1)} %")
    print(f"level 20 won in average to {np.round(np.mean(results[40,:]),1)} %") 
    print("all models done") 
    return results 

#2 grid in l2regularziation and max depth
#parameters: feature of train, target of train, feature of test, target of test,
#minimum max deoth, maximum max depth, minimum l2 regularization,
#factor of increase, number of steps, output file name regression=True default
#save=True default, result saved as file otherwise returns the results 
def loop_reg2(feature_train, target_train, feature_test, target_test,max_depth_start,max_depth_stop,reg_start,reg_increase,reg_steps,file_name,regression=True,save=True):
    #creates file to be saved 
    resb=np.zeros((4,reg_steps,int(max_depth_stop-max_depth_start+1)))
    #regularization grid
    for i in range(reg_steps):
        print(f"regularization doing case {i}")
        #max depth grid 
        for j in range(resb.shape[2]):
            regularization=reg_start*reg_increase**i
            max_depth=j+max_depth_start
            #regression
            if regression==True:
                ar=do_xgb(feature_train, target_train, feature_test, target_test,max_depth,reg=regularization)
            #classification
            else:
                ar=do_xgb_class(feature_train, target_train, feature_test, target_test,max_depth,reg=regularization)
            resb[:,i,j]=ar
    if save==True:        
        np.save(file_name, resb) 
    else:        
        return resb        
    
#does comparisons against a number of cases
#open_vars and co and and are lists of the variables to be used 
#open_21 and teh on are the 
#reliazations are number of models tried in one, trials are number of MC models for level 21
#wfrac when it aborts early
#wfrac2 performance of worst case .
def many_comparisons(open_vars,discard_vars,value_vars,open_21,discard_21,value_21,realizations,wfrac=0.10,wfrac2=0.551):
    n_it=realizations
    trials=len(open_21)
    results=np.zeros((41,trials,len(open_vars)))
    for j in range(trials):
        print(f"doing case {j} for level 21")
        #create random values for level 21 within the ranges, save are used for all level 20 models
        level21_open_variable=np.array(open_21[j])
        level21_discard_variable=np.array(discard_21[j])
        level21_value_variable=np.array(value_21[j])          
        for k in range(len(open_vars)):
            print(f"trying case {k} for level 20")
            level20_open_variable=np.array(open_vars[k])
            level20_discard_variable=np.array(discard_vars[k])
            level20_value_variable=np.array(value_vars[k])            
            results[0:6,j,k]=level20_open_variable
            results[6:12,j,k]=level20_discard_variable
            results[12:19,j,k]=level20_value_variable    
            results[19:25,j,k]=level21_open_variable
            results[25:31,j,k]=level21_discard_variable
            results[31:38,j,k]=level21_value_variable         
            win20=0
            it_counter=0
            start_time=time.time()
            n_it1=n_it
            if n_it1>22:
                n_it1=22
            #22 always     
            for i in range(n_it1):
                names=['alpha','beta']
                nature=['computer','computer']
                levels=[20,21]
                it_counter+=1
                winner=skyjo_game(names,nature,levels,0,True,False,level20_open_variable=level20_open_variable,level21_open_variable=level21_open_variable,level20_discard_variable=level20_discard_variable,level21_discard_variable=level21_discard_variable,level20_value_variable=level20_value_variable,level21_value_variable=level21_value_variable)
                if winner[0]==1:
                    win20+=1
            #98% ownside win conditions tested here   2.33 sigma   stops early when new model clearly bad or good 
            while it_counter<n_it and abs((win20-it_counter/2)/np.sqrt(it_counter))<2.33:  
                names=['alpha','beta']
                nature=['computer','computer']
                levels=[20,21]
                it_counter+=1
                winner=skyjo_game(names,nature,levels,0,True,False,level20_open_variable=level20_open_variable,level21_open_variable=level21_open_variable,level20_discard_variable=level20_discard_variable,level21_discard_variable=level21_discard_variable,level20_value_variable=level20_value_variable,level21_value_variable=level21_value_variable)
                if winner[0]==1:
                    win20+=1                
            #now checking whether 98% sigficant on bad performance        
            results[38,j,k]=it_counter
            results[39,j,k]=win20
            results[40,j,k]=100*win20/it_counter         
            stop_time=time.time()
            print(f"{it_counter} games need {np.round(stop_time-start_time,3)} seconds")
            print(f"level 20 won to {np.round(results[40,j,k],1)} %")
        print(f"level 20 won in average to {np.round(np.mean(results[40,j,:]),1)} %") 
        if np.mean(results[40,j,:]/100)<wfrac and np.max(results[40,j,:]/100)<wfrac2:
            print("better model found")
            if j==0:
                return True, results[:,j,:], results[:,j,:]
            else:
                #give back fit converged or not, best models, all models 
                return True, results[:,j,:], results[:,0:j+1,:]
    print("no better model found") 
    #second parameter is noen will not be used later
    return False, None, results   


#function to select wanted range in win frac frac
#parameters, wins, minwin, max win
def select_range(params,wins,minwin=0,maxwin=100):
    selparams=np.zeros((params.shape[0],params.shape[1]))
    selwins=np.zeros((wins.shape[0]))
    c=0
    for i in range(wins.shape[0]):
        if wins[i]>=minwin and wins[i]<=maxwin:
            selparams[:,c]=params[:,i]
            selwins[c]=wins[i]
            c+=1
    print(f"{c} cases are selected")    
    if c>1:
        return c, selparams[:,0:c], selwins[0:c]
    elif c==1:
        return c, selparams[:,0], selwins[0]
    else:
        return c, None, None
    
def random_xgboost(params,wins,train_size=0.67,random_start=1,random_stop=2,minwin=0,maxwin=100,max_depth_start=1,max_depth_stop=6,reg_start=1,reg_factor=1.414,reg_steps=10):
    #select wanted range
    counter,selparams,selwins=select_range(params,wins,minwin=minwin, maxwin=maxwin)
    #collect statistics
    xmodelstats_all=np.zeros((4,reg_steps,max_depth_stop-max_depth_start+1,random_stop-random_start+1))
    #loop over random indixes
    for i in range(random_start,random_stop+1):
        print(f"doing xgboost with seed {i}")
        feature_train,feature_test,target_train,target_test=train_test_split(selparams.T,selwins,train_size=train_size, shuffle=True, random_state=i)
        xmodelstats_single=loop_reg2(feature_train, target_train, feature_test, target_test,max_depth_start,max_depth_stop,reg_start,reg_factor,reg_steps,"dummy",save=False)
        print(xmodelstats_single.shape)
        xmodelstats_all[:,:,:,i-1]=xmodelstats_single
    #average over random realizations    
    xmodelstats_avg=np.mean(xmodelstats_all[:,:,:,:],3)
    return xmodelstats_avg    

def get_best_xgboost(stats,features,target,minwin=0,maxwin=100):
    counter,selparams,selwins=select_range(features,target,minwin=minwin, maxwin=maxwin)
    ind = np.unravel_index(np.argmin(stats[3], axis=None), stats[3].shape)
    print("best model has")
    print(f"reg={stats[0,ind[0],ind[1]]}, max-depth={int(stats[1,ind[0],ind[1]])},train-scatter={stats[2,ind[0],ind[1]]},test-scatter={stats[3,ind[0],ind[1]]}")
    #run model 
    xmodel1=XGBRegressor(max_depth=int(stats[1,ind[0],ind[1]]),reg_alpha=stats[0,ind[0],ind[1]]).fit(selparams.T,selwins)
    return xmodel1

def get_sample(xmodel1,its_trails=1000000,minstars=1000,max_win=50,seed2=1,filename1='selparams_xgb1.npy',filename2='pred_xgb1.npy'):
    seed(seed2)
    aselmodels=np.zeros((19,minstars*3))
    axselmodels=np.zeros((minstars*3))
    c=0
    n=0
    while c<minstars:
        n+=1
        rand2 =rand(its_trails,19)*0.2-0.1
        c2=0
        xpred=xmodel1.predict(rand2)
        for i in range(len(xpred)):
            if xpred[i]<max_win:
                c+=1
                c2+=1
        print(f"{c} cases accumulated")
        print(f"minimum win is {min(xpred)}")
        if c2>=minstars and c==c2:    
            selmodels=np.zeros((19,c2))
            xselmodels=np.zeros((c2))
            c=0
            for i in range(len(xpred)):
                if xpred[i]<max_win:
                    selmodels[:,c]=rand2[i,:]
                    xselmodels[c]=xpred[i]
                    c+=1
            print("just one was enough")        
            np.save(filename1,selmodels)       
            np.save(filename2,xselmodels) 
        elif c<minstars:
            selmodels=np.zeros((19,c2))
            xselmodels=np.zeros((c2))
            c2=0
            for i in range(len(xpred)):
                if xpred[i]<max_win:
                    selmodels[:,c2]=rand2[i,:]
                    xselmodels[c2]=xpred[i]
                    c2+=1       
            aselmodels[:,c-c2:c]=selmodels
            axselmodels[c-c2:c]=xselmodels
        else:  
            selmodels=np.zeros((19,c2))
            xselmodels=np.zeros((c2))
            c2=0
            for i in range(len(xpred)):
                if xpred[i]<max_win:
                    selmodels[:,c2]=rand2[i,:]
                    xselmodels[c2]=xpred[i]
                    c2+=1       
            aselmodels[:,c-c2:c]=selmodels
            axselmodels[c-c2:c]=xselmodels
            print(f"{n} were enough")        
            np.save(filename1,aselmodels[:,0:c])       
            np.save(filename2,axselmodels[0:c]) 
            
#gradient gets local gradient compared to input models (can be own but should several to avoid too
#much orienation to a single )
#input models open_vars,discard_vars,value_vars
#single model (can be mean or something else) around which steps are done open_avg,discard_avg,value_avg
#step size vectors to step to explore open_step,discard_step,value_step
#n_games number of games done for the exploring models
#the inout model is not done here but separately 
def gradient_step2(open_vars,discard_vars,value_vars,open_avg,discard_avg,value_avg,open_step,discard_step,value_step,n_games):
    #always 19 because of doing 19 parameters separately plus the base model
    results=np.zeros((41,19,len(open_vars)))
    for j in range(19):
        level21_open_variable=np.array(open_avg)
        level21_discard_variable=np.array(discard_avg)      
        level21_value_variable=np.array(value_avg)                      
        #start now gradient with open 
        if j<6:
            print(f"changing open variable {j}")  
            level21_open_variable[j]+=open_step[j]                 
        elif j<12:
            print(f"changing discard variable {j-6}")  
            level21_discard_variable[j-6]+=discard_step[j-6]
        else:
            print(f"changing value variable {j-12}")             
            level21_value_variable[j-12]+=value_step[j-12]        
        for k in range(len(open_vars)):
            print(f"doing case {k} for level 20")
            level20_open_variable=np.array(open_vars[k])
            level20_discard_variable=np.array(discard_vars[k])
            level20_value_variable=np.array(value_vars[k])             
            results[0:6,j,k]=level20_open_variable
            results[6:12,j,k]=level20_discard_variable
            results[12:19,j,k]=level20_value_variable    
            results[19:25,j,k]=level21_open_variable
            results[25:31,j,k]=level21_discard_variable
            results[31:38,j,k]=level21_value_variable         
            win20=0
            start_time=time.time()   
            for i in range(n_games):
                names=['alpha','beta']
                nature=['computer','computer']
                levels=[20,21]
                winner=skyjo_game(names,nature,levels,0,True,False,level20_open_variable=level20_open_variable,level21_open_variable=level21_open_variable,level20_discard_variable=level20_discard_variable,level21_discard_variable=level21_discard_variable,level20_value_variable=level20_value_variable,level21_value_variable=level21_value_variable)
                if winner[0]==1:
                    win20+=1                    
            results[38,j,k]=n_games
            results[39,j,k]=win20
            results[40,j,k]=100*win20/n_games     
            stop_time=time.time()
            print(f"{n_games} games need {np.round(stop_time-start_time,3)} seconds")
            print(f"level 20 won to {np.round(results[40,j,k],1)} %")
        print(f"level 20 won in average to {np.round(np.mean(results[40,j,:]),1)} %") 
    print("all models done") 
    return results    

#parameters,list of open paremeters, discard parameters, value parameters for level 20
# open base, discard base, value base parameters for level 21
# open steps, discard steps, value steps parameters for level 21
#then optional the parameter to adjust step size of get_new_parameters3
# factor for next base is divided by signifciance
#factor on step when not significant
#factor on step when signifciance divide by signifant
#tolerance of how many sigma the main fit can get larger to be acceptable 
#maximum number of steps to improve base
#max_time the maximum time in hours allowed
#min_win  win in pecentage when it go below then it is aborted early 
def gradient_fit3(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,open_step,discard_step,value_step,n_games=100,max_iter=10,output_name="gradient2_fit1_it",fact_new_par2=1.5,fact_new_step_no2=2.5,fact_new_step_sig2=2.0,tolerance=1.5,max_base_iter=10,max_time=100,min_win=0.):
    #start_time for stopping running when too long, still not working seem to be recreated too often
    full_start_time=time.time()
    #to save parameters and steps 
    results=np.zeros((19,max_iter,2))
    #to save base steps also not used onces, is large created plan is not to use all 3 more to insert different counters
    all_base_results=np.zeros((44,max_iter*100,len(open_vars)))
    #counter of all base models 
    all_base=0
    #first setp does for sure 
    for i in range(max_iter):
        #time to use as delyta time in check 
        if i>0:
            hours=(time.time()-full_start_time)/3600.
            win_percentage=np.mean(base_res[40,:])
            if hours>max_time or win_percentage<=min_win:
                if hours>max_time:
                    print(f"fit ends early because of time limit of {max_time} reached")
                if win_percentage<=min_win:    
                    print(f"fit ends early because win limit of {min_win} reached")
                    #ending with early return 
                return results[:,0:i,:], all_base_results[:,0:all_base,:]   
        #add some end conditions time and already good enough fit 
        print(f"doing iteration {i}")
        print("doing base step")
        #4 times as often to reduce error in this step 
        #counter within model 
        base_it=0
        base_res=run_level21(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,n_games*4)
        all_base_results[0:41,all_base,:]=base_res
        #+1 that unfilled is not relevant number
        all_base_results[41,all_base,:]=i+1
        all_base_results[42,all_base,:]=base_it+1
        all_base_results[43,all_base,:]=all_base+1   
        base_it+=1
        all_base+=1
        #if not first iteration compare with previous iteration
        if i>0:
            #find previous to compare mean one iteration les and maximal base_it
            maxv=0
            index_max=0
            for j in range(all_base+1):
                if all_base_results[41,j,0]==i and all_base_results[42,j,0]>maxv:
                    maxv=all_base_results[42,j,0]
                    index_max=j
            err=100*np.sqrt(np.sum(base_res[38,:])/np.sum(base_res[38,:])**2+np.sum(all_base_results[38,index_max,:])/np.sum(all_base_results[38,index_max,:])**2)
            diff=np.mean(base_res[40,:])-np.mean(all_base_results[40,index_max,:])
            if diff/err>tolerance:
                print(f"win fraction decreased by {diff/err} sigma optimize")
                #at most 10 setsp just for avoiding always running
                while diff/err>tolerance and base_it<max_base_iter:
                    print(f"doing iteration {base_it} for improving base value")
                    #decrease base step size each time by sqrt(2), gradient step size also by the same 
                    new_par1,new_step1=get_new_parameters3(resgrad2,fact_new_par=fact_new_par2/np.sqrt(2)**base_it,fact_new_step_no=fact_new_step_no2/np.sqrt(2)**base_it,fact_new_step_sig=fact_new_step_sig2/np.sqrt(2)**base_it)
                    results[:,i-1,0]=new_par1
                    results[:,i-1,1]=new_step1  
                    #pass to used variables
                    base_open=new_par1[0:6]
                    base_discard=new_par1[6:12]
                    base_value=new_par1[12:19]        
                    #passing as new steps
                    open_step=new_step1[0:6]
                    discard_step=new_step1[6:12]
                    value_step=new_step1[12:19]                    
                    #next base trial
                    base_res=run_level21(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,n_games*4)
                    all_base_results[0:41,all_base,:]=base_res
                    #+1 that unfilled is not relevant number
                    all_base_results[41,all_base,:]=i+1
                    all_base_results[42,all_base,:]=base_it+1
                    all_base_results[43,all_base,:]=all_base+1   
                    base_it+=1
                    all_base+=1
                    maxv=0
                    index_max=0
                    for j in range(all_base+1):
                        if all_base_results[41,j,0]==i and all_base_results[42,j,0]>maxv:
                            maxv=all_base_results[42,j,0]
                            index_max=j
                    err=100*np.sqrt(np.sum(base_res[38,:])/np.sum(base_res[38,:])**2+np.sum(all_base_results[38,index_max,:])/np.sum(all_base_results[38,index_max,:])**2)
                    diff=np.mean(base_res[40,:])-np.mean(all_base_results[40,index_max,:])   
                    if base_it==max_base_iter: 
                        print("maximum base iteration reached")
                        print(f"win fraction decreased by {diff/err}")
                    if diff/err>tolerance:
                        print(f"win fraction decreased by {diff/err} sigma optimize")  
                    else: 
                        print(f"win fraction decreased by {diff/err} sigma acceptable")    
            else: 
                print(f"win fraction decreased by {diff/err} sigma acceptable")
        else:
            print("no comparison done with previous iteration in iteration 0")
        start_time=time.time()
        resgrad1=gradient_step2(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,open_step,discard_step,value_step,n_games=n_games)
        #combine base and steps
        resgrad2=np.zeros((41,20,len(open_vars)))
        resgrad2[:,0,:]=base_res
        resgrad2[:,1:20,:]=resgrad1
        #adding leading zeros
        if i<10:
            np.save(output_name+"00"+str(i)+".npy",resgrad2)
        elif i<100:
            np.save(output_name+"0"+str(i)+".npy",resgrad2)
        else:
            np.save(output_name+str(i)+".npy",resgrad2)            
        stop_time=time.time()
        print(f"{n_games} games need {np.round(stop_time-start_time,3)} seconds")
        #getting new steps and parameters
        new_par1,new_step1=get_new_parameters3(resgrad2,fact_new_par=fact_new_par2,fact_new_step_no=fact_new_step_no2,fact_new_step_sig=fact_new_step_sig2)
        results[:,i,0]=new_par1
        results[:,i,1]=new_step1        
        #passing as new base
        base_open=new_par1[0:6]
        base_discard=new_par1[6:12]
        base_value=new_par1[12:19]        
        #passing as new steps
        open_step=new_step1[0:6]
        discard_step=new_step1[6:12]
        value_step=new_step1[12:19]
    return results, all_base_results[:,0:all_base,:]    


# past fit result,  factor for next base is divided by signifciance
#factor on step when not significant
#factor on step when signifciance divide by signifant
#power factor factor to decrease step for positive 
def get_new_parameters4(result,fact_new_par=1.5,fact_new_step_no=2.5,fact_new_step_sig=2.0,power_incr=1.):
    new_par=np.zeros(19)
    new_steps=np.zeros(19)
    for i in range(1,20):
        err=100*np.sqrt(np.sum(result[38,i,:])/np.sum(result[38,i,:])**2+np.sum(result[38,0,:])/np.sum(result[38,0,:])**2)
        diff=np.mean(result[40,i,:])-np.mean(result[40,0,:])
        #if really bad results or if significance less than 1 just use current base values 
        #changing that it also works for very winning model 
        if np.mean(result[40,i,:])-np.mean(result[40,0,:])>(100-np.mean(result[40,0,:]))*0.5 or abs(diff/err)<1:
            new_par[i-1]=result[18+i,0,0]
            if abs(diff/err)<1:
                #direction cannot be known in thisc case increase it and change sign randomly
                #was 3 before
                new_steps[i-1]=fact_new_step_no*(result[18+i,i,0]-result[18+i,0,0])*np.sign(random.random()-0.5)
            elif np.mean(result[40,i,:])-np.mean(result[40,0,:])<(100-np.mean(result[40,0,:]))*0.75:
                #reverse direction and less
                new_steps[i-1]=-1/3*(result[18+i,i,0]-result[18+i,0,0])
            elif np.mean(result[40,i,:])-np.mean(result[40,0,:])<(100-np.mean(result[40,0,:]))*0.875:
                new_steps[i-1]=-1/5*(result[18+i,i,0]-result[18+i,0,0])                 
            elif np.mean(result[40,i,:])-np.mean(result[40,0,:])<(100-np.mean(result[40,0,:]))*0.94:
                new_steps[i-1]=-1/9*(result[18+i,i,0]-result[18+i,0,0])             
            else:
                new_steps[i-1]=-1/17*(result[18+i,i,0]-result[18+i,0,0])               
        #else use the tried step
        else:
            # just that for negative (good chance found)
            if diff<0:
                new_par[i-1]=result[18+i,0,0]+(-result[18+i,i,0]+result[18+i,0,0])*np.sign(diff)*fact_new_par/abs(diff/err)
                new_steps[i-1]=-fact_new_step_sig*(result[18+i,i,0]-result[18+i,0,0])/(diff/err)
            #potential smaller witha power
            else:
                new_par[i-1]=result[18+i,0,0]+(-result[18+i,i,0]+result[18+i,0,0])*np.sign(diff)*fact_new_par/abs(diff/err)**power_incr 
                #power also here 
                new_steps[i-1]=-fact_new_step_sig*(result[18+i,i,0]-result[18+i,0,0])/(diff/err)**power_incr
    return new_par,new_steps 


#parameters,list of open paremeters, discard parameters, value parameters for level 20
# open base, discard base, value base parameters for level 21
# open steps, discard steps, value steps parameters for level 21
#then optional the parameter to adjust step size of get_new_parameters4
# factor for next base is divided by signifciance
#factor on step when not significant
#factor on step when signifciance divide by signifant
#tolerance of how many sigma the main fit can get larger to be acceptable 
#maximum number of steps to improve base
#max_time the maximum time in hours allowed
#min_win  win in pecentage when it go below then it is aborted early 
#power_incr 
def gradient_fit4(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,open_step,discard_step,value_step,n_games=100,max_iter=10,output_name="gradient2_fit1_it",fact_new_par2=1.5,fact_new_step_no2=2.5,fact_new_step_sig2=2.0,tolerance_one=1.5,tolerance_later=1.5,max_base_iter=10,max_time=100,min_win=0.,power_incr=1.):
    #start_time for stopping running when too long, still not working seem to be recreated too often
    full_start_time=time.time()
    #to save parameters and steps 
    results=np.zeros((19,max_iter,2))
    #to save base steps also not used onces, is large created plan is not to use all 3 more to insert different counters
    all_base_results=np.zeros((44,max_iter*100,len(open_vars)))
    #counter of all base models 
    all_base=0
    #first setp does for sure 
    for i in range(max_iter):
        #time to use as delyta time in check 
        if i>0:
            hours=(time.time()-full_start_time)/3600.
            print(f"ran for {np.round(hours,3)} hours")
            win_percentage=np.mean(base_res[40,:])
            if hours>max_time or win_percentage<=min_win:
                if hours>max_time:
                    print(f"fit ends early because of time limit of {max_time} reached")
                if win_percentage<=min_win:    
                    print(f"fit ends early because win limit of {min_win} reached")
                    #ending with early return 
                return results[:,0:i,:], all_base_results[:,0:all_base,:]   
        #add some end conditions time and already good enough fit 
        print(f"doing iteration {i}")
        print("doing base step")
        #4 times as often to reduce error in this step 
        #counter within model 
        base_it=0
        base_res=run_level21(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,n_games*4)
        all_base_results[0:41,all_base,:]=base_res
        #+1 that unfilled is not relevant number
        all_base_results[41,all_base,:]=i+1
        all_base_results[42,all_base,:]=base_it+1
        all_base_results[43,all_base,:]=all_base+1   
        base_it+=1
        all_base+=1
        #if not first iteration compare with previous iteration
        if i>0:
            #find previous to compare mean one iteration les and maximal base_it
            maxv=0
            index_max=0
            for j in range(all_base+1):
                if all_base_results[41,j,0]==i and all_base_results[42,j,0]>maxv:
                    maxv=all_base_results[42,j,0]
                    index_max=j
            err=100*np.sqrt(np.sum(base_res[38,:])/np.sum(base_res[38,:])**2+np.sum(all_base_results[38,index_max,:])/np.sum(all_base_results[38,index_max,:])**2)
            diff=np.mean(base_res[40,:])-np.mean(all_base_results[40,index_max,:])
            #tolerance one here can be less strict 
            if diff/err>tolerance_one:
                print(f"win fraction decreased by {diff/err} sigma optimize")
                #at most 10 setsp just for avoiding always running
                while diff/err>tolerance_later and base_it<max_base_iter:
                    print(f"doing iteration {base_it} for improving base value")
                    #decrease base step size each time by sqrt(2), gradient step size also by the same 
                    new_par1,new_step1=get_new_parameters4(resgrad2,fact_new_par=fact_new_par2/np.sqrt(2)**base_it,fact_new_step_no=fact_new_step_no2/np.sqrt(2)**base_it,fact_new_step_sig=fact_new_step_sig2/np.sqrt(2)**base_it,power_incr=power_incr)
                    results[:,i-1,0]=new_par1
                    results[:,i-1,1]=new_step1  
                    #pass to used variables
                    base_open=new_par1[0:6]
                    base_discard=new_par1[6:12]
                    base_value=new_par1[12:19]        
                    #passing as new steps
                    open_step=new_step1[0:6]
                    discard_step=new_step1[6:12]
                    value_step=new_step1[12:19]                    
                    #next base trial
                    base_res=run_level21(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,n_games*4)
                    all_base_results[0:41,all_base,:]=base_res
                    #+1 that unfilled is not relevant number
                    all_base_results[41,all_base,:]=i+1
                    all_base_results[42,all_base,:]=base_it+1
                    all_base_results[43,all_base,:]=all_base+1   
                    base_it+=1
                    all_base+=1
                    maxv=0
                    index_max=0
                    for j in range(all_base+1):
                        if all_base_results[41,j,0]==i and all_base_results[42,j,0]>maxv:
                            maxv=all_base_results[42,j,0]
                            index_max=j
                    err=100*np.sqrt(np.sum(base_res[38,:])/np.sum(base_res[38,:])**2+np.sum(all_base_results[38,index_max,:])/np.sum(all_base_results[38,index_max,:])**2)
                    diff=np.mean(base_res[40,:])-np.mean(all_base_results[40,index_max,:])   
                    if base_it==max_base_iter: 
                        print("maximum base iteration reached")
                        print(f"win fraction decreased by {diff/err}")
                    if diff/err>tolerance_later:
                        print(f"win fraction decreased by {diff/err} sigma optimize")  
                    else: 
                        print(f"win fraction decreased by {diff/err} sigma acceptable")    
            else: 
                print(f"win fraction decreased by {diff/err} sigma acceptable")
        else:
            print("no comparison done with previous iteration in iteration 0")
        start_time=time.time()
        resgrad1=gradient_step2(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,open_step,discard_step,value_step,n_games=n_games)
        #combine base and steps
        resgrad2=np.zeros((41,20,len(open_vars)))
        resgrad2[:,0,:]=base_res
        resgrad2[:,1:20,:]=resgrad1
        #adding leading zeros
        if i<10:
            np.save(output_name+"00"+str(i)+".npy",resgrad2)
        elif i<100:
            np.save(output_name+"0"+str(i)+".npy",resgrad2)
        else:
            np.save(output_name+str(i)+".npy",resgrad2)            
        stop_time=time.time()
        print(f"{n_games} games need {np.round(stop_time-start_time,3)} seconds")
        #getting new steps and parameters
        new_par1,new_step1=get_new_parameters4(resgrad2,fact_new_par=fact_new_par2,fact_new_step_no=fact_new_step_no2,fact_new_step_sig=fact_new_step_sig2,power_incr=power_incr)
        results[:,i,0]=new_par1
        results[:,i,1]=new_step1        
        #passing as new base
        base_open=new_par1[0:6]
        base_discard=new_par1[6:12]
        base_value=new_par1[12:19]        
        #passing as new steps
        open_step=new_step1[0:6]
        discard_step=new_step1[6:12]
        value_step=new_step1[12:19]
    return results, all_base_results[:,0:all_base,:]    



#does Monte carlo against a number of cases
#open_vars and co and and are lists of the variables to be used 
#open_ranges and discard_ranges are 2,6 np arrays, value_ranges is 2,7 np array, 
#reliazations are number of models tried in one, trials are number of MC models for level 21
#wfrac when it aborts early
#wfrac2 performance of worst case .
#here mean and gauusian error araound it
#save means how much output is save in False less is saved 
def montecarlo_trials3(open_vars,discard_vars,value_vars,open_mean,discard_mean,value_mean,open_std,discard_std,value_std,realizations,trials,wfrac=0.10,wfrac2=0.551,tot_score_collect=False):
    n_it=realizations
    if tot_score_collect==False:
        results=np.zeros((41,trials,len(open_vars)))
    else:
        results=np.zeros((42,trials,len(open_vars)))     
    for j in range(trials):
        #only needed for when score is collected 
        tot_scores=np.zeros((2))
        print(f"doing Monte Carlo {j} for level 21")
        #create random values for level 21 within the ranges, save are used for all level 20 models
        level21_open_variable=np.zeros((6))
        level21_discard_variable=np.zeros((6))
        level21_value_variable=np.zeros((7))
        for k in range(7):
            level21_value_variable[k]=random.gauss(value_mean[k],value_std[k])
            if k<6:
                level21_open_variable[k]=random.gauss(open_mean[k],open_std[k]) 
                level21_discard_variable[k]=random.gauss(discard_mean[k],discard_std[k]) 
        for k in range(len(open_vars)):
            print(f"trying case {k} for level 20")
            level20_open_variable=np.array(open_vars[k])
            level20_discard_variable=np.array(discard_vars[k])
            level20_value_variable=np.array(value_vars[k])             
            results[0:6,j,k]=level20_open_variable
            results[6:12,j,k]=level20_discard_variable
            results[12:19,j,k]=level20_value_variable    
            results[19:25,j,k]=level21_open_variable
            results[25:31,j,k]=level21_discard_variable
            results[31:38,j,k]=level21_value_variable         
            win20=0
            it_counter=0
            start_time=time.time()
            n_it1=n_it
            if n_it1>22:
                n_it1=22
            #22 always     
            for i in range(n_it1):
                names=['alpha','beta']
                nature=['computer','computer']
                levels=[20,21]
                it_counter+=1
                if tot_score_collect==False:
                    winner=skyjo_game(names,nature,levels,0,True,tot_score_collect=False,level20_open_variable=level20_open_variable,level21_open_variable=level21_open_variable,level20_discard_variable=level20_discard_variable,level21_discard_variable=level21_discard_variable,level20_value_variable=level20_value_variable,level21_value_variable=level21_value_variable)
                else:
                    winner,tot_score=skyjo_game(names,nature,levels,0,True,tot_score_collect=True,level20_open_variable=level20_open_variable,level21_open_variable=level21_open_variable,level20_discard_variable=level20_discard_variable,level21_discard_variable=level21_discard_variable,level20_value_variable=level20_value_variable,level21_value_variable=level21_value_variable)   
                    tot_scores+=tot_score
                if winner[0]==1:
                    win20+=1
            #98% ownside win conditions tested here   2.33 sigma   stops early when new model clearly bad or good 
            while it_counter<n_it and abs((win20-it_counter/2)/np.sqrt(it_counter))<2.33:  
                names=['alpha','beta']
                nature=['computer','computer']
                levels=[20,21]
                it_counter+=1
                if tot_score_collect==False:
                    winner=skyjo_game(names,nature,levels,0,True,tot_score_collect=False,level20_open_variable=level20_open_variable,level21_open_variable=level21_open_variable,level20_discard_variable=level20_discard_variable,level21_discard_variable=level21_discard_variable,level20_value_variable=level20_value_variable,level21_value_variable=level21_value_variable)
                else:
                    winner,tot_score=skyjo_game(names,nature,levels,0,True,tot_score_collect=True,level20_open_variable=level20_open_variable,level21_open_variable=level21_open_variable,level20_discard_variable=level20_discard_variable,level21_discard_variable=level21_discard_variable,level20_value_variable=level20_value_variable,level21_value_variable=level21_value_variable)   
                    tot_scores+=tot_score                    
                if winner[0]==1:
                    win20+=1                
            #now checking whether 98% sigficant on bad performance        
            results[38,j,k]=it_counter
            results[39,j,k]=win20
            results[40,j,k]=100*win20/it_counter    
            if tot_score_collect==True:
                results[41,j,k]=(tot_scores[0]-tot_scores[1])/it_counter
            stop_time=time.time()
            print(f"{it_counter} games need {np.round(stop_time-start_time,3)} seconds")
            print(f"level 20 won to {np.round(results[40,j,k],1)} %")
        print(f"level 20 won in average to {np.round(np.mean(results[40,j,:]),1)} %") 
        if np.mean(results[40,j,:]/100)<wfrac and np.max(results[40,j,:]/100)<wfrac2:
            print("better model found")
            if j==0:
                return True, results[:,j,:], results[:,j,:]
            else:
                #give back fit converged or not, best models, all models 
                return True, results[:,j,:], results[:,0:j+1,:]
    print("no better model found") 
    #second parameter is noen will not be used later
    return False, None, results 


#now real gradient NotImplement
# past fit result,  factor for next base is divided by signifciance
#factor on step when not significant
#border step below which step is increased somewhat
#power factor factor to decrease step for positive 
#min_sigma minium required significance to care 
#alpha factor of step, alpha 1 means euqal to largest test step (right direction( is done 
def get_new_parameters5(result,fact_new_step_no=2.5,border_sigma_step=2.0,power_incr=1.0,min_sigma=1.0,alpha=1.0):
    new_par=np.zeros(19)
    new_steps=np.zeros(19)
    #collect significant differences which are then considered or not 
    diffs=np.zeros(20)
    ndiffs=np.zeros(20)
    for i in range(1,20):
        #apply get error function
        err=get_error_binary2(result[40,0,:],result[40,i,:],result[38,0,:],result[38,i,:]
        diff=np.mean(result[40,i,:])-np.mean(result[40,0,:])
        ndiffs[i]=diff/err
        #if really bad results or if significance less than 1 just use current base values 
        #changing that it also works for very winning model 
        if np.mean(result[40,i,:])-np.mean(result[40,0,:])>(100-np.mean(result[40,0,:]))*0.5 or abs(diff/err)<min_sigma:
            new_par[i-1]=result[18+i,0,0]
            diffs[i]=0
            if abs(diff/err)<min_sigma:
                #direction cannot be known in thisc case increase it and change sign randomly
                #was 3 before
                new_steps[i-1]=fact_new_step_no*(result[18+i,i,0]-result[18+i,0,0])*np.sign(random.random()-0.5)
            elif np.mean(result[40,i,:])-np.mean(result[40,0,:])<(100-np.mean(result[40,0,:]))*0.75:
                #reverse direction and less
                new_steps[i-1]=-1/3*(result[18+i,i,0]-result[18+i,0,0])
            elif np.mean(result[40,i,:])-np.mean(result[40,0,:])<(100-np.mean(result[40,0,:]))*0.875:
                new_steps[i-1]=-1/5*(result[18+i,i,0]-result[18+i,0,0])                 
            elif np.mean(result[40,i,:])-np.mean(result[40,0,:])<(100-np.mean(result[40,0,:]))*0.94:
                new_steps[i-1]=-1/9*(result[18+i,i,0]-result[18+i,0,0])             
            else:
                new_steps[i-1]=-1/17*(result[18+i,i,0]-result[18+i,0,0])               
        #else use the tried step
        else:
            # if good trend found just collect the gradients 
            if diff<0:
                diffs[i]=diff/err
            #potential smaller witha power
            else:
                diffs[i]=(diff/err)**(1/power_incr)
    print(ndiffs)            
    #do now all which are have significant and goood chance
    print(diffs)
    for i in range(1,20):  
        if diffs[i]!=0:
            if diffs[i]<0:
                #now real gradient step made small with factor relative to largest 
                new_par[i-1]=result[18+i,0,0]+(-result[18+i,i,0]+result[18+i,0,0])*np.sign(diff)*alpha*diffs[i]/np.max(np.abs(diffs))
                #if step small increase
                if np.abs(border_sigma_step/diffs[i])<border_sigma_step:
                    new_steps[i-1]=-border_sigma_step/diffs[i]*(result[18+i,i,0]-result[18+i,0,0])
                #otherwise just keep
                else:
                    new_steps[i-1]=(result[18+i,i,0]-result[18+i,0,0])
            else:
                #for base potential smaller with a power
                new_par[i-1]=result[18+i,0,0]+(-result[18+i,i,0]+result[18+i,0,0])*np.sign(diff)*alpha*diffs[i]/np.max(np.abs(diffs))
                #here continous decrease because more likely too large otherwise 
                new_steps[i-1]=-border_sigma_step*(result[18+i,i,0]-result[18+i,0,0])/diffs[i]**power_incr            
    return new_par,new_steps 


#parameters,list of open paremeters, discard parameters, value parameters for level 20
# open base, discard base, value base parameters for level 21
# open steps, discard steps, value steps parameters for level 21
#then optional the parameter to adjust step size of get_new_parameters5
# factor relavive to maximum step explored
#factor on step when not significant
#factor on step when signifciance is borderline
#minium sigma required to care 
#tolerance of how many sigma the main fit can get larger to be acceptable 
##power_incr hiow doen weight on negative changes 
#maximum number of steps to improve base
#max_time the maximum time in hours allowed
#min_win  win in pecentage when it go below then it is aborted early 
#error version for base
#restarted fit (true or false)
#give old results in old_results and old_base 
def gradient_fit5(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,open_step,discard_step,value_step,n_games=100,max_iter=10,output_name="gradient5_fit1_it",alpha2=1.0,fact_new_step_no2=2.5,border_sigma_step2=2.0,min_sigma2=1.0,power_incr2=1.,tolerance_one=1.5,tolerance_later=1.5,max_base_iter=10,max_time=100,min_win=0.,error_version_base=1,restart=False,old_result=None,old_base=None):
    #start_time for stopping running when too long, still not working seem to be recreated too often
    full_start_time=time.time()
    #if new fit and no restart
    if restart==False:
        #to save parameters and steps 
        results=np.zeros((19,max_iter,2))
        #to save base steps also not used onces, is large created plan is not to use all 3 more to insert different counters
        all_base_results=np.zeros((44,max_iter*100,len(open_vars)))
        $counter here yero 
        start_i=0
        all_base=0 
    else:
        #here previous results are filled in first 
        results=np.zeros((19,max_iter,2))
        results[:,old_result.shape[1],:]=old_result
        #to save base steps also not used onces, is large created plan is not to use all 3 more to insert different counters
        all_base_results=np.zeros((44,max_iter*100,len(open_vars)))
        all_base_results[:,old_base.shape[1],:]=old_base
        start_i=old_result.shape[1]
        all_base=old_base.shape[1]
        

    #first setp does for sure 
    for i in range(start_i,max_iter):
        #time to use as delyta time in check 
        if i>0:
            hours=(time.time()-full_start_time)/3600.
            print(f"ran for {np.round(hours,3)} hours")
            win_percentage=np.mean(base_res[40,:])
            if hours>max_time or win_percentage<=min_win:
                if hours>max_time:
                    print(f"fit ends early because of time limit of {max_time} reached")
                if win_percentage<=min_win:    
                    print(f"fit ends early because win limit of {min_win} reached")
                    #ending with early return 
                return results[:,0:i,:], all_base_results[:,0:all_base,:]   
        #add some end conditions time and already good enough fit 
        print(f"doing iteration {i}")
        print("doing base step")
        #4 times as often to reduce error in this step 
        #counter within model 
        base_it=0
        base_res=run_level21(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,n_games*4)
        all_base_results[0:41,all_base,:]=base_res
        #+1 that unfilled is not relevant number
        all_base_results[41,all_base,:]=i+1
        all_base_results[42,all_base,:]=base_it+1
        all_base_results[43,all_base,:]=all_base+1   
        base_it+=1
        all_base+=1
        #if not first iteration compare with previous iteration
        if i>0:
            #find previous to compare mean one iteration les and maximal base_it
            maxv=0
            index_max=0
            for j in range(all_base+1):
                if all_base_results[41,j,0]==i and all_base_results[42,j,0]>maxv:
                    maxv=all_base_results[42,j,0]
                    index_max=j
            #also using error function here         
            err=get_error_binary2(base_res[40,:],all_base_results[40,index_max,:],base_res[38,:],all_base_results[38,index_max,:],version=error_version_base)
            diff=np.mean(base_res[40,:])-np.mean(all_base_results[40,index_max,:])
            #tolerance one here can be less strict 
            if diff/err>tolerance_one:
                print(f"win fraction decreased by {diff/err} sigma optimize")
                #at most 10 setsp just for avoiding always running
                while diff/err>tolerance_later and base_it<max_base_iter:
                    print(f"doing iteration {base_it} for improving base value")
                    #decrease base step size each time by sqrt(2)
                    new_par1,new_step1=get_new_parameters5(resgrad2,fact_new_step_no=fact_new_step_no2,border_sigma_step=border_sigma_step2,power_incr=power_incr2,min_sigma=min_sigma2,alpha=alpha2/np.sqrt(2)**base_it)
                    results[:,i-1,0]=new_par1
                    results[:,i-1,1]=new_step1  
                    #pass to used variables
                    base_open=new_par1[0:6]
                    base_discard=new_par1[6:12]
                    base_value=new_par1[12:19]        
                    #passing as new steps
                    open_step=new_step1[0:6]
                    discard_step=new_step1[6:12]
                    value_step=new_step1[12:19]                    
                    #next base trial
                    base_res=run_level21(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,n_games*4)
                    all_base_results[0:41,all_base,:]=base_res
                    #+1 that unfilled is not relevant number
                    all_base_results[41,all_base,:]=i+1
                    all_base_results[42,all_base,:]=base_it+1
                    all_base_results[43,all_base,:]=all_base+1   
                    base_it+=1
                    all_base+=1
                    maxv=0
                    index_max=0
                    for j in range(all_base+1):
                        if all_base_results[41,j,0]==i and all_base_results[42,j,0]>maxv:
                            maxv=all_base_results[42,j,0]
                            index_max=j
                    err=100*np.sqrt(np.sum(base_res[38,:])/np.sum(base_res[38,:])**2+np.sum(all_base_results[38,index_max,:])/np.sum(all_base_results[38,index_max,:])**2)
                    diff=np.mean(base_res[40,:])-np.mean(all_base_results[40,index_max,:])   
                    if base_it==max_base_iter: 
                        print("maximum base iteration reached")
                        print(f"win fraction decreased by {diff/err}")
                    if diff/err>tolerance_later:
                        print(f"win fraction decreased by {diff/err} sigma optimize")  
                    else: 
                        print(f"win fraction decreased by {diff/err} sigma acceptable")    
            else: 
                print(f"win fraction decreased by {diff/err} sigma acceptable")
        else:
            print("no comparison done with previous iteration in iteration 0")
        start_time=time.time()
        resgrad1=gradient_step2(open_vars,discard_vars,value_vars,base_open,base_discard,base_value,open_step,discard_step,value_step,n_games=n_games)
        #combine base and steps
        resgrad2=np.zeros((41,20,len(open_vars)))
        resgrad2[:,0,:]=base_res
        resgrad2[:,1:20,:]=resgrad1
        #adding leading zeros
        if i<10:
            np.save(output_name+"00"+str(i)+".npy",resgrad2)
        elif i<100:
            np.save(output_name+"0"+str(i)+".npy",resgrad2)
        else:
            np.save(output_name+str(i)+".npy",resgrad2)            
        stop_time=time.time()
        print(f"{n_games} games need {np.round(stop_time-start_time,3)} seconds")
        #getting new steps and parameters
        new_par1,new_step1=get_new_parameters5(resgrad2,fact_new_step_no=fact_new_step_no2,border_sigma_step=border_sigma_step2,power_incr=power_incr2,min_sigma=min_sigma2,alpha=alpha2)
        results[:,i,0]=new_par1
        results[:,i,1]=new_step1        
        #passing as new base
        base_open=new_par1[0:6]
        base_discard=new_par1[6:12]
        base_value=new_par1[12:19]        
        #passing as new steps
        open_step=new_step1[0:6]
        discard_step=new_step1[6:12]
        value_step=new_step1[12:19]
    return results, all_base_results[:,0:all_base,:]    

#eror for binary data 2 values added together, small n effects not considered well
#has version 1 is mostly gaussian
#0 is too large errors by using 50% assumption
def get_error_binary2(win_main,win_other,games_main,games_other,version=1):
    if version==1:
        p1=np.mean(win_main)/100
        p2=np.mean(win_other)/100
        n1=np.sum(games_main)
        n2=np.sum(games_other)     
        #if zero or one adds/subtract one is not perfect likely but for now 
        if p1==1:
            p1=1-1/n1
        elif p1==0:
            p1=1/n1
        if p2==1:
            p2=1-1/n2
        elif p2==0:
            p2=1/n2            
        err=100*np.sqrt(p1*(1-p1)/n1+p2*(1-p2)/n2)
        return err
    elif version==0:
        err=100*np.sqrt(np.sum(games_main)/games_main)**2+np.sum(games_other)/np.sum(games_other)**2)
        return err
     
