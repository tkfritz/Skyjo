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
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
import os
#skyjo game classes and functions 
from skyjo_functions4 import *
#own functions for machine learning
from ml_functions2 import *
#for plotting 
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
import seaborn as sns

#for confidence intervalls
from scipy.stats import beta
#for fitting of x y data 
from scipy.optimize import curve_fit
#for splitting
from sklearn.model_selection import train_test_split
#confusing matrix
from sklearn.metrics import confusion_matrix
#for saving and loading of stranger object
import pickle
from fit_functions import *

#script to restart fits 

#loading models to compare with
allres2=np.load("mc_v9_all.npy")      
moreruns3a=np.load('testbest26a_v2.npy')
moreruns3b=np.load('testbest26b_v2.npy')
moreruns3c=np.load('testbest26c_v2.npy')
moreruns3=np.zeros((41,6,25))
moreruns3[:,:,0:8]=moreruns3a[:,:,0:8]
moreruns3[:,:,8:16]=moreruns3b[:,:,8:16]
moreruns3[:,:,16:25]=moreruns3c[:,:,16:25]
xmin=np.argmin(np.mean(moreruns3[40,:,:],0))

sel_new=np.zeros((19,5))
sel_new[:,0]=moreruns3[19:38,0,xmin]
mcg2=np.load("mcg_v2_all.npy")

c=1
for i in range(6):
    #is greater because the best result the one from moreruns3
    if np.mean(mcg2[40,:,:],0)[i]>30.5:
        sel_new[:,c]=mcg2[0:19,0,i]
        c+=1
print(c)        

list_open8=[]
list_discard8=[]
list_value8=[]
for i in range(5):
    list_open8.append(sel_new[0:6,i])
    list_discard8.append(sel_new[6:12,i])
    list_value8.append(sel_new[12:19,i])  
        


start_time=time.time()
#same parameters as 5_1 
#seeting up the fit 
#number games per iteration
n_games=100
#number of iterations
max_iter=60
#slightly changed name 
output_name="gradient5_fit1b_it"
#tolerance for getting worse in step (sigma) is first 
tolerance_one=1.75
#for stopping is
tolerance_later=0.25
#running time hours, only checked at that of iteration 
max_time=9
#minium win percent other stop conditions
min_win=20
#reduces step for steps which lead to negative win change 
power_incr=1.5 #smaller since negative changes seem less likely succcesul
#iteration number to avoid decrease 
max_base_iter=30
#factor increase when not significant, now smaller 
fact_new_step_no=2.5
#range in which step size increases
border_sigma_step=2.0 #larger because divded by sigma 
#minium required step szie to care 
min_sigma=1.9 #now a little larger than 50% chnage probablity  
#base step size compaer to larged in step
alpha=1.0
fac=2.0 #step reduced by factor 2 because produce to large value /2 did do nearly nothing (7.937316927022633->7.700146102510005) because of power? 
error_version_base2=0
#old iteration for restart 
old_result=np.load('gradient5_fit1.npy')

old_base=np.load('gradient5_fit1_all.npy')
#old_result contain main and gradient step to be used 

last_it2=np.load('gradient5_fit1_it007.npy')
        
gradient_res,gradient_allres=gradient_fit5(list_open8,list_discard8,list_value8,old_result[0:6,old_result.shape[1]-1,0],old_result[6:12,old_result.shape[1]-1,0],old_result[12:19,old_result.shape[1]-1,0],old_result[0:6,old_result.shape[1]-1,1]/fac,old_result[6:12,old_result.shape[1]-1,1]/fac,old_result[12:19,old_result.shape[1]-1,1]/fac,n_games=n_games,max_iter=max_iter,output_name=output_name,tolerance_one=tolerance_one,tolerance_later=tolerance_later,max_time=max_time,min_win=min_win,power_incr2=power_incr,fact_new_step_no2=fact_new_step_no,max_base_iter=max_base_iter,min_sigma2=min_sigma,alpha2=alpha,border_sigma_step2=border_sigma_step,error_version_base=error_version_base2,old_result=old_result,old_base=old_base,restart=True,last_it=last_it2)
#saving output 
np.save('gradient5_fit1b.npy',gradient_res)
np.save('gradient5_fit1b_all.npy',gradient_allres)
stop_time=time.time()
print(f"Needed {np.round(stop_time-start_time,3)} seconds") 
