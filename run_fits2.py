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


gra1=np.loadtxt("gradient_level20_v1.txt")
gra2=np.loadtxt("gradient_level20_v2.txt")
gra3=np.loadtxt("gradient_level20_v3.txt")
gra4=np.loadtxt("gradient_level20_v4.txt")
gra5=np.loadtxt("gradient_level20_v5.txt")
gra6=np.loadtxt("gradient_level20_v6.txt")
gra7=np.loadtxt("gradient_level20_v7.txt")
gra8=np.loadtxt("gradient_level20_v8.txt")
gra9=np.loadtxt("gradient_level20_v9.txt")
gra10=np.loadtxt("gradient_level20_v10.txt")
gra11=np.loadtxt("gradient_level20_v11.txt")
gra12=np.loadtxt("gradient_level20_v12.txt")
gra13=np.loadtxt("gradient_level20_v13.txt")
gra14=np.loadtxt("gradient_level20_v14.txt")
compp=np.zeros((20,14))
compp[0,:]=np.array([1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002,0.001,0.0005,0.0002,0.0001,0.00005])
compp[1:20,0]=gra1[39,:]/gra1[38,:]
compp[1:20,1]=gra2[39,:]/gra2[38,:]
compp[1:20,2]=gra3[39,:]/gra3[38,:]
compp[1:20,3]=gra4[39,:]/gra4[38,:]
compp[1:20,4]=gra5[39,:]/gra5[38,:]
compp[1:20,5]=gra6[39,:]/gra6[38,:]
compp[1:20,6]=gra7[39,:]/gra7[38,:]
compp[1:20,7]=gra8[39,:]/gra8[38,:]
compp[1:20,8]=gra9[39,:]/gra9[38,:]
compp[1:20,9]=gra10[39,:]/gra10[38,:]
compp[1:20,10]=gra11[39,:]/gra11[38,:]
compp[1:20,11]=gra12[39,:]/gra12[38,:]
compp[1:20,12]=gra13[39,:]/gra13[38,:]
compp[1:20,13]=gra14[39,:]/gra14[38,:]
gram1=np.loadtxt("gradient_level20_vm0.txt")
gram2=np.loadtxt("gradient_level20_vm1.txt")
gram3=np.loadtxt("gradient_level20_vm2.txt")
gram4=np.loadtxt("gradient_level20_vm3.txt")
gram5=np.loadtxt("gradient_level20_vm4.txt")
gram6=np.loadtxt("gradient_level20_vm5.txt")
gram7=np.loadtxt("gradient_level20_vm6.txt")
gram8=np.loadtxt("gradient_level20_vm7.txt")
gram9=np.loadtxt("gradient_level20_vm8.txt")
gram10=np.loadtxt("gradient_level20_vm9.txt")
gram11=np.loadtxt("gradient_level20_vm10.txt")
compm=np.zeros((20,11))
compm[0,:]=np.array([0.00005,0.0001,0.0002,0.0005,0.001,0.002,0.005,0.01,0.02,0.05,0.1])
compm[1:20,0]=gram1[39,:]/gram1[38,:]
compm[1:20,1]=gram2[39,:]/gram2[38,:]
compm[1:20,2]=gram3[39,:]/gram3[38,:]
compm[1:20,3]=gram4[39,:]/gram4[38,:]
compm[1:20,4]=gram5[39,:]/gram5[38,:]
compm[1:20,5]=gram6[39,:]/gram6[38,:]
compm[1:20,6]=gram7[39,:]/gram7[38,:]
compm[1:20,7]=gram8[39,:]/gram8[38,:]
compm[1:20,8]=gram9[39,:]/gram9[38,:]
compm[1:20,9]=gram10[39,:]/gram10[38,:]
compm[1:20,10]=gram11[39,:]/gram11[38,:]
compa=np.zeros((compm.shape[0],compm.shape[1]+compp.shape[1]))
compm=np.flip(compm,1)
compp=np.flip(compp,1)
compa[0,0:compm.shape[1]]=-compm[0]
compa[1:20,0:compm.shape[1]]=compm[1:20]
compa[:,compm.shape[1]:compa.shape[1]]=compp
#using logistic fit result as inout
logpar=np.zeros((4,19))
ares=np.zeros((19,compa.shape[1]))
for i in range(19):
    popt, pcov = curve_fit(logistic_adapted, compa[0,:], 100*compa[1+i,:])
    logpar[0:3,i]=popt
    fres=np.zeros((compa.shape[1]))
    for j in range(ares.shape[0]):
        fres[j]=logistic_adapted(compa[0,j],popt[0],popt[1],popt[2])  
    ares[i,:]=fres
#get change sclae
logpar[3]=1.1/logpar[1]  
#loading models

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

    if np.mean(mcg2[40,:,:],0)[i]>30.5:
        sel_new[:,c]=mcg2[0:19,0,i]
        c+=1

list_open8=[]
list_discard8=[]
list_value8=[]
for i in range(5):
    list_open8.append(sel_new[0:6,i])
    list_discard8.append(sel_new[6:12,i])
    list_value8.append(sel_new[12:19,i])  
        
open_steps2=logpar[3,0:6]
print("used steps are")
print(open_steps2)
discard_steps2=logpar[3,6:12]
print(discard_steps2)
value_steps2=logpar[3,12:19]
print(value_steps2)

start_time=time.time()
#seeting up the fit 
#number games per iteration
n_games=100
#number of iterations
max_iter=50
output_name="gradient5_fit2_it"
#tolerance for getting worse in step (sigma) is first 
tolerance_one=1.75
#for stopping is
tolerance_later=0.25
#running time hours, only checked at that of iteration 
max_time=14 
#minium win percent other stop conditions
min_win=20
#reduces step for steps which lead to negative win change 
power_incr=1.75 #smaller since negative changes seem less likely succcesul
#iteration number to avoid decrease 
max_base_iter=30
#factor increase when not significant, now smaller 
fact_new_step_no=1.5
#range in which step size increases
border_sigma_step=3.0 #larger because divded by sigma 
#minium required step szie to care 
min_sigma=2.0 #now a little larger than 50% chnage probablity  
#base step size compaer to larged in step
alpha=1.0
fac=1.0
#using first of the good models 
gradient_res,gradient_allres=gradient_fit5(list_open8,list_discard8,list_value8,np.zeros((6)),np.zeros((6)),np.zeros((7)),fac*open_steps2,fac*discard_steps2,fac*value_steps2,n_games=n_games,max_iter=max_iter,output_name=output_name,tolerance_one=tolerance_one,tolerance_later=tolerance_later,max_time=max_time,min_win=min_win,power_incr2=power_incr,fact_new_step_no2=fact_new_step_no,max_base_iter=max_base_iter,min_sigma2=min_sigma,alpha2=alpha,border_sigma_step2=border_sigma_step)
#saving output 
np.save('gradient5_fit2.npy',gradient_res)
np.save('gradient5_fit2_all.npy',gradient_allres)
stop_time=time.time()
print(f"Needed {np.round(stop_time-start_time,3)} seconds") 
#now sigma corrected and some power increase
#5_1 is n_games=100,max_iter=50,tolerance_one=1.75,tolerance_later=0.25,max_time=5,min_win=20,power_incr=1.5, ,max_base_iter=30,fact_new_step_no=2.5,border_sigma_step=2.0,min_sigma=1.9,alpha=1.0,fac=1.0 

