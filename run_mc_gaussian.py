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
c=0
list_open7=[]
list_discard7=[]
list_value7=[]
#selecting models
for i in range(850):
    if np.mean(allres2[40,i,4:6])<45 and np.max(allres2[40,i,:])<60:
        list_open7.append(allres2[19:25,i,0])
        list_discard7.append(allres2[25:31,i,0])        
        list_value7.append(allres2[31:38,i,0]) 
        c+=1
print(f"{c} improved models used")        
       

#best of gradient 4_1 seems not randomly good 
set1b=np.load("testbest32a_v2.npy")
set2b=np.load("testbest32b_v2.npy")
selected_models=np.zeros((41,6,11))
selected_models[:,:,0:5]=set1b[:,:,0:5]
selected_models[:,:,5:11]=set2b[:,:,5:11]

print(selected_models.shape)
good_models=selected_models[19:38,0,:]
mcg1=np.load("mcg_v1_all.npy")
sel_modelg=np.zeros((19,23))
csel=0
for i in range(mcg1.shape[1]):
    if np.mean(mcg1[40,i,:])<28.0:
        print(np.mean(mcg1[40,i,:]))
        sel_modelg[:,csel]=mcg1[19:38,i,0]
        csel+=1
print(csel)        
#factor to adjust scatter since native is too little 
factor=8
start_time=time.time()
#best is 6
x=6
#now 11
x=11
print(sel_modelg[:,11])
tot_score_output=True

boolean,bestres,allres=montecarlo_trials3(list_open7,list_discard7,list_value7,sel_modelg[0:6,x],sel_modelg[6:12,x],sel_modelg[12:19,x],factor*logpar[3,0:6],factor*logpar[3,6:12],factor*logpar[3,12:19],50,5,wfrac=0.001,wfrac2=0.002,tot_score_collect=tot_score_output)
#v1 was factor of 12 and using good_model 11
np.save("mcg_v3_all.npy",allres)
stop_time=time.time()
print(f"ran for {np.round(stop_time-start_time,2)} seconds")

print(f"mean win is {np.round(np.mean(np.mean(allres[40,:,:],1)),3)}  %")
print(f"win standard deviation is {np.round(np.std(np.mean(allres[40,:,:],1)),3)} %")
if allres.shape[0]==42:
    print("also total score collected")
    print(np.mean(allres[40,:,:],1))
    print(np.mean(allres[41,:,:],1))
    print(f"mean score differential is {np.round(np.mean(np.mean(allres[41,:,:],1)),3)}  points")
    print(f"score differential standard deviation is {np.round(np.std(np.mean(allres[41,:,:],1)),3)} points")
#factor 2 was std of 3 
#factor 4 was std of 3 mean of 34
#factor 10 was std of  10, mean of 35 
#factor 20 was std of 7 mean of 41

#5 50 wo output collection 372 seconds 
