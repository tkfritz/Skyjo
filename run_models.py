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
gradfit15=np.load('gradient_fit1_it15.npy')
myPath='/home/tobias/ml-testing/games/skyjo'
list_grad4=[f for f in os.listdir(myPath) 
    if (f.startswith('gradient3_fit2_it') )]
list_grad4.sort()

allgrad4=np.zeros((gradfit15.shape[0],gradfit15.shape[1],gradfit15.shape[2],len(list_grad4)))
for i in range(len(list_grad4)):
    allgrad4[:,:,:,i]=np.load(list_grad4[i])

list_grad5=[f for f in os.listdir(myPath) 
    if (f.startswith('gradient3_fit3_it') )]
list_grad5.sort()

allgrad5=np.zeros((gradfit15.shape[0],gradfit15.shape[1],gradfit15.shape[2],len(list_grad5)))
range2=len(list_grad5)
for i in range(len(list_grad5)):
    allgrad5[:,:,:,i]=np.load(list_grad5[i])
    

c=0
good_model2=np.zeros((19,41))
for i in range(allgrad4.shape[3]):
    for j in range(allgrad4.shape[1]):
        if np.mean(allgrad4[40,j,:,i])<34.0 and j==0:
            print(f"case {i} {j} losses to {np.mean(allgrad4[40,j,:,i])}")
            good_model2[:,c]=allgrad4[19:38,j,0,i]
            c+=1
        if np.mean(allgrad4[40,j,:,i])<32.0 and j!=0:
            print(f"case {i} {j} losses to {np.mean(allgrad4[40,j,:,i])}")  
            good_model2[:,c]=allgrad4[19:38,j,0,i]
            c+=1
print(f"{c} selected")         
print(allgrad4.shape)
for i in range(allgrad5.shape[3]):
    for j in range(allgrad5.shape[1]):
        if np.mean(allgrad5[40,j,:,i])<34.0 and j==0 and i!=0:
            print(f"case {i} {j} losses to {np.mean(allgrad5[40,j,:,i])}")
            good_model2[:,c]=allgrad5[19:38,j,0,i]
            c+=1
        if np.mean(allgrad5[40,j,:,i])<32.0 and j!=0:
            print(f"case {i} {j} losses to {np.mean(allgrad5[40,j,:,i])}")  
            good_model2[:,c]=allgrad5[19:38,j,0,i]
            c+=1
print(f"{c} selected")    

good_model3=np.zeros((19,40))
c2=0
for i in range(41):
    good=True
    for j in range(41):
        if i>j:
            if np.mean(good_model2[:,i])==np.mean(good_model2[:,j]):
                print(i,j)
                good=False
    if good==True:
        good_model3[:,c2]=good_model2[:,i]
        c2+=1
print(c2)    

#now better then 32 fits 
list_grad6=[f for f in os.listdir(myPath) 
    if (f.startswith('gradient4_fit1_it') )]
list_grad6.sort()
range2=len(list_grad6)

allgrad6=np.zeros((gradfit15.shape[0],gradfit15.shape[1],gradfit15.shape[2],range2))

for i in range(range2):
    allgrad6[:,:,:,i]=np.load(list_grad6[i])
good_models2b=np.zeros((19,11))
c=0
for i in range(1,allgrad6.shape[3]):
    for j in range(allgrad6.shape[1]):
        #excluding first model which same as before
        if np.mean(allgrad6[40,j,:,i])<32.0 and j==0 and i!=0 and np.mean(allgrad6[19:38,j,0,i])!=np.mean(allgrad6[19:38,0,0,0]):
            print(c)
            print(f"case {i} {j} losses to {np.mean(allgrad6[40,j,:,i])}")
            good_models2b[:,c]=allgrad6[19:38,j,0,i]
            c+=1
        if np.mean(allgrad6[40,j,:,i])<30.0 and j!=0:
            print(c)
            print(f"case {i} {j} losses to {np.mean(allgrad6[40,j,:,i])}")  
            good_models2b[:,c]=allgrad6[19:38,j,0,i]
            c+=1
print(f"{c} selected") 
good_model3b=np.zeros((19,11))
c2=0
for i in range(11):
    good=True
    for j in range(11):
        if i>j:
            if np.mean(good_models2b[:,i])==np.mean(good_models2b[:,j]):
                print(i,j)
                good=False
    if good==True:
        good_model3b[:,c2]=good_models2b[:,i]
        c2+=1
print(c2)  
#now better than 28 from mc gaussian 
mcg1=np.load("mcg_v1_all.npy")
#first than 28.0
sel_modelg=np.zeros((19,23))
csel=0
for i in range(mcg1.shape[1]):
    if np.mean(mcg1[40,i,:])<28.0:
        print(np.mean(mcg1[40,i,:]))
        sel_modelg[:,csel]=mcg1[19:38,i,0]
        csel+=1
print(f"{csel} are selected")    
#now between 28 and 29.5
sel_modelg=np.zeros((19,34))
csel=0
for i in range(mcg1.shape[1]):
    if np.mean(mcg1[40,i,:])>=28.0 and  np.mean(mcg1[40,i,:])<29.5:
        print(np.mean(mcg1[40,i,:]))
        sel_modelg[:,csel]=mcg1[19:38,i,0]
        csel+=1
print(f"{csel} are selected")    

allres2=np.load("mc_v9_all.npy")
c=0
list_open7=[]
list_discard7=[]
list_value7=[]
for i in range(850):
    if np.mean(allres2[40,i,4:6])<45 and np.max(allres2[40,i,:])<60:
        list_open7.append(allres2[19:25,i,0])
        list_discard7.append(allres2[25:31,i,0])        
        list_value7.append(allres2[31:38,i,0]) 
        c+=1
allres8=np.zeros((41,6,23))
start_time=time.time()
n_games=400
#all them with 400
#for 34 use good_model3
#doing in pieces first 10 is testbest34a_v2.npy
#10 to 20 is testbest34b_v2.npy
#20 to 25 is testbest34d_v2.npy
#25 to 30 is testbest34c_v2.npy
#30 to 35 is testbest34e_v2.npy
#35 to 40 is testbest34f_v2.npy
#now using good_model3b
#0 to 5 testbest32a_v2.npy
# 5 to 11 testbest32b_v2.npy
#now 200 
# 0 11 is testbest29a_v2.npy aa another rerun 
# 11 23 is testbest29b_v2.npy  bb anotyher random rerun
#now 400 again now of same 28 to 29.5 models 
#0 10 is a

for i in range(0,10):
    print(f"doing level 21 case {i}")
    allres8[:,:,i]=run_level21(list_open7,list_discard7,list_value7,sel_modelg[0:6,i],sel_modelg[6:12,i],sel_modelg[12:19,i],n_games)
np.save("testbest30a_v2.npy",allres8)
stop_time=time.time()
print(f"{n_games} ran for {np.round(stop_time-start_time,2)} seconds") 

