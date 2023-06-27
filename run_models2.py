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
#used models previously generated 
mcg1=np.load("mcg_v1_all.npy") 
#standard good models
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


tot_score_output=True
n_games=50

#running subparts of mcg1 

#run 10 times to have subsets 
for j in range(10):
    allres8=np.zeros((int(41+tot_score_output),6,25))
    for i in range(25):
        print(f"doing level 21 case {i}")
        allres8[:,:,i]=run_level21(list_open7,list_discard7,list_value7,mcg1[19:25,i,0],mcg1[25:31,i,0],mcg1[31:38,i,0],n_games,tot_score_collect=tot_score_output)
    np.save("testbest_c"+str(j)+"_v1.npy",allres8)

stop_time=time.time()
print(f"{n_games} ran for {np.round(stop_time-start_time,2)} seconds") 

#2*25 need 
