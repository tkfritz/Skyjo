#standard modukles
import numpy as np
import random as random
import time
import pandas as pd
#using simpleguitk for display, is not needed for computer game
#likely not needed un this notebook 
import simpleguitk as simplegui
#foor plotting 
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure
#for efficient saving and loading
import pickle
#skyjp game classes and functions
from skyjo_functions4 import *
#own functions for machine learning
from ml_functions2 import *
#for machine learning
from xgboost import XGBRegressor

#to supress sklearn warning that the columns have no names
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

#needed to produce outut 
columns=['open_pile_card']
for i in range(12):
    columns.append('own_cards_'+str(i))
for i in range(12):
    columns.append('other_player_cards_'+str(i))    
columns.append('action_take_open')
columns.append('action_discard')
columns.append('discard_value')
columns.append('id_player_card')
columns.append('numeric_player_card')            
columns.append('score_self')
columns.append('score_other')
columns.append('round')

sel2=np.loadtxt("xgb_model1_column2.txt")
#the score columns need to be marked back 
sel1=np.copy(sel2)
#the score set to 1 since need to be output
sel1[49:51]=1



#nature of players 
names=('alpha','beta')
nature=('computer','computer')
levels=(20,0)
#13 15 getn more into infinte loops



#test round non silent 

scores,turns,last_player,output=skyjo_round(names,nature,levels,0,True,False,True)
#main run
n_it=10
start_time=time.time()
df=round_for_ml(names,nature,levels,sel1,n_it,columns)
stop_time=time.time()
print(f"running time was {round(stop_time-start_time,4)} seconds")
print(df.head())
print(df.tail())
df.to_pickle('level_rand_levels20_0_'+str(n_it)+'.pkl')
