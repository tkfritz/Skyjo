#helper functions devoped in machine_learning_skyjo1.ipynb
#libraries neded for it 
import numpy as np
import random as random
import time
import pandas as pd
#using simpleguitk for display, is not needed for computer game
#likely not needed un this notebook 
import simpleguitk as simplegui
from xgboost import XGBRegressor

#other function needed for it 
#upwards compatible
from skyjo_functions4 import *

#file needed to run
#models
level1_2players_columns=np.loadtxt("xgb_model1_column2.txt")
level1_2players_model = XGBRegressor()
level1_2players_model.load_model("xgb_model2.json")

def round_for_ml(names,nature,levels,selected_features,nn,column):
    for i in range(nn):
        if i%4==0:
            #print sometimes
            print("case "+str(i))
        scores,turns,last_player,numeric=skyjo_round(names,nature,levels,0,True,True,True) 
        num2=np.zeros((int(sum(selected_features)+1),numeric.shape[1]),int)
        #counter in array add the columns which should be collected
        c=0
        for j in range(numeric.shape[0]):
            if selected_features[j]==1:
                num2[c,:]=np.round(numeric[j,:])
                c+=1
        #add number of round that different rounds can be easily distinguished         
        num2[c,:]=int(i)
        if i==0:
            #create first data frame
            df = pd.DataFrame(data = num2.T, index=range(num2.shape[1]),columns = column)
        else:
            #create new one of new data
            df2 = pd.DataFrame(data = num2.T, index=range(df.shape[1],df.shape[1]+num2.shape[1]),columns = column)
            #merge again, merging in every turn, is slower than the round for large number
            # likely other data type is better used before a table is made at the end
            #but below is discovered that this anyway not the best way to generate data
            df=pd.concat([df,df2])
    print(f"number of rows is {df.shape[0]}")          
    return df

#parameters: feature, target, controll series (only split on none equal in controll), 
#split 1 (between >0 and <1), optional split2 (if 1 it is not done)
def split_test(feature,target,controll,split_1,split_2=1):
    #only split into two sets in this case
    if split_2==1:
        xx=round(split_1*feature.shape[0])
        #orginal split
        print(xx)
        #first subset is always larger than specified, that is not an important problem for the sizes here 
        while controll.iat[xx-1]==controll.iat[xx]:
            xx+=1
        #final split    
        print(xx)    
        feature_train=feature.iloc[0:xx,:]    
        feature_test=feature.iloc[xx:feature.shape[0],:]
        target_train=target.iloc[0:xx]    
        target_test=target.iloc[xx:feature.shape[0]]        
        return feature_train, feature_test, target_train, target_test
    else:
        #now implement to splits
        xx=round(split_1*feature.shape[0])
        print(xx)
        while controll.iat[xx-1]==controll.iat[xx]:
            xx+=1
        print(xx)
        yy=round(split_2*feature.shape[0])
        print(yy)        
        while controll.iat[yy-1]==controll.iat[yy]:
            yy+=1
        print(yy)    
        feature_train=feature.iloc[0:xx,:]    
        feature_test=feature.iloc[xx:yy,:]
        feature_valid=feature.iloc[yy:feature.shape[0],:]        
        target_train=target.iloc[0:xx]    
        target_test=target.iloc[xx:yy]      
        target_valid=target.iloc[yy:feature.shape[0]]        
        return feature_train, feature_test, feature_valid, target_train, target_test, target_valid    

#selects a single random row per round
def round_for_ml_sel(names,nature,levels,selected_features,nn,column):
    numall=np.zeros((int(sum(selected_features)+1),nn),int)
    for i in range(nn):
        if i%1000==0:
            print("case "+str(i))
        scores,turns,last_player,numeric=skyjo_round(names,nature,levels,0,True,True,True) 
        #counter of columns in the array
        c=0
        b=random.randrange(numeric.shape[1])
        for j in range(numeric.shape[0]):
            if selected_features[j]==1:
                numall[c,i]=np.round(numeric[j,b])
                c+=1
        #round number was missing in some runs, it is anyway just identical to the index now 
        numall[c,i]=int(i)
    df = pd.DataFrame(data = numall.T, index=range(numall.shape[1]),columns = column)        
    print(f"number of rows is {df.shape[0]}")      
    return df        

#parameters feature target, split 1(between >0 and <1), optional split2 (if 1 not done)
def split_test_valid(feature,target,split_1,split_2=1):
    #only split into two sets in this case
    if split_2==1:
        xx=round(split_1*feature.shape[0])
        print(xx)   
        feature_train=feature.iloc[0:xx,:]    
        feature_test=feature.iloc[xx:feature.shape[0],:]
        target_train=target.iloc[0:xx]    
        target_test=target.iloc[xx:feature.shape[0]]        
        return feature_train, feature_test, target_train, target_test
    else:
        xx=round(split_1*feature.shape[0])
        print(xx)
        yy=round(split_2*feature.shape[0])
        print(yy)           
        feature_train=feature.iloc[0:xx,:]    
        feature_test=feature.iloc[xx:yy,:]
        feature_valid=feature.iloc[yy:feature.shape[0],:]        
        target_train=target.iloc[0:xx]    
        target_test=target.iloc[xx:yy]      
        target_valid=target.iloc[yy:feature.shape[0]]   
        return feature_train, feature_test, feature_valid, target_train, target_test, target_valid   

#feature_train, target_train, feature_test, target_train, max depth of xgb, needs always be set *6 is equal to default), optional regularization alpha (larger less overfitting)
def do_xgb(feature_train, target_train, feature_test, target_test,max_depth,reg=0):
    start_time=time.time()
    #no regularization option
    if reg==0:
        regxl27=XGBRegressor(max_depth=max_depth).fit(feature_train, target_train)
    else:
        regxl27=XGBRegressor(max_depth=max_depth,reg_alpha=reg).fit(feature_train, target_train)        
    stop_time=time.time()
    print(f"xgb took {round(stop_time-start_time,4)} seconds")
    predli1texl27=regxl27.predict(feature_test)
    predli1trxl27=regxl27.predict(feature_train)
    test_scatter=np.std(predli1texl27-target_test)
    train_scatter=np.std(predli1trxl27-target_train)
    print(f"standard deviation of test {round(np.std(target_test),4)} points")
    print(f"standard deviation of train {round(np.std(target_train),4)}  points")
    print(f"standard deviation of prediction-test {round(test_scatter,4)} points")
    print(f"standard deviation of prediction-train {round(train_scatter,4)} points")
    #copy result to array which can be used by other function
    ar=np.zeros((4))
    ar[0]=reg
    ar[1]=max_depth
    ar[2]=train_scatter
    ar[3]=test_scatter
    return ar


#feature, target(train), #feature, target(test), max_depth, start_reg, factor of increase, number of steps
#name of output file
def loop_reg(feature_train, target_train, feature_test, target_test,max_depth,reg_start,reg_increase,reg_steps,file_name):
    #that takes now some time
    resb=np.zeros((4,reg_steps))
    for i in range(reg_steps):
        print(f"doing case {i}")
        regularization=reg_start*reg_increase**i
        ar=do_xgb(feature_train, target_train, feature_test, target_test,max_depth,reg=regularization)
        resb[:,i]=ar
        #saved at each step because it sometimes crashes 
        np.savetxt(file_name, resb) 
