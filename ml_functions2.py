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
from xgboost import XGBClassifier 
#metrics
from sklearn.metrics import confusion_matrix

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
def loop_reg(feature_train, target_train, feature_test, target_test,max_depth,reg_start,reg_increase,reg_steps,file_name,regression=True):
    #that takes now some time
    resb=np.zeros((4,reg_steps))
    for i in range(reg_steps):
        print(f"doing case {i}")
        regularization=reg_start*reg_increase**i
        #regression
        if regression==True:
            ar=do_xgb(feature_train, target_train, feature_test, target_test,max_depth,reg=regularization)
        #classification
        else:
            ar=do_xgb_class(feature_train, target_train, feature_test, target_test,max_depth,reg=regularization)
        resb[:,i]=ar
        #saved at each step because it sometimes crashes 
        np.savetxt(file_name, resb) 

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
    
#f1 measure row x 
def fmeas(conf_matrix,x):
    prec=conf_matrix[x,x]/np.sum(conf_matrix[:,x])
    rec=conf_matrix[x,x]/np.sum(conf_matrix[x])
    f1=(2*prec*rec)/(prec+rec)
    if prec==0 and rec==0:
        f1=0
    return f1

#returns fractions of wrong  predicted
def perwrong(conf_matrix):
    return 1-(np.sum(conf_matrix)-conf_matrix[0,1]-conf_matrix[1,0])/np.sum(conf_matrix)


#for xgb calssfier metric, is percentage wrong
#feature_train, target_train, feature_test, target_train, max depth of xgb, needs always be set *6 is equal to default), optional regularization alpha (larger less overfitting)
def do_xgb_class(feature_train, target_train, feature_test, target_test,max_depth,reg=0):
    start_time=time.time()
    #no regularization option
    if reg==0:
        regxl27=XGBClassifier(max_depth=max_depth).fit(feature_train, target_train)
    else:
        regxl27=XGBClassifier(max_depth=max_depth,reg_alpha=reg).fit(feature_train, target_train)        
    stop_time=time.time()
    print(f"xgb took {round(stop_time-start_time,4)} seconds")
    predtest=regxl27.predict(feature_test)
    predtrain=regxl27.predict(feature_train)
    conf_train = confusion_matrix(target_train, predtrain)
    conf_test = confusion_matrix(target_test, predtest)    
    test=perwrong(conf_test)
    train=perwrong(conf_train)
    print(f"percentage wrong test {round(test*100,2)}")
    print(f"percentage wrong train {round(train*100,2)} ")
    #copy result to array which can be used by other function
    ar=np.zeros((4))
    ar[0]=reg
    ar[1]=max_depth
    ar[2]=train
    ar[3]=test
    return ar

#find (and run) best xgbosst (regression and classification) of a collection
#parameters are list of the files with the metric and parameters, train_features, train_targtes, whether regression (default) or classification
def find_best(list_inputs,feature_train,target_train,output_file_name,regression=True):
    a=np.loadtxt(list_inputs[0])
    all_metrics=np.zeros((5,len(list_inputs),a.shape[1]))

    for i in range(len(list_inputs)):
        a=np.loadtxt(list_inputs[i])
        all_metrics[0:4,i,:]=a
    #first just using minimum in data
    s1=np.unravel_index(np.argmin(all_metrics[3,:,:]),all_metrics[3,:,:].shape)
    s2=np.argsort(all_metrics[3,:,:],axis=None)
    if regression==True:
        #to the minium seems fine for regression 
        print(f"minimum of {round(all_metrics[3,s1[0],s1[1]],2)} points is at alpha={round(all_metrics[0,s1[0],s1[1]],2)} and max_depth={int(all_metrics[1,s1[0],s1[1]])}")
        reg4=XGBRegressor(max_depth=int(all_metrics[1,s1[0],s1[1]]),reg_alpha=all_metrics[0,s1[0],s1[1]]).fit(feature_train, target_train)
        #and save it
        reg4.save_model(output_file_name)
    else:
        #for classification the minimum seems not good defined
        #but many choices get a similar floor value. The actual minum is likely dependent on the eaxct test sample and thus not necessary reliable. To chose a more relaible, the following procedure is used. Quantile in the test sample are calcauted from 5% onwards in steps of 10%. For that only max depth 5 and larger is used because the minium is always is those for classification. 
        #This quantiles are then used to define the allowed values of the test metric, it is always the one of 15%, it is enlarged, when the quantile slope is getting lower still outside of it. 
        #That define the maximum allowed metric value
        r=np.quantile(all_metrics[3,4:8,:],[0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95])
        print("quantiles of larger max depth half")
        print(r)
        #use them until they get larger again but at least the first 15%
        max_wrong=r[1]
        c=0
        while r[c+2]-r[c+1] <=(r[c+1]-r[0])/(1+c):
            max_wrong=r[c+2]
            c+=1
        print(f"accepted percentage  is {round(100*max_wrong,2)}")  
        #get maximum alpha within this limit, by increasing in alpha in loop start value is mimumum
        #loop goes the other way in max_depth to have this as small as possible
        value=[all_metrics[1,s1[0],s1[1]],all_metrics[0,s1[0],s1[1]]]
        per=all_metrics[3,s1[0],s1[1]]
        for j in range(all_metrics.shape[2]): 
            for i in range(all_metrics.shape[1]):
                if all_metrics[3,7-i,j]<=max_wrong:
                    value[0]=all_metrics[1,7-i,j]
                    value[1]=all_metrics[0,7-i,j] 
                    per=all_metrics[3,7-i,j] 
        print(f"minimum of {round(100*all_metrics[3,s1[0],s1[1]],2)} % is at alpha={round(all_metrics[0,s1[0],s1[1]],2)} and max_depth={int(all_metrics[1,s1[0],s1[1]])}")
        print(f"used of {round(100*per,2)} % is at alpha={round(value[1],2)} and max_depth={int(value[0])}")        
        reg4=XGBClassifier(max_depth=int(value[0]),reg_alpha=value[1]).fit(feature_train, target_train)
        #and save it
        reg4.save_model(output_file_name)  

def lists_arrays_to_one(listf,int2=True):
    #determine length of output
    c=0
    for i in range(len(listf)):
        c+=listf[i].shape[1]
        #create array
    ar=np.zeros((listf[0].shape[0],c))   
    c=0
    #now fill it
    for i in range(len(listf)):
        if int2==True:
            ar[:,c:c+listf[i].shape[1]]=np.round(listf[i])
        else:
            ar[:,c:c+listf[i].shape[1]]=np.round(listf[i])
        c+=listf[i].shape[1]
    return ar   

def logistic_function(coefs,data):
    res=1/(1+np.exp(-coefs[0]-np.matmul(coefs[1:coefs.shape[0]],data)))
    return res

