# Skyjo

In this directory I develop and implement the card game Skyjo with python to play with a computer. 

## Playing Skyjo

The aim of the game is to finish with the least number of points. It starts with 2 open cards. When one player finishes opening all cards, the others may take one more card. Then the numerical values of the cards are added together. Caution, when the first player to finish does not have the smallest number of points his points are then doubled. The detailed actions are explained in the GUI. For playing SKyjo against a computer it is necessary to download the following files and to place them in one directory: skyjo_human_developing1.py xgb_model1_column2.txt xgb_model2.json

The computer needs to have the standard python libraries (check what is imported in the top of skyjo_human_developing1.py) and also xgboost and simpleguitk.

Then just run python skyjo_human_developing1.py and the rest are mouse clicks in the GUI.


## Developing Skjo


Here I mean the implementing and the functionality of creating the game. This includes applying machine learning results. It is developed, tested and partly also run for output skyjo_exploring.ipynb, skyjo_developing.ipynb and skyjo_developing.ipynb in that order, the last in parallel with skyjo_human_developing1.py because the GIU works better outside a notebook The skyjo functions are also in skyjo_functions1.py, skyjo_functions2.py, skyjo_functions3.py, skyjo_functions4.py The most recent are in the latter which is used most often.


## Machine Learning for Skyjo


The first explorative machine learning is in machine_learning_skyjo1.ipynb. The first part where you can see the machine is actually learning is in machine_learning_skyjo2.ipynb. In order to achieve this result, functions are developed which are stored in ml_functions1.py and ml_functions2.py (the newer) version. For the next machine learning step more data is produced with produce_data_for_machine_learning.py It is in the process of being used in machine_learning_skyjo3.ipynb. Future games of human versus the computer are analyzed in analyzing_human1.ipynb. 



