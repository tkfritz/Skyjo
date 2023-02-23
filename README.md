# Skyjo

In this directory I develop and implement the card game Skyjo with python to play with a computer. 

## Playing Skyjo

The aim of the game is to have as little as possible points at the end. It starts with 2 open cards.
When one player finished opening all cards, the other have still one card. Then the cards are added together. Caution, when a the first finisher does not have the uniquely smallest point number, his are doubled. The detailed actions are explained in the GUI 
For playing it against a computer is necessary to download the following files and to place them in one directory:
skyjo_human_developing1.py
xgb_model1_column2.txt
xgb_model2.json

The computer needs to have the standard python libraries (check what is imported in the top of skyjo_human_developing1.py)
and also  xgboost and simpleguitk.

Then just run python skyjo_human_developing1.py
and the rest are mouse clicks in the GUI

## Developing Skjo

Here I mean with it implementing the functionality of it, which includes applying machine learning but the machine learning itself. 
It is developed, tested and partly also run for output skyjo_exploring.ipynb, skyjo_developing.ipynb and skyjo_developing.ipynb in that order, the last in parallel with skyjo_human_developing1.py because the GIU works better outside a notebook
The skyjo functions are also in skyjo_functions1.py, skyjo_functions2.py, skyjo_functions3.py, skyjo_functions4.py The nwest are in the latest which is used usually.

## Machine Learning for Skyjo

The first explorative machine learning is in machine_learning_skyjo1.ipynb. The first part of used machine learning is done in machine_learning_skyjo2.ipynb. 
For it functions are developed which are stored in ml_functions1.py and ml_functions2.py (the newer) version.
For the next machine learning step more data is produced with produce_data_for_machine_learning.py
It is in process of being used in machine_learning_skyjo3.ipynb
Further games of human against the computer are analyzed in analyzing_human1.ipynb



