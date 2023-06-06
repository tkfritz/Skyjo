import numpy as np
import random as random
#using simpleguitk for display, is not needed for computer game
import simpleguitk as simplegui
import time
#new ones
import pandas as pd
from xgboost import XGBRegressor
import os
#skjo functions
from skyjo_functions4 import *

#for level 1 computer needed     (not really here needed)                 
level1_2players_columns=np.loadtxt("xgb_model1_column2.txt")
level1_2players_model = XGBRegressor()
level1_2players_model.load_model("xgb_model2.json")

#define player parameters
names=('You','Computer')
mode=('human','computer')
level=(1,1)
global card_c,in_play,pile_closed,pile_open,players

#create pile and players
pile_closed=Pile('create_closed',False)
pile_open=Pile('create_open',pile_closed)
alpha=Player(names[0],mode[0],level[0],pile_closed)
beta=Player(names[1],mode[1],level[1],pile_closed) 
players=[alpha,beta]

#in play card just for visualization 
card_c=None
#play parameter
in_play=True
endcounter=0
end_score=[]
numeric=[]
discard=False
take_open=True
finisher=0
step=0
output=True
silent=True

#fill endscore with 0
for i in range(len(players)):
    end_score.append(0)

player=who_starts(True,players,None,silent=silent)

#index of starter player
counter=players.index(player)
#starting the GUI defining frame size
if len(names)==2:
    frame = simplegui.create_frame("Skyjo", 290+280*(1), 100+3*50+30) 
if abs(len(names)-3.5)==0.5:
    frame = simplegui.create_frame("Skyjo", 290+280*(1), 100+3*50*2+60)
if abs(len(names)-5.5)==0.5:
    frame = simplegui.create_frame("Skyjo", 290+280*(2)+10, 100+3*50*2+60)    
if abs(len(names)-7.5)==0.5:
    frame = simplegui.create_frame("Skyjo", 290+280*(3)+20, 100+3*50*2+60)  
frame.set_canvas_background("White")
#add buttons
frame.add_button("new game", new_game, 200)    
frame.add_button("discard", discard_yes, 200)
frame.add_button("keep", discard_no, 200)

#add handlers
frame.set_mouseclick_handler(mouseclick)
#makes problems currently
frame.set_draw_handler(draw)
#start game 
frame.start()

