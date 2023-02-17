import numpy as np
import random as random
#using simpleguitk for display, is not needed for computer game
import simpleguitk as simplegui
import time
#new ones
import pandas as pd
from xgboost import XGBRegressor


#cards of the game
class Card:
    #initiate cards
    def __init__(self, number):
        #dictionary for defining colors
        dict_col={-2:'blue',-1:'blue',0:'cyan',1:'green',2:'green',3:'green',4:'green',5:'yellow',6:'yellow',7:'yellow',8:'yellow',9:'red',10:'red',11:'red',12:'red'}
        self.number = number   
        self.color = dict_col[number]
        #open or closed, default is closed
        self.open =False
        #whether selected in a turn
        self.inturn=False
        #default position is None
        self.position=[None,None]  
    #print output    
    def __str__(self):
        if self.open==False:
            return "Card has Value "+str(self.number)+" and is closed."+ \
                " Its position is at "+str(self.position)
        if self.open==True:
            return "Card has Value "+str(self.number)+" and is open."+ \
                " Its position is at "+str(self.position)
    #set card to a state
    def set_state(self,state):
        self.open=state
    def set_turn(self,turn):
        self.inturn=turn    
    #set card to number, mainly for testing    
    def set_number(self,number):
        #dictionary twice defined that class does not need outside information
        dict_col={-2:'blue',-1:'blue',0:'cyan',1:'green',2:'green',3:'green',4:'green',5:'yellow',6:'yellow',7:'yellow',8:'yellow',9:'red',10:'red',11:'red',12:'red'}
        self.number=number
        self.color = dict_col[number]
    #set position for display  
    def set_position(self,posinput):
        self.position=posinput
    #draw on GUi    
    def draw(self,canvas):
        card_size=[70,50]
        corners=[[self.position],[self.position[0]+card_size[0],self.position[1]],[self.position[0]+card_size[0],self.position[1]+card_size[1]],[self.position[0],self.position[1]+card_size[1]]]
        if self.open==False:
            #closed black
            canvas.draw_polygon(corners,1,'Black','Black')
        else:
            #open colored number on gray
            canvas.draw_polygon(corners,1,'Black','Light Gray')
            centerb=list(self.position)
            centerb[1]+=58
            centerb[0]+=5
            canvas.draw_text(self.number,centerb,50,self.color)
            #if in turn add colored rectangle
            if self.inturn==True:
                canvas.draw_polygon(corners,3,'Magenta')
    #get value (not getting number) when card not open before a turn is finished        
    def get_numeric(self):
        if self.open==1:
            value=self.number
        else:
            value=20
        return value            
    
    
#pile of cards can be open or closed, only top most accessable
class Pile:
    #create parameter: string of mode, where to take cards for it, can be None
    def __init__(self,mode,take_from):
        #create new close pile, shuffles it also 
        if mode=='create_closed':
            #all closed in this mode
            self.open=False
            #postion in gui
            self.position=[20,10]
            #number of cards per type
            dict_num={-2:5,-1:10,0:15,1:10,2:10,3:10,4:10,5:10,6:10,7:10,8:10,9:10,10:10,11:10,12:10}
            #create empty list
            self.list_cards=[]
            #add as many cards as exist of each type
            for i in range(15):
                for j in range(dict_num[i-2]):
                    #create card
                    cardin=Card(i-2)
                    #use position of Pile
                    cardin.set_position(self.position)
                    #and state of Pile
                    cardin.set_state(self.open)
                    #append
                    self.list_cards.append(cardin)    
                    #print content 
                    self.list_cards
            #shuffle cards        
            random.shuffle(self.list_cards)     
        #create open pile from other    starts with just one card
        if mode=='create_open': 
            #postion in gui
            self.position=[220,10]   
            #is open 
            self.open=True
            self.list_cards=[]
            self.list_cards.append(take_from.list_cards.pop(-1))
            #set card position properties to Pile properties
            self.list_cards[0].set_state(self.open)
            self.list_cards[0].set_position(self.position)            
    def __str__(self): 
        #print Pile properties
        if self.open==False:
            return "Pile has length of "+str(len(self.list_cards))+" Cards and is closed."+ \
                " Its position is at "+str(self.position)
        if self.open==True:
            return "Pile has length of "+str(len(self.list_cards))+" Cards and is open."+ \
                " Its position is at "+str(self.position)
    #giving a card default is top most (last) cards
    def give_card(self,position=-1):
        card=self.list_cards.pop(position)
        return card
    #copy card for simulation
    def copy_card(self,position=-1):
        card=Card(self.list_cards[position].number)
        card.set_state(self.open)
        #old let to unwanted changes in pile
        #card=self.list_cards[position]
        return card
    #get card, added at as the last (top most ) always
    def get_card(self,card):
        self.list_cards.append(card)
        #set card position and state to position ans state of pile
        self.list_cards[-1].set_position(self.position)
        self.list_cards[-1].set_state(self.open)
    #recreate pile if nearly empty. parameters: self=pile to be recreated), other pile to be used        
    def refill(self,p_open):
        #refill if there are less than 3 cards
        if len(self.list_cards)<3:
            while len(p_open.list_cards)>1:
                #card closest to the bottom is taken until there is only 1 card
                #cannot use other method in line below, then the top most is taken many times
                card_a=p_open.list_cards.pop(0)
                self.get_card(card_a)
            #shuffle again
            random.shuffle(self.list_cards)
            #return other pile, self in anyway right
            return p_open
    #gets output for machine learning input     
    #gets numpy formated 1D array of 150 length (could be 147 in principle, 3 cards are always elsewhere), value is number when open, closed 20, non existend 30   
    #starts with end because the is top most, most important card
    #all cards besides the first one are ordered by value since the order doe snot matter for 
    #them only the first one can be gottten back
    def get_numeric(self):
        ar=np.zeros((16),int)
        #first is top most card because it can be accesed it is separated
        #only on open cards as savegard
        if self.open==1:
            ar[0]=self.list_cards[-1].number
        #if other cards histogram of the rest, -2 to 12 in order
        if len(self.list_cards)>1:
            for i in range(len(self.list_cards)-1):
                if self.open==1:
                    ar[self.list_cards[i].number+3]+=1
        
        return ar         

#A Card Position can be empty. I use two lists here for the essential properties, one to check for empty and
#one which has stored the card. 
class Player:
    #parameters: string name, mode (human, computer),  level (subkind for mode, more options for computer then human) 
    def __init__(self,name,modes,level,take_from=None):
        #grid positions only for display the game logic uses the list position
        self.positionx=[0,70,140,210,0,70,140,210,0,70,140,210]
        self.positiony=[0,0,0,0,50,50,50,50,100,100,100,100]
        #whether cards exist
        self.exist=[]
        #which cards is there
        self.list_cards=[]
        #name of player
        self.name=name
        #check mode computer (different levels)/human
        self.mode=modes
        #level how random much randomness is there is     
        self.level=level 
        #if take_from is None, only player properties are defined but no cards given
        if take_from!=None:
            for i in range(12):
                #add card which exist
                self.exist.append(1)
                #from pile
                self.list_cards.append(take_from.list_cards.pop(-1))
                #set state to closed
                self.list_cards[-1].set_state(False)
                #set position of card to player grid
                pos=list([self.positionx[i],self.positiony[i]])
                self.list_cards[-1].set_position(pos) 
            #opening 2 cards, currently only open randomly is implemented
            if level<10:
                rang=list(range(12))
                random.shuffle(rang)
                self.list_cards[rang[0]].set_state(True)
                self.list_cards[rang[1]].set_state(True)
        #option for player to do that interactive to be added   
    #for further round in a game only recreate, rest is the same as above in in __init__
    def restart(self,take_from):
        self.exist=[]
        self.list_cards=[]   
        for i in range(12):
            self.exist.append(1)
            self.list_cards.append(take_from.list_cards.pop(-1))
            self.list_cards[-1].set_state(False)
            pos=list([self.positionx[i],self.positiony[i]])
            self.list_cards[-1].set_position(pos) 
        if self.level<10:
            rang=list(range(12))
            random.shuffle(rang)
            self.list_cards[rang[0]].set_state(True)
            self.list_cards[rang[1]].set_state(True) 
    #printing basic player properties        
    def __str__(self):
        return "Player "+self.name+" is a "+self.mode+" in level "+str(self.level)+"."
    #give a card, choosen by index, vanishes: is set to None and not existing
    def give_card(self,i):
        card=self.list_cards[i]
        self.exist[i]=0
        self.list_cards[i]=None
        return card
    #copy card for simulation
    def copy_card(self,i):
        card=Card(self.list_cards[i].number)
        card.set_state(self.list_cards[i].open)
        #old let to unwanted changes in given objects partly
        #card=self.list_cards[i]
        return card    
    #Get a card (parameters: card and index), is always open by design
    def get_card(self,card,i):
        #change ccard
        self.list_cards[i]=card
        #set to existing
        self.exist[i]=1
        #get position of the adress
        pos=list([self.positionx[i],self.positiony[i]])
        #use it
        self.list_cards[i].set_position(pos)
        self.list_cards[i].set_state(True)
    #set_value method only  used in testing    
    #paremeters, index of card in list, number, state     
    def set_value(self,i,number,state):
        #set to existing, open state and desired value
        self.exist[i]=1  
        self.list_cards[i].set_state(True)     
        self.list_cards[i].set_number(number)        
    #method to get list of all closed cards, optional print of how many there are    
    def get_all_closed(self,silent=True):
        #create empty list, append
        closed=[]
        for j in range(12):
            if self.exist[j]==1:
                if self.list_cards[j].open==False:
                    closed.append(j)
        if silent==False:      
            print(self.name+" has "+str(len(closed))+" closed cards")            
        return closed     
    #method to get list of all open cards, optional print of how many there are  
    #same structure as above
    def get_all_open(self,silent=True):   
        open2=[]
        for j in range(12):
            if self.exist[j]==1:
                if self.list_cards[j].open==True:
                    open2.append(j)
        if silent==False:
            print(self.name+" has "+str(len(open2))+" open cards")            
        return open2 
    #method to get list of all existing cards, optional print of how many there are  
    #same structure as above    
    def get_all_cards(self,silent=True):   
        existing=[]
        for j in range(12):
            if self.exist[j]==1:
                existing.append(j)
        if silent==False:
            print(self.name+" has "+str(len(existing))+" cards")            
        return existing    
    #get score at the end of a round, adidng up all number values of existing cards
    def get_score(self):   
        score=0
        for j in range(12):
            if self.exist[j]==1:
                score+=self.list_cards[j].number         
        return score 
    #when 3 open cards in column have the same value, they vanish 
    #check whether cards vanish
    def check_vanish_cards(self):
        #-1 is default when no vanished
        vanish=-1
        #checking the four options
        for j in range(4):
            #3 open of same number
            if self.exist[0+j]==True and self.exist[4+j]==True and self.exist[8+j]==True: 
                if self.list_cards[0+j].open==True and self.list_cards[4+j].open==True and self.list_cards[8+j].open==True and self.list_cards[0+j].number==self.list_cards[4+j].number and self.list_cards[0+j].number==self.list_cards[8+j].number:
                    #is yes index is returned by design there can only one row be added in each turn, thus it is fine
                    #that the methiod cannot handle several
                    vanish=j
        return vanish  
    #method which return vanished when previous method marks them (gets more than -1)
    def vanish_cards(self,row):
        card1=self.list_cards[0+row]
        card2=self.list_cards[4+row]
        card3=self.list_cards[8+row]
        self.exist[0+row]=0
        self.exist[4+row]=0
        self.exist[8+row]=0
        self.list_cards[0+row]=None
        self.list_cards[4+row]=None
        self.list_cards[8+row]=None
        return card1, card2, card3 
    #gets numeric output for machine learning, same principle than for piles
    #gets numpy formated 1D array of 12 length, values is number when open, closed 20, non existedn 30   
    def get_numeric(self):
        ar=np.zeros((12),int)
        for i in range(12):
            if self.exist[i]==0:
                ar[i]=30
            else:
                if self.list_cards[i].open==1:
                    ar[i]=self.list_cards[i].number
                else:
                    ar[i]=20
        return ar   

#function which dtermines who storts
#parameters: whether is first round of game, player_list, which player ended the last round, 
#,silent whether the result is printed on screen
def who_starts(first,players,last_ender,silent=True):
    #if first round of a game, player with largest sum of open values starts
    if first==True:
        #counter list for all players
        counters=[]
        for i in range(len(players)):
            #counting for each player
            counter=0
            #get all open cards #get there indixes
            open_cards=players[i].get_all_open(silent=True)
            for j in range(len(open_cards)):
                #get values of these cards
                counter+=players[i].list_cards[open_cards[j]].number
            counters.append(counter)            
        #get maximum
        maxc=max(counters)
        #get index of it, not important whoch chossen when there are several
        starter=players[counters.index(maxc)]
        if silent==False:
            print("Player "+str(starter.name)+" starts")        
        return starter 
    #otherwise the ender of the previous round starts
    else:
        if silent==False:
            print("Player "+str(last_ender.name)+" starts") 
        return last_ender  

#arguments lists which contaon lists of all the same lengths, default is an integer array since here all are integers
def lists_to_numpy(the_list,int2=True):
    if int2==True:
        array=np.zeros((len(the_list),len(the_list[0])),int)
    else:
        array=np.zeros((len(the_list),len(the_list[0])))
    for i in range(len(the_list)):
        array[i]=the_list[i]
    #transpose it to consistent with previous way below    
    return array.T    

#arguments lists which contaon lists of all the same lengths, default is an integer array since here all are integers
def list_of_array_to_numpy(the_list,int2=True):
    if int2==True:
        if the_list[0].ndim==1:
            array=np.zeros((len(the_list),np.shape(the_list[0])[0]),int)
        if the_list[0].ndim==2:
            array=np.zeros((len(the_list),np.shape(the_list[0])[0],np.shape(the_list[0])[1]),int)            
    else:
        array=np.zeros((len(the_list),len(the_list[0])))
    for i in range(len(the_list)):
        array[i]=the_list[i]
    #transpose it to consistent with previous way below    
    return array.T


#defines actions they are either done or just simulated(=explored to see which is best)
#parameters, plater, all players card, closed pile, discarded pile, take_open, discard
#optionla non silent, simulated, and whether a certain card (index of player list) should be used, round_number is for in the simulation
#now always output collect and propagated easier this way
#not ideal build too much repetation in the three different cases 
def actions(player,players,pile,discarded,take_open, discard,silent=True,simulated=False,card=-1,round_number=0):
    #now numeric output always collected (but necessarily passed) since needed for simulating level 1
    num=np.zeros((23+12*len(players)),int)
    
    #which players turn it is
    for i in range(len(players)):
        if players[i]==player:
            num[0]=i
    #discard and closed pile properties in data collection       
    num[1:17]=discarded.get_numeric()
    num[17]=len(pile.list_cards)  
    #cards of current players turn
    num[18:30]=player.get_numeric()
    #get of other players
    #counting the player to insert at right place
    player_counter=0
    for i in range(len(players)):
        #exclduing current player
        if players[i]!=player:
            player_counter+=1
            other_player=players[i]
            #get numeric of other players
            num[18+player_counter*12:30+player_counter*12]=other_player.get_numeric()    
    #numeric data collectd in lists (length unknown) in case of simulated  
    if simulated==True:
        data=[]
        if round_number==0:
            #the several option for closed are tested here
            data2=[]
    #actions array for simulated 3 action options 3 2 array
    act_array=np.array([[1,0],[0,1],[0,0]])
    #iterate over options alwys  how many depends on mode
    for j in range(3):
        if j==0 and ((simulated==False and act_array[j,0]==take_open and act_array[j,1]==discard) or (simulated==True and round_number==0)):
            if silent==False:
                print(f"case {j} take_open={act_array[j,0]} discard={act_array[j,1]}")            
            num2=np.copy(num)
            num2[30+player_counter*12]=act_array[j,0]
            num2[31+player_counter*12]=act_array[j,1]
            if act_array[j,0]==1 and act_array[j,1]==0:
                if silent==False and simulated==False:
                    print("take from open swaping with own")
            #use method to get existing cards
            existing=player.get_all_cards()
            #select from them
            #randomly for implemented levels 0
            if player.mode=='computer':
                if player.level>=-3 and player.level<=0:
                    #index of selected card
                    selected=[random.choice(existing)]
                if player.level==1 and simulated==False:
                    #card directly this it uses the direct index
                    selected=[card]
                #for simulated this one currently one implemented    
                if player.level==1 and simulated==True:
                    selected=existing.copy()
                #for human    
            if player.level==1 and player.mode=='human':
                    selected=[card]    
            #get selected cards
            for i in range(len(selected)):
                if simulated==False:
                    card_a=player.give_card(selected[i])                          
                else:    
                    card_a=player.copy_card(selected[i])                
                #dummy for closed pile value that entry is not empty
                num3=np.copy(num2)
                num3[32+player_counter*12]=30
                # get selected card properties
                num3[33+player_counter*12]=selected[i]
                num3[34+player_counter*12]=card_a.get_numeric()
                #exchanging cards with open pile
                if simulated==False:
                    card_b=pile_open.give_card()
                    pile_open.get_card(card_a)
                    player.get_card(card_b,selected[i])
                else:
                    #collect data
                    data.append(num3)
            if silent==False and simulated==False:            
                print("player "+player.name+" has now "+str(player.list_cards[selected[i]].number))
                print("open Pile has now on "+str(pile_open.list_cards[-1].number))                
        #next option take from close pile and dicards it, player opens closed card
        if j==1 and ((simulated==False and act_array[j,0]==take_open and act_array[j,1]==discard) or simulated==True):         
            if silent==False:
                print(f"case {j} take_open={act_array[j,0]} discard={act_array[j,1]}")   
            num2=np.copy(num)
            num2[30+player_counter*12]=act_array[j,0]
            num2[31+player_counter*12]=act_array[j,1]
            if silent==False and simulated==False:                
                print("take from closed discards it")
            #put card on open pile from closed
            if simulated==False:
                card_b=[pile_closed.give_card()]
            if simulated==True and round_number==1:
                card_b=[pile_closed.copy_card()]    
            if simulated==True and round_number==0:
                card_b=[]
                for k in range(15):
                    card_b.append(Card(k-2))                 
            if simulated==False:
                pile_open.get_card(card_b[0])
            #get lists of all closed player cards
            closed=player.get_all_closed()
            #computer options
            if player.mode=='computer':               
                #all implemented are random               
                if player.level<=0 and player.level>=-3:
                     selected=[random.choice(closed)] 
                if player.level==1 and simulated==False:
                    selected=[card]        
                if player.level==1 and simulated==True:
                    selected=closed.copy()         
                                #for human    
            if player.level==1 and player.mode=='human':
                    selected=[card]       
            #that random choice is passed to collected data
            if simulated==True and silent==False:
                print(f"length is {len(selected)}")
            for i in range(len(selected)):
                if silent==False and simulated==False:
                    print("Player opens "+str(player.list_cards[selected[i]].number))   
                num3=num2.copy()
                if len(card_b)==1:
                    #if not round 1
                    num3[32+player_counter*12]=card_b[0].number
                    num3[33+player_counter*12]=selected[i]
                    num3[34+player_counter*12]=player.list_cards[selected[i]].get_numeric()
                else:
                    num4=np.zeros((23+12*len(players),15),int)
                    for k in range(15):
                        num4[:32+player_counter*12,k]=num3[:32+player_counter*12]
                        num4[32+player_counter*12,k]=card_b[k].number
                        num4[33+player_counter*12,k]=selected[i]
                        num4[34+player_counter*12,k]=player.list_cards[selected[i]].get_numeric()
                #selected card is set to open
                if simulated==False:          
                    player.list_cards[selected[i]].set_state(True)     
                else:  
                    # add numeric to data
                    if round_number==1:
                        data.append(num3) 
                    if round_number==0:
                        data2.append(num4)                         
            if silent==False and simulated==False:                
                print("open Pile has now on top "+str(pile_open.list_cards[-1].number))     
        #next (last) action option swap closed pile card with own
        if j==2 and ((simulated==False and act_array[j,0]==take_open and act_array[j,1]==discard) or simulated==True):              
            if silent==False:
                print(f"case {j} take_open={act_array[j,0]} discard={act_array[j,1]}")   
            num2=np.copy(num)
            num2[30+player_counter*12]=act_array[j,0]
            num2[31+player_counter*12]=act_array[j,1]
            if silent==False and simulated==False:
                print("take from closed swaps with own")
            #get all existing cards
            existing=player.get_all_cards()
            if player.mode=='computer':
                if player.level>=-3 and player.level<=0:
                    #index of selected card
                    selected=[random.choice(existing)]
                #for simulated this one currently one implemented    
                if player.level==1 and simulated==True:
                    selected=existing.copy()
                if player.level==1 and simulated==False:
                    selected=[card]   
            #for human    
            if player.level==1 and player.mode=='human':
                selected=[card]                    
            #get selected cards
            for i in range(len(selected)):
                num3=num2.copy() 
                if simulated==False:
                    card_a=player.give_card(selected[i])
                    card_b=[pile_closed.give_card()] 
                else:    
                    card_a=player.copy_card(selected[i])   
                    
                    if round_number==1:
                        card_b=[pile_closed.copy_card()]    
                    elif round_number==0:
                        card_b=[]
                        for k in range(15):
                            card_b.append(Card(k-2))   
                    
                if len(card_b)==1:
                    #if not round 0
                    num3[32+player_counter*12]=card_b[0].number
                    num3[33+player_counter*12]=selected[i]
                    num3[34+player_counter*12]=card_a.get_numeric()
                else:
                    num4=np.zeros((23+12*len(players),15),int)
                    for k in range(15):
                        num4[:32+player_counter*12,k]=num3[:32+player_counter*12]
                        num4[32+player_counter*12,k]=card_b[k].number
                        num4[33+player_counter*12,k]=selected[i]
                        num4[34+player_counter*12,k]=card_a.get_numeric()
                #exchanging cards with open pile
                if simulated==False:
                    pile_open.get_card(card_a)
                    player.get_card(card_b[0],selected[i])
                else:
                    #collect data
                    if round_number==1:
                        data.append(num3)
                    elif round_number==0:
                        data2.append(num4)
                        
            if silent==False and simulated==False:                
                    print("player "+player.name+" has now "+str(player.list_cards[selected[i]].number))                
                    print("open Pile has now in top "+str(pile_open.list_cards[-1].number))    
        if silent==False and simulated==True:
            print(f"number of simulations collected until now: {len(data)}")
    #full aray outut when simulated         
    if simulated==True and round_number==1:
        return lists_to_numpy(data)
    if simulated==True and round_number==0:
        return lists_to_numpy(data), list_of_array_to_numpy(data2)    
    #only 1D array when not
    if simulated==False:
        return num3    
    
    
#name of model, which column to be used, inut file1, index, open_card column, discard column, round_number, silent
#optional second input file 
def determine_best_option(model,columns,input1,index, take_open,discard,n_inputs,silent=True,input2=0):
    if n_inputs==1:
        #round 1 option
        if silent==False:
            print("doing round 1 simulation")
        #output collected, open pile marker, discard marker, predicted score, card id used     
        all_scores=np.zeros((4,input1.shape[1]))
    else:
        if silent==False:
            print("doing round 0 simulation")        
        all_scores=np.zeros((4,input1.shape[1]+input2.shape[2]))
    #doing part for all known cards now
    selected=np.zeros((int(sum(columns)),input1.shape[1]))
    #get the right columns
    counter=0
    for i in range(input1.shape[0]):
        if  columns[i]==1:
            selected[counter,:]=input1[i,:]
            counter+=1
    #predict scores using xgb model, transpoised needed for it        
    pred_scores1=model.predict(selected.T)
    all_scores[0,:input1.shape[1]]=selected.T[:,take_open]
    all_scores[1,:input1.shape[1]]=selected.T[:,discard]
    all_scores[2,:input1.shape[1]]=pred_scores1
    all_scores[3,:input1.shape[1]]=selected.T[:,index]
    #if second inout exist
    if n_inputs==2:
        #weights for card values       
        weight_vec=np.array([5/150,10/150,15/150,10/150,10/150,10/150,10/150,10/150,10/150,10/150,10/150,10/150,10/150,10/150,10/150]) 
        #go over different cards
        for k in range(input2.shape[2]):
            #get prediction for all possible closed card values
            selected=np.zeros((int(sum(columns)),15))
            #get the needed columns
            counter=0
            for i in range(input2.shape[1]):
                if  columns[i]==1:
                    selected[counter,:]=input2[:,i,k]
                    counter+=1
            #transposed need to be used for xgb prediction
            pred_scores2=model.predict(selected.T)
            #weighted average them 
            weight_avg=np.dot(pred_scores2,weight_vec)
            #all have same values in selected 0 from same coordinate why
            all_scores[0,input1.shape[1]+k]=selected.T[0,take_open]
            all_scores[1,input1.shape[1]+k]=selected.T[0,discard]            
            all_scores[2,input1.shape[1]+k]=weight_avg
            all_scores[3,input1.shape[1]+k]=selected.T[0,index]   
    #get position of minum (best choice)
    x=np.argmin(all_scores[2])
    if all_scores[0,x]==1 or n_inputs==1:
        if silent==False:
            if all_scores[0,x]==1:
                print(f"minimum is for using open, and  giving own card {int(all_scores[3,x])}") 
            elif all_scores[0,x]==0 and all_scores[1,x]==0:
                print(f"minimum is for using closed, and giving own card {int(all_scores[3,x])}") 
            elif all_scores[0,x]==0 and all_scores[1,x]==1:
                print(f"minimum is for discard closed, and swaping own card {int(all_scores[3,x])}")                               
        #return take_open, discard, index of card if clear, otherise return -1, -1, -1
        #position of minimum 
        return int(all_scores[0,x]), int(all_scores[1,x]), int(all_scores[3,x])
    else:
        if silent==False:
            print(f"minimum is for using a closed card")   
        return -1, -1, -1            

#parameters: current player, all players (only needed for numeric output collection and for choosing startegry in some levels, closed_pile, discarded_pile, 
#Currently implemented mode with levels 0, -1, -2, -3
def turn(player,players,pile,discarded,silent=True,output=False):
    #global in_play parameter to check whether the game is over for one player
    global in_play,step  #, player_2models, player_2columns for later   
    #set take_open and discard
    if player.mode=='computer':
        #dictionaries here used level number to: models, column to be used, colomns of open, discard, index of card
        #for 2 players
        player_2models={1:level1_2players_model}
        player_2columns={1:level1_2players_columns}
        player_2take_open={1:25}
        player_2discard={1:26}
        player_2index={1:28}        
        #in level 0 random 50% choice of action
        if player.level==0:
            r_number1=random.random()
            if r_number1<0.5:
                take_open=False
            else:    
                take_open=True
            #discard random if one case
            if take_open==False:
                r_number2=random.random()                
                if r_number2<0.5:
                    discard=False
                else:    
                    discard=True   
            else:
                #set to false for take_open is true
                discard=False             
        #level -1/-2/-3 have assigned choices        
        if player.level==-1 or player.level==-2:
            take_open=False
        if player.level==-3:
            take_open=True
        if player.level==-1 or player.level==-3:
                discard=False
        if player.level==-2:
                discard=True
        #first level which implements machine learning   is implemented  
        if player.level==1 and len(players)==2:
            #simulations are done first, taken_open and discard are meaning less here
            num1,num2=actions(player,players,pile_closed,pile_open,True, False, silent=True,simulated=True,round_number=0)
            take_open,discard,selected_card=determine_best_option(player_2models[player.level],player_2columns[player.level],num1,player_2index[player.level],player_2take_open[player.level],player_2discard[player.level],2,silent=silent,input2=num2)
            #round 2 if best option is closed
            if take_open==-1:
                num1=actions(player,players,pile_closed,pile_open,True, False, silent=True,simulated=True,round_number=1)
                take_open,discard,selected_card=determine_best_option(player_2models[player.level],player_2columns[player.level],num1,player_2index[player.level],player_2take_open[player.level],player_2discard[player.level],1,silent=silent)   
                
    #now action function
    if silent==False:
        print("player "+player.name+" turn")
        
    #no card is preselected for first computer modes
    if player.level<=0:    
        num=actions(player,players,pile_closed,pile_open,take_open,discard,silent=silent)   
    if player.level>0:
        if player.mode=='computer':
            num=actions(player,players,pile_closed,pile_open,take_open,discard,silent=silent,card=selected_card)   
        #at leats need at add step setting back
        if player.mode=='human':
            num=actions(player,players,pile_closed,pile_open,take_open,discard,silent=silent,card=card)
            step=0
        
    #check for vanishing cards
    van=player.check_vanish_cards()
    #marker for card vanishing
    card_needs_to_vanish=False
    if van>-1:
        card_needs_to_vanish=True
        #cards are taken from player
        card1,card2,card3=player.vanish_cards(van)
        #given to open pile
        pile_open.get_card(card1)
        pile_open.get_card(card2)
        pile_open.get_card(card3)
        if silent==False:        
            print("3 "+str(card1.number)+" vanish from Player")
    #refill closed pile if needed 
    pile_closed.refill(pile_open)
    #check whether there are still closed cards 
    closed=player.get_all_closed()
    #if not play ends for this player and marker is set to not in_play
    if len(closed)==0:
        in_play=False 
        if silent==False:        
            print("player "+player.name+" opened all cards")
    if output==False:
        #with no output just collected the input again (needed?)
        return player, players, pile, discarded  
    else:
        #otherwise collect output and also marker on whether cards were vanished 
        #(useful to know whether enough data was collected that the machine learning can consider it)
        #combine num, card to vanish
        num2=np.zeros((len(num)+1))
        num2[:len(num)]=num
        num2[len(num)]=card_needs_to_vanish
        return player, players, pile, discarded, num2


def allowed_modes(names,nature,levels):
    #list of allowed computer level for 2 players
    #less implemented for more players
    comp_level_list2 = [1,0,-1,-2,-3]
    comp_level_list3 = [0,-1,-2,-3]
    comp_level_list4 = [0,-1,-2,-3]
    comp_level_list5 = [0,-1,-2,-3]
    comp_level_list6 = [0,-1,-2,-3]
    comp_level_list7 = [0,-1,-2,-3]
    comp_level_list8 = [0,-1,-2,-3]
    #human 
    human_level_list=[1]
    
    #default is true
    allowed=True
    #between 2 and 8 players and input lists of same length
    if len(names)<2 or len(names)>8 or len(names)!=len(nature) or len(names)!=len(levels):
        allowed=False
    else:
        for i in range(len(names)):
            #for computer player 2
            if nature[i]=='computer' and len(names)==2 and any(levels[i] in comp_level_list2 for item in comp_level_list2)==False:
                allowed=False
            #all option of player numbers implemented    
            elif nature[i]=='computer' and len(names)==3 and any(levels[i] in comp_level_list3 for item in comp_level_list3)==False:
                allowed=False    
            elif nature[i]=='computer' and len(names)==4 and any(levels[i] in comp_level_list4 for item in comp_level_list4)==False:
                allowed=False  
            elif nature[i]=='computer' and len(names)==5 and any(levels[i] in comp_level_list5 for item in comp_level_list5)==False:
                allowed=False  
            elif nature[i]=='computer' and len(names)==6 and any(levels[i] in comp_level_list6 for item in comp_level_list6)==False:
                allowed=False               
            elif nature[i]=='computer' and len(names)==7 and any(levels[i] in comp_level_list7 for item in comp_level_list7)==False:
                allowed=False  
            elif nature[i]=='computer' and len(names)==8 and any(levels[i] in comp_level_list8 for item in comp_level_list8)==False:
                allowed=False                    
            #for human   
            elif nature[i]=='human' and any(levels[i] in human_level_list for item in human_level_list)==False:
                allowed=False
            #rest does not exist
            else:
                allowed=False                
    return allowed             

#parameters index of acting player of a turn, and the result of the players (some kind of score)
def reorder_players(player,result):
    xx=len(result)
    array=np.zeros((xx))    
    #acting players
    array[0]=result[int(player)]
    for j in range(1,xx):
        #get scores of following players
        array[j]=result[int((j+player)%xx)] 
    return array    


#parameters: player names, modes, level (all list of same length), pause per turn (seconds can be zero), 
#whether it is first round either True, or player which starts 
#silent printing optional 
#output collection optional , vanish card count is now part of numeric output
def skyjo_round(names,nature,levels,pause,first_round,silent=True,output=False):
    #check whether input is allowed and defined
    allowed=allowed_modes(names,nature,levels)
    #report and abort when not defined            
    if allowed==False:
        print("The player input \nnames:"+str(names)+"\nnature:"+str(nature)+"\nlevel:"+str(levels))
        print("is not allowed.")
    else:
        #globals (needed? all)
        global in_play, players, pile_open, pile_closed
        #create piles at the beggining
        pile_closed=Pile('create_closed',False)
        pile_open=Pile('create_open',pile_closed)
        if output==True:
            #for output collection first a list is created which is at the end converted to an numpy array
            outputs=[]
        if first_round==True:
            #In first round: create players and add the players to list 
            players=[]
            for i in range(len(names)):
                players.append(Player(names[i],nature[i],levels[i]))
        for i in range(len(names)):
            #fill Player always from pile 
            players[i].restart(pile_closed)            
        if silent==False:
            print("closed pile has "+str(len(pile_closed.list_cards))+" cards")        
        #drawing here?    does not work to be added for interactive
        #frame.set_draw_handler(draw)
        #determine start player in first round 
        if first_round==True:
            if silent==False:
                print("game starts")
            for i in range(len(players)):
                if silent==False:
                    print(players[i])
            starter=who_starts(True,players,None)
        else:
            #could omit function in that case
            starter=who_starts(False,players,first_round)
            if silent==False:
                print("new round starts")
        if silent==False: 
            print("Player "+starter.name+" starts")
        #index of starter player
        idx=players.index(starter)
        if pause>0:
            #slow round to be able to observe in GUI or print out
            time.sleep(pause)
        #first turn
        if output==False:
            turn(starter,players,pile_closed,pile_open,silent,False)
        else:
            starter,players,pile,discarded,num=turn(starter,players,pile_closed,pile_open,silent,True)
            outputs.append(num)
        if silent==False:
            print("closed pile has "+str(len(pile_closed.list_cards))+" cards")     
        #counter set to index of starter, used to get index of next player
        counter=players.index(starter)
        #play starts
        in_play=True
        #index of player who finishes first with cards
        finisher=0
        #continue until one player finishes
        while in_play==True:
            #optionally slowing down
            if pause>0:
                time.sleep(pause)
            #now alternating turns on all player while in play
            counter+=1
            player=players[counter%len(players)]
            if silent==False:
                print("next turn is of "+player.name)            
            if output==False:
                #turn with and without output collectiob
                turn(player,players,pile_closed,pile_open,silent,False)
            else:
                player,players, pile,discarded,num=turn(player,players,pile_closed,pile_open,silent,True)   
                outputs.append(num)
            if silent==False:
                print(player.name)
                print("closed pile has "+str(len(pile_closed.list_cards))+" cards") 
            #index of finishing players    
            finisher=counter%len(players)        
        #one turn of the other players at the  end
        #last player will start next round in a game of several rounds
        last_player=None
        if in_play==False:
            for i in range(len(players)-1):
                #optional slowing
                if pause>0:
                    time.sleep(pause)
                #counting to get index of players    
                counter+=1
                player=players[counter%len(players)]
                #last_player will start next round in a game
                last_player=player 
                if silent==False:
                    print("next turn is of "+player.name)
                if output==False:
                    player,players,pile_closed,pile_open=turn(player,players,pile_closed,pile_open,silent,False)
                else:
                    player,players, pile,discarded,num=turn(player,players,pile_closed,pile_open,silent,True)    
                    outputs.append(num)
                if silent==False:
                    print("closed pile has "+str(len(pile_closed.list_cards))+" cards")
        if silent==False:
            print(str(counter-idx+1)+" turns were made")            
  
        #get score of round, list        
        scores=[]
        for i in range(len(players)):
            #get score of each player
            scores.append(players[i].get_score())   
        #need to consider who finished the round
        #if not the unique lowest (and positive) finishs first then score of the finisher is doubled
        #not_finisher controll parameter to avoid several doublings
        not_finisher=False
        if scores[finisher]>0:
            for i in range(len(players)):       
                if i!=finisher and not_finisher==False:
                    if scores[finisher]>=scores[i]:
                        scores[finisher]*=2
                        not_finisher=True
        if silent==False:
            print("score of round is "+str(scores))
        if output==True: 
            #output mostly the output of the round, plus round number and 
            #at the end the scores in the same order as players in that turn
            num2=np.zeros((len(num)+len(scores)+1,len(outputs)),int)
            for i in range(len(outputs)):
                num2[:len(num),i]=outputs[i]
                #turn number
                num2[len(num),i]=i
                #reorder scores
                res=reorder_players(num2[0,i],scores)
                #pass to array
                num2[len(num)+1:len(num)+len(scores)+1,i]=res
        if output==False:    
            return scores, counter-idx+1, last_player
        else:    
            return scores, counter-idx+1, last_player, num2    
        
        
#parameters: player names, modes, level (all list of same length),
#pause length (can be zero) 
#printing optional 
#output collection optional 
def skyjo_game(names,nature,levels,pause,silent=True,output=False):
    #check whether parameters are allowed and defined)
    allowed=allowed_modes(names,nature,levels)
    #report and abort when not defined            
    if allowed==False:
        print("The player input \nnames:"+str(names)+"\nnature:"+str(nature)+"\nlevel:"+str(levels))
        print("is not allowed.")
    else:
        #play the game
        #total book keeping round and score (list of player length)
        round_counter=0
        tot_score=[]
        #starts at zero
        for i in range(len(names)):
            tot_score.append(0)
        #marker for game over   
        over=False
        #create players and add to list
        players=[]
        #for output collection first a list which is at the end converted to array
        for i in range(len(names)):
            #player nature creation
            players.append(Player(names[i],nature[i],levels[i]))
        if output==False:
            score,turns,last_player=skyjo_round(names,nature,levels,pause,True,True,output) 
        else:
            #for output collection first a list which is at the end converted to array
            outputs=[]
            score,turns,last_player,num=skyjo_round(names,nature,levels,pause,True,True,output)   
            outputs.append(num)
        if silent==False:
            print("New Skyjo game")
            print("first round")
            print("score is "+str(score))
        
        for i in range(len(tot_score)):
            tot_score[i]+=score[i]
        if max(tot_score)>=100:
            over=True
        else:
            #more rounds until first player has at least 100
            while max(tot_score)<100:
                round_counter+=1
                if silent==False:
                    print("playing round "+str(round_counter+1))
                if output==False:
                    score,turns,last_player=skyjo_round(names,nature,levels,pause,last_player,True,output)
                else: 
                    score,turns,last_player,num=skyjo_round(names,nature,levels,pause,last_player,True,output)
                    outputs.append(num)
                for i in range(len(tot_score)):
                    tot_score[i]+=score[i] 
                if silent==False:
                    print("current score is "+str(tot_score))   
            else:
                over=True
        if over==True:
            if silent==False:
                print("game is over")   
            winner=[]    
            for i in range(len(players)):
                if tot_score[i]==min(tot_score):
                    winner.append(1)
                    if silent==False:
                        print("Player "+players[i].name+" won")    
                else:
                    winner.append(0)
            if output==True:
                #combine outputs in one array 
                #figure out size of it 
                row_counter=0
                for i in range(round_counter+1):
                    row_counter+=outputs[i].shape[1]
                #final numeric output
                #include besides propagate of round also round number, and reordered to player winner vector 
                final=np.zeros((outputs[0].shape[0]+1+len(players),row_counter),int)
                row_counter=0
                for i in range(round_counter+1):
                    final[0:outputs[0].shape[0],row_counter:row_counter+outputs[i].shape[1]]=outputs[i]
                    for j in range(outputs[i].shape[1]):
                        final[outputs[0].shape[0],row_counter+j]=i
                        #reorder to the winner to the player order
                        final[outputs[0].shape[0]+1:outputs[0].shape[0]+len(players)+1,row_counter+j]=reorder_players(final[0,row_counter+j],winner)
                    row_counter+=outputs[i].shape[1] 
                return final, winner
            else:
                #no numeric, output just winner
                return winner
            
# draw function
def draw(canvas):
    global pile_open,pile_closed, players,card_b, card_a, step, discard, take_open, player, end_score, player
    #display the top most card
    p_open=pile_open.list_cards[-1]
    p_closed=pile_closed.list_cards[-1]
    # test to make sure that card.draw works
    p_open.draw(canvas)
    p_closed.draw(canvas)
    pos_text=[293,45]
    #if not ended 
    if sum(np.abs(end_score))==0 and player.mode=='human':
        #instructions what to do 
        if step==0:
            canvas.draw_text("Choose open or closed pile",pos_text,15,'Black')
        if step==1:
            canvas.draw_text("Discard or keep?",pos_text,15,'Black')
        if step==2:
            if discard==True and take_open==False:
               canvas.draw_text("Choose closed card",pos_text,15,'Black')
            else:    
               canvas.draw_text("Choose any card",pos_text,15,'Black')
    elif player.mode=='computer' and sum(np.abs(end_score))==0:
        canvas.draw_text("Click anywhere for computer",pos_text,15,'Black')
    else:
        canvas.draw_text("Player "+players[np.argmin(end_score)].name+" won",pos_text,15,'Black')
        
    if card_c!=None:
        card_c.draw(canvas)
    #if card_a!=None: not done currently
    #    card_a.draw(canvas)        
    for i in range(len(players)):
        for j in range(12):
            #only cards which exist are drawn:
            if players[i].exist[j]==1:
                card=players[i].list_cards[j]
                if len(players)==2:
                    drawpos=list(card.position)
                    drawpos[0]=i*290+players[i].positionx[j]
                    drawpos[1]=100+players[i].positiony[j]
                    card.set_position(drawpos)
                    card.draw(canvas)
                    if players[i]!=player  and sum(np.abs(end_score))==0:
                        canvas.draw_text(players[i].name,(100+(i%2)*290,185*(1+(i//2))+90),15,'Black')
                    elif players[i]==player and sum(np.abs(end_score))==0 :   
                        #indicate whose turn it is
                        canvas.draw_text(players[i].name+" turn",(100+(i%2)*290,185*(1+(i//2))+90),15,'Black')   
                    else:
                        canvas.draw_text(players[i].name+" "+str(end_score[i]),(100+(i%2)*290,185*(1+(i//2))+90),15,'Black')                         
                if len(players)>2:
                    #paramter how to structure the layout
                    x=round(len(players)/2)
                    drawpos=list(card.position)
                    drawpos[0]=(i%x)*290+players[i].positionx[j]
                    drawpos[1]=180*(i//x)+100+players[i].positiony[j]
                    card.set_position(drawpos)
                    card.draw(canvas)
                    if players[i]!=player and sum(np.abs(end_score))==0:
                        canvas.draw_text(players[i].name,(100+(i%x)*290,185*(1+(i//x))+90),15,'Black')   
                    elif players[i]==player and sum(np.abs(end_score))==0: 
                        canvas.draw_text(players[i].name+" turn",(100+(i%x)*290,185*(1+(i//x))+90),15,'Black')                        
                    else:
                        canvas.draw_text(players[i].name+" "+str(end_score[i]),(100+(i%x)*290,185*(1+(i//x))+90),15,'Black')  
                        
#for level 1 computer needed                     
level1_2players_columns=np.loadtxt("xgb_model1_column2.txt")
level1_2players_model = XGBRegressor()
level1_2players_model.load_model("xgb_model2.json")

#play a game for computer 

#names=('alpha','beta')
#nature=('computer','computer')
#levels=(1,0)
#print(names,nature,levels)
#winner=skyjo_game(names,nature,levels,0,False,False)

#now human mode developement
#for restart (should be full game at some point)
def new_game():
    global mousepos, take_open, discard, player, canvas, card_c, step, card, in_play, counter, endcounter, end_score, finisher, players
    pile_closed=Pile('create_closed',False)
    pile_open=Pile('create_open',pile_closed)
    alpha=Player("alpha",'human',1,pile_closed)
    beta=Player("beta",'human',1,pile_closed) 
    card_c=None
    card=-1
    in_play=True
    silent=False
    endcounter=0
    take_open=False
    step=0
    players=[alpha,beta]
    end_score=[]
    for i in range(len(players)):
        end_score.append(0)
    
    player=who_starts(True,players,None,silent=True)
    counter=players.index(player)
    print(player.name)

def discard_yes():
    global step,discard
    if step==1:
        discard=True
        step=2
        print(discard, step)
        return discard    

def discard_no():
    global step, discard
    if step==1:
        discard=False
        step=2
        print(discard, step)
        return discard  

def mouseclick(pos):
    #card_c is just for display not actually used 
    global mousepos, take_open, discard, player, canvas, card_c, step, card, in_play, counter, endcounter, end_score, finisher
    #if game is going one
    if sum(np.abs(end_score))==0:
       #only for human
        if player.mode=='human':
            mousepos = list(pos)
            #selecting which pile
            pilepos=np.array([pile_closed.position,pile_open.position])
            for i in range(2):
                if step==0 and abs(mousepos[0]-(pilepos[i,0]+35))<35 and  abs(mousepos[1]-(pilepos[i,1]+25))<25:
                    if i==0:
                        step=1
                        take_open=False
                        card_c=pile_closed.copy_card()
                    else:
                        #step 1 not needed 
                        step=2
                        take_open=True
                        discard=False
                        #problem that open pile is empty if taken
                        card_c=pile_open.copy_card()
                    newpos=[pilepos[i,0],pilepos[i,1]]
                    card_c.set_turn(True)
                    card_c.set_position(newpos)
                    card_c.set_state(True)                
            #now selecting card
            if step==2:
                if take_open==False and discard==True:
                   cards=player.get_all_closed()
                else:
                    cards=player.get_all_cards()  
                for i in range(len(cards)):
                    if abs(mousepos[0]-(player.list_cards[cards[i]].position[0]+35))<35 and  abs(mousepos[1]-(player.list_cards[cards[i]].position[1]+25))<25:
                        #continue to next step
                        step=3
                        #index number to be used
                        card=cards[i]
                        print(card)
                        actions(player,players,pile_closed,pile_open,take_open,discard,silent=False,card=card)
                        card_c=None
                        step=0
                        counter+=1
                        if in_play==False:
                            endcounter+=1
                        #display selected card? 
                        #card_b.set_turn(True)
                    
                        return take_open, discard, card
        if player.mode=='computer': 
            #two clicks needed to advance computer 
            turn(player,players,pile_open,pile_closed,silent=False,output=False)
            counter+=1
            if in_play==False:
                endcounter+=1
                
        #end of turn actions     
        van=player.check_vanish_cards()
        #marker for card vanishing
        card_needs_to_vanish=False
        if van>-1:
            card_needs_to_vanish=True
            #cards are taken from player
            card1,card2,card3=player.vanish_cards(van)
            #given to open pile
            pile_open.get_card(card1)
            pile_open.get_card(card2)
            pile_open.get_card(card3)
            if silent==False:        
                print("3 "+str(card1.number)+" vanish from Player")
        #refill closed pile if needed 
        pile_closed.refill(pile_open)
        #check whether there are still closed cards 
        closed=player.get_all_closed()
        print(len(closed),in_play)
        #if not play ends for this player and marker is set to not in_play
        if len(closed)==0:
            #finisher needed for score
            if in_play==True:
                #counter -1 because enlarged before this is tested 
                finisher=(counter-1)%len(players)
            in_play=False
            if silent==False:        
                print("player "+player.name+" opened all cards")   
        player=players[counter%len(players)] 
    
    if endcounter==len(players)-1:
        #open all cards
        for i in range(len(players)):
            cards=players[i].get_all_cards()
            for j in range(len(cards)):
                players[i].list_cards[cards[j]].set_state(True)
        print(finisher,counter)
        scores=[]
        for i in range(len(players)):
            #get score of each player
            scores.append(players[i].get_score())   
        #need to consider who finished the round
        #if not the unique lowest (and positive) finishs first then score of the finisher is doubled
        #not_finisher controll parameter to avoid several doublings
        not_finisher=False
        if scores[finisher]>0:
            for i in range(len(players)):       
                if i!=finisher and not_finisher==False:
                    if scores[finisher]>=scores[i]:
                        scores[finisher]*=2
                        not_finisher=True
        end_score=scores.copy()                
        if silent==False:
            print("score of round is "+str(scores))

#correct besides that word and vanishing appear one later than wanted                    


pile_closed=Pile('create_closed',False)
pile_open=Pile('create_open',pile_closed)
alpha=Player("alpha",'human',1,pile_closed)
beta=Player("beta",'computer',1,pile_closed) 
players=[alpha,beta]
card_c=None
card=-1
in_play=True
silent=False
endcounter=0
end_score=[]
for i in range(len(players)):
    end_score.append(0)
finisher=0
take_open=False
step=0
player=who_starts(True,players,None,silent=True)
print(player.name)
#index of starter player
counter=players.index(player)
if len(players)==2:
    frame = simplegui.create_frame("Skyjo", 290+280*(1), 100+3*50+30) 
if abs(len(players)-3.5)==0.5:
    frame = simplegui.create_frame("Skyjo", 290+280*(1), 100+3*50*2+60)
if abs(len(players)-5.5)==0.5:
    frame = simplegui.create_frame("Skyjo", 290+280*(2)+10, 100+3*50*2+60)    
if abs(len(players)-7.5)==0.5:
    frame = simplegui.create_frame("Skyjo", 290+280*(3)+20, 100+3*50*2+60)  
frame.set_canvas_background("White")
frame.add_button("new game", new_game, 200)    
frame.add_button("discard", discard_yes, 200)
frame.add_button("keep", discard_no, 200)


frame.set_mouseclick_handler(mouseclick)
frame.set_draw_handler(draw)
frame.start()

