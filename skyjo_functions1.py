import numpy as np
import random as random
#using simpleguitk for display, is not needed for computer game
import simpleguitk as simplegui
import time

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
            
#pile of cards can be open or closed, only top most accessable
class Pile:
    #create parameter: string of mode, where to take cards for it, can be None
    def __init__(self,mode,take_from):
        #create new close pile, shuffles it also 
        if mode=='create_closed':
            #all closed in this mode
            self.open=False
            #postion in gui
            self.position=[100,10]
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
            self.position=[300,10]   
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
        ar=np.zeros((150),int)
        #non existent is default
        ar[:]=30
        
        for i in range(len(self.list_cards)):
            #if open give number
            if self.list_cards[-i-1].open==1:
                ar[i]=self.list_cards[-i-1].number
            #otherwise 20 (not oepnening here,but for player cards)
            else:
                ar[i]=20
        #ordering by value        
        ar[1:150]=np.sort(ar[1:150])        
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


#parameters: current player, all players (only needed for numeric output collection, closed_pile, discarded_pile, 
#Currently implemented mode with levels 0, -1, -2, -3
def turn(player,players,pile,discarded,silent=True,output=False):
    #global in_play parameter to check whether the game is over for one player
    global in_play
    if output==True:
        #optionally numeric output collection
        num=np.zeros((158+12*len(players)),int)
        #which players turn it is
        for i in range(len(players)):
            if players[i]==player:
                num[0]=i
        #discard and closed pile properties in data collection       
        num[1:151]=discarded.get_numeric()
        num[151]=len(pile.list_cards)  
        #cards of current players turn
        num[152:164]=player.get_numeric()
        #get of other players
        #counting the player to insert at right place
        player_counter=0
        for i in range(len(players)):
            #exclduing current player
            if players[i]!=player:
                player_counter+=1
                other_player=players[i]
                #get numeric of other players
                num[152+player_counter*12:164+player_counter*12]=other_player.get_numeric()
    #now choosing of actions, depends on mode and level        
    if player.mode=='computer':
        #in level 0 random 50% choice of action
        if player.level==0:
            r_number1=random.random()
            if r_number1<0.5:
                take_open=False
            else:    
                take_open=True
        #level -1/-2/-3 have assigned choices        
        if player.level==-1 or player.level==-2:
            take_open=False
        if player.level==-3:
            take_open=True
        #adding to numeric output
        if output==True:
            num[164+player_counter*12]=take_open   
    #genenral game structure does not depend on player  mode and level  
    if take_open==True:
        #by design, just to have not an empty data entry in the collected data
        discard=False
        if output==True:
            num[165+player_counter*12]=discard
        if silent==False:
            print("take from open swaping with own")
        #use method to get existing cards
        existing=player.get_all_cards()
        #select from them
        #randomly for implemented levels 0
        if player.mode=='computer':
            if player.level>=-3 and player.level<=0:
                #index of selected card
                selected=random.choice(existing)
        #get selected card        
        card_a=player.give_card(selected)
        if output==True:
            #dummy for closed pile value that entry is not empty
            num[166+player_counter*12]=30
            # get selected card properties
            num[167+player_counter*12]=selected
            num[168+player_counter*12]=card_a.number
            num[169+player_counter*12]=card_a.open
        #exchanging cards with open pile
        card_b=pile_open.give_card()
        pile_open.get_card(card_a)
        player.get_card(card_b,selected)
        if silent==False:            
            print("player "+player.name+" has now "+str(player.list_cards[selected].number))
            print("open Pile has now on "+str(pile_open.list_cards[-1].number))  
    #other options take from close pile        
    else:
        #currently only computer modes implemented
        if player.mode=='computer':        
            #in level 0 random 50% choice of sub action
            if player.level==0:
                r_number2=random.random()
                if r_number2<0.5:
                    discard=False
                else:    
                    discard=True
            #other modes have always the same actions        
            if player.level==-1 or player.level==-3:
                discard=False
            if player.level==-2:
                discard=True
        if output==True:
            num[164+player_counter*12]=take_open     
            num[165+player_counter*12]=discard 
        #taking if discard true
        if discard==True:
            if silent==False:                
                print("take from closed discards it")
            #put card on open pile from closed
            card_a=pile_closed.give_card()
            if output==True:
                #numeric output collection
                num[166+player_counter*12]=card_a.number 
            pile_open.get_card(card_a)
            #get lists of all closed player cards
            closed=player.get_all_closed()
            #computer options
            if player.mode=='computer':  
                #all implemented are random               
                if player.level<=0 and player.level>=-3:
                    selected=random.choice(closed)  
            #that random choice is passed to collected data
            if output==True:
                num[167+player_counter*12]=selected
                num[168+player_counter*12]=player.list_cards[selected].number
                num[169+player_counter*12]=player.list_cards[selected].open    
            #slected card is set to open    
            player.list_cards[selected].set_state(True)
            if silent==False:                
                print("open Pile has now on top "+str(pile_open.list_cards[-1].number))     
        #next (last) action option swap closed pile card with own 
        else:
            if silent==False:
                print("take from closed swaps with own")
            #get all existing cards
            existing=player.get_all_cards()
            if player.mode=='computer':              
                #randomly for all plemented levels
                if player.level<=0 and player.level>=-3:
                    selected=random.choice(existing)
            #get selected card  from player      
            card_a=player.give_card(selected)
            #get card from closed
            card_b=pile_closed.give_card()            
            if output==True:
                num[166+player_counter*12]=card_b.number
                num[167+player_counter*12]=selected
                num[168+player_counter*12]=card_a.number
                num[169+player_counter*12]=card_a.open
            #distributing the cards
            pile_open.get_card(card_a)
            player.get_card(card_b,selected)
            if silent==False:                
                print("player "+player.name+" has now "+str(player.list_cards[selected].number))                
                print("open Pile has now in top "+str(pile_open.list_cards[-1].number))
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
        return player, players, pile, discarded, num, card_needs_to_vanish

def allowed_modes(names,nature,levels):
    #list of allowed computer level, no humman currently implemented
    comp_level_list = [0,-1,-2,-3]
    #default is true
    allowed=True
    #between 2 and 8 players and input lists of same length
    if len(names)<2 or len(names)>8 or len(names)!=len(nature) or len(names)!=len(levels):
        allowed=False
    else:
        for i in range(len(names)):
            #for computer
            if nature[i]=='computer' and any(levels[i] in comp_level_list for item in comp_level_list)==False:
                allowed=False
            #human is not implemented and will have different levels    
            #for now because human not implemented    
            if nature[i]!='computer':
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
#output collection optional 
def skyjo_round(names,nature,levels,pause,first_round,silent=True,output=False):
    #check whether input is allowed and defined
    allowed=allowed_modes(names,nature,levels)
    #report and abort when not defined            
    if allowed==False:
        print("The player input \nnames:"+str(names)+"\nnature:"+str(nature)+"\nlevel:"+str(levels))
        print("is not allowed.")
    else:
        #globals (needed?)
        global in_play, players, pile_open, pile_closed
        #create piles at the beggining
        pile_closed=Pile('create_closed',False)
        pile_open=Pile('create_open',pile_closed)
        #counts how often cards vanish to see whethr that happens often enough for machine learning
        vanish_count=0
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
            starter,players,pile,discarded,num,van=turn(starter,players,pile_closed,pile_open,silent,True)
            #count how often card vanish in a round
            vanish_count+=van
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
                player,players, pile,discarded,num,van=turn(player,players,pile_closed,pile_open,silent,True)   
                outputs.append(num)
                #count how often card vanish in a round
                vanish_count+=van
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
                    player,players, pile,discarded,num,van=turn(player,players,pile_closed,pile_open,silent,True)    
                    outputs.append(num)
                    #count how often card vanish in a round
                    vanish_count+=van
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
                print(str(vanish_count)+" times equal card vanished")
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
            return scores, counter-idx+1, last_player, num2, vanish_count    
        
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
            #adding counts of vanishing card events
            van_count=0
            score,turns,last_player,num,van=skyjo_round(names,nature,levels,pause,True,True,output)   
            outputs.append(num)
            van_count=van 
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
                    score,turns,last_player,num,can=skyjo_round(names,nature,levels,pause,last_player,True,output)
                    outputs.append(num)
                    van_count+=van
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
                return final, winner, van_count
            else:
                #no numeric, output just winner
                return winner 
            
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
