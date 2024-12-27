import re  
import gc  
import numpy as np  
import pandas as pd 
import chess  
#column index mapping - change letters into numbers and vice versa
letter_2_num = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
num_2_letter = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}
# Idea : CNN's can acces 3D inputs !
# Create feature map for each chess piece type {pawn, knight, bishop} , white pieces =1, black pieces =-1
# CNN board can learn the rule and make moves based on the state of the game
board = chess.Board()

def board_2_rep(board):   
  pieces = ['p', 'r', 'n', 'b', 'q', 'k']
  layers = []
  for piece in pieces:
    #create feature map for each chess type
    layers.append(create_rep_layer(board, piece)) 
  board_rep = np.stack(layers)
  return board_rep

def create_rep_layer(board, type):
  #replace components 
  s = str(board)
  s = re.sub(f'[^{type}{type.upper()} \n]', '.', s) 
  s = re.sub(f'{type}', '-1', s)                    
  s = re.sub(f'{type.upper()}','1',s)              
  s = re.sub(f'\.', '0', s)                         
  board_mat = []
  for row in s.split('\n'):                      
    row = row.split(' ')                           
    row = [int(x) for x in row]                  
    board_mat.append(row)                          
  return np.array(board_mat)

def move_2_rep(move, board ):
  #convert dataset from SAN into UCI  format : 4 letters {d4e5}
  board.push_san(move).uci()                                    
  move = str(board.pop())
  #determine pos of "from"
  from_output_layer = np.zeros((8,8))                           
  from_row = 8 - int(move[1])                                 
  from_column = letter_2_num[move[0]]                          
  from_output_layer[from_row, from_column] = 1 
  #determine pos of "to"                 
  to_ouptut_layer = np.zeros((8,8))                             
  to_row = 8 - int(move[3])                                     
  to_column = letter_2_num[move[2]]                            
  to_ouptut_layer[to_row,to_column] = 1                         
  return np.stack([from_output_layer, to_ouptut_layer])         

def create_move_list(s):
  return re.sub('\d*\. ', '',s).split(' ')[:-1]                

#Kaggle Dataset
chess_data_raw = pd.read_csv('chess_games.csv', usecols =['AN','WhiteElo']) #asses only the columns we need
chess_data = chess_data_raw[chess_data_raw['WhiteElo']> 2000] #filter by elo
del chess_data_raw #delete games which are less then required elo
gc.collect() #clear RAMSPACE
chess_data = chess_data[['AN']] # del 'WhiteElo'
chess_data = chess_data[~chess_data['AN'].str.contains('{')] #clear up characters which code can't handle
chess_data = chess_data[chess_data['AN'].str.len()>20] #clear too short games
print(chess_data.shape[0])


