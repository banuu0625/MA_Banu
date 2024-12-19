
# General Imports
import re  # For regular expressions
import gc  # For garbage collection
import os  # For file operations
import numpy as np  # Numerical computations
import pandas as pd  # For loading and handling data
import chess  # Python-chess package to manage chess moves and rules


# PyTorch Imports
import torch  # PyTorch core library
import torch.nn as nn  # Neural network layers and functions
import torch.nn.functional as F  # Non-linear activations like ReLU
from torch.utils.data import Dataset, DataLoader  # DataLoader and Dataset

# Matplotlib for visualization (optional)
import matplotlib.pyplot as plt  # For plotting results

#column index mapping - change letters into numbers and vice versa
letter_2_num = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
num_2_letter = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}
#Chess board to matrix

# Idea : CNN's can acces 3D inputs !
# Create feature map for each chess piece type {pawn, knight, bishop} , white pieces =1, black pieces =-1
# CNN board can learn the rule and make moves based on the state of the game

board = chess.Board()

def board_2_rep(board):   #board object from chess package
  pieces = ['p', 'r', 'n', 'b', 'q', 'k']
  layers = []
  for piece in pieces:
    layers.append(create_rep_layer(board, piece)) #create feature map for each chess type
  board_rep = np.stack(layers) #transform feature maps into 3D-tensor
  return board_rep


def create_rep_layer(board, type):

  s = str(board)
  s = re.sub(f'[^{type}{type.upper()} \n]', '.', s) #replace everything with a '.' EXCEPT[^] desired piece (e.g pawns -> p and P)
  s = re.sub(f'{type}', '-1', s)                    #replace black pawns with -1's
  s = re.sub(f'{type.upper()}','1',s)              #replace white pawns with 1's
  s = re.sub(f'\.', '0', s)                         #replace dots with 0's


  board_mat = []
  for row in s.split('\n'):                         #loop through lines
    row = row.split(' ')                            #split by whitespaces
    row = [int(x) for x in row]                     #replace string numbers with actual integers
    board_mat.append(row)                           #represend in np array

  return np.array(board_mat)


print(f'{board}\n')
board_2_rep(board)


def move_2_rep(move, board ):
  board.push_san(move).uci()                                    #convert dataset from SAN into UCI  format : 4 letters {d4e5}
  move = str(board.pop())

  from_output_layer = np.zeros((8,8))                           #create 2D array 8x8 zeros
  from_row = 8 - int(move[1])                                   #determine row
  from_column = letter_2_num[move[0]]                           #determine column, convert into number -> now we now the intial position(row,column)
  from_output_layer[from_row, from_column] = 1                  #convert 0 to 1 on determined position

  to_ouptut_layer = np.zeros((8,8))                             #create 2D array 8x8 zeros
  to_row = 8 - int(move[3])                                     #determine row
  to_column = letter_2_num[move[2]]                             #determine clomn -> we know the position the piece moves to
  to_ouptut_layer[to_row,to_column] = 1                         #convert 0 into a 1 on determined position

  return np.stack([from_output_layer, to_ouptut_layer])         #convert to a numpy array (stack = adding together)


move_2_rep('e2e4', board)

def create_move_list(s):
  return re.sub('\d*\. ', '',s).split(' ')[:-1]                 #dataset = 1. (move) 2. (move), we should clear numbers followed by . (e.g 1., 2., 3.) with a white space ('')

chess_data_raw = pd.read_csv('/content/chess_games.csv', usecols =['AN','WhiteElo']) #asses only the columns we need
chess_data = chess_data_raw[chess_data_raw['WhiteElo']> 2000] #filter by elo
del chess_data_raw #delete games which are less then required elo
gc.collect() #clear RAMSPACE
chess_data = chess_data[['AN']] # del 'WhiteElo'
chess_data = chess_data[~chess_data['AN'].str.contains('{')] #clear up characters which code can't handle
chess_data = chess_data[chess_data['AN'].str.len()>20] #clear too short games
print(chess_data.shape[0])


