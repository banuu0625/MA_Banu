from preprocessing import create_move_list, board_2_rep, move_2_rep, chess_data
import numpy as np
# PyTorch Imports
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
from torch.utils.data import Dataset, DataLoader 
import chess

class ChessDataset(Dataset):
  def __init__(self,games):
    super(ChessDataset, self).__init__()                      
    self.games = games

  def __len__(self):
      #use 40k games
      return 40_000                                           

  def __getitem__(self, index):
    game_i = np.random.randint(self.games.shape[0])          
    random_game = chess_data['AN'].values[game_i]             
    moves = create_move_list(random_game)                     
    game_state_i = np.random.randint(len(moves)-1)           
    next_move = moves[game_state_i]                          
    moves = moves[:game_state_i]                              
    board = chess.Board()
    for move in moves :
      board.push_san(move)                                    
    #feature map for played move
    x = board_2_rep(board)                                 
    y = move_2_rep(next_move, board)                          
    if game_state_i % 2 == 1:     
      x*= -1
    return x, y

#Network = several modules one after another
class Module(nn.Module):
    def __init__(self, hidden_size):
        super(Module, self).__init__()
        #2 Convolutional Layers
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)          
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        #2BN layers
        self.bn1 = nn.BatchNorm2d(hidden_size)                                            
        self.bn2 = nn.BatchNorm2d(hidden_size)
        #2BN layers
        self.activation1 = nn.SELU()                                                    
        self.activation2 = nn.SELU()

    def forward(self, x):
        x_input = torch.clone(x)                                                          
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        #Residual Connection
        x = x + x_input                                                                   
        x = self.activation2(x)
        return x

class ChessNet(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=200):
        super(ChessNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList([Module(hidden_size) for i in range(hidden_layers)]) 
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)
        for i in range(self.hidden_layers):
            x = self.module_list[i](x)
        x = self.output_layer(x)
        return x
    





