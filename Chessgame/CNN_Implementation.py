# General Imports
import re  # For regular expressions
import gc  # For garbage collection
import os  # For file operations
import numpy as np  # Numerical computations
import pandas as pd  # For loading and handling data
import chess  # Python-chess package to manage chess moves and rules
import pygame
import sys #module which works with the interpreter

# PyTorch Imports
import torch  # PyTorch core library
import torch.nn as nn  # Neural network layers and functions
import torch.nn.functional as F  # Non-linear activations like ReLU
from torch.utils.data import Dataset, DataLoader  # DataLoader and Dataset

# Matplotlib for visualization (optional)
import matplotlib.pyplot as plt  # For plotting results

#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################
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

#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

pygame.init()

# Chessboard constants
BOARD_WIDTH, BOARD_HEIGHT = 800, 800
MARGIN = 50
SQUARE_SIZE = BOARD_WIDTH // 8

# Screen size
SCREEN_WIDTH = BOARD_WIDTH + 2 * MARGIN
SCREEN_HEIGHT = BOARD_HEIGHT + 2 * MARGIN

# Colors
WHITE = (232, 235, 234)
BLACK = (125, 135, 150)
GREEN = (36, 232, 19)

# Create the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Chess Game")

# Chess piece class to build the individual chess pieces
class ChessPiece:
    def __init__(self, color, type, image):
        self.color = color
        self.type = type
        self.image = pygame.image.load(image)
        self.image = pygame.transform.scale(self.image, (SQUARE_SIZE, SQUARE_SIZE))
        self.has_moved = False

# Initialize the board
board = [[None for _ in range(8)] for _ in range(8)]

#[[None, None, None, None, None, None, None, None],
#[None, None, None, None, None, None, None, None],     col
#[None, None, None, None, None, None, None, None],      ^
#[None, None, None, None, None, None, None, None],      |
#[None, None, None, None, None, None, None, None],      -> row
#[None, None, None, None, None, None, None, None],
#[None, None, None, None, None, None, None, None],
#[None, None, None, None, None, None, None, None]]

game_board = chess.Board()  # This will handle all valid move checking and game state

# r n b q k b n r
# p p p p p p p p
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# . . . . . . . .
# P P P P P P P P
# R N B Q K B N R


# Selected piece
selected_piece = None
selected_pos = None
valid_moves = []  # List of valid moves for the currently selected piece

# Initialize the chess board with pieces using created class
def init_board():
    # Pawns
    for col in range(8):
        board[1][col] = ChessPiece('black', 'pawn', 'Chessgame/images/black_pawn.png')
        board[6][col] = ChessPiece('white', 'pawn', 'Chessgame/images/white_pawn.png')

    # Rooks
    board[0][0] = board[0][7] = ChessPiece('black', 'rook', 'Chessgame/images/black_rook.png')
    board[7][0] = board[7][7] = ChessPiece('white', 'rook', 'Chessgame/images/white_rook.png')

    # Knights
    board[0][1] = board[0][6] = ChessPiece('black', 'knight', 'Chessgame/images/black_knight.png')
    board[7][1] = board[7][6] = ChessPiece('white', 'knight', 'Chessgame/images/white_knight.png')

    # Bishops
    board[0][2] = board[0][5] = ChessPiece('black', 'bishop', 'Chessgame/images/black_bishop.png')
    board[7][2] = board[7][5] = ChessPiece('white', 'bishop', 'Chessgame/images/white_bishop.png')

    # Queens
    board[0][3] = ChessPiece('black', 'queen', 'Chessgame/images/black_queen.png')
    board[7][3] = ChessPiece('white', 'queen', 'Chessgame/images/white_queen.png')

    # Kings
    board[0][4] = ChessPiece('black', 'king', 'Chessgame/images/black_king.png')
    board[7][4] = ChessPiece('white', 'king', 'Chessgame/images/white_king.png')


# Draw the chessboard
def draw_board():
    for row in range(8):
        for col in range(8):
            color = WHITE if (row + col) % 2 == 0 else BLACK #alternating colour
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE + MARGIN, row * SQUARE_SIZE + MARGIN, SQUARE_SIZE, SQUARE_SIZE))
    
    # Highlight the selected square if any
    if selected_pos:
        pygame.draw.rect(screen, GREEN, (selected_pos[1] * SQUARE_SIZE + MARGIN, selected_pos[0] * SQUARE_SIZE + MARGIN, SQUARE_SIZE, SQUARE_SIZE))

# Draw the chess pieces on the board
def draw_pieces():
    for row in range(8):
        for col in range(8):
            piece = board[row][col]
            if piece:
                screen.blit(piece.image, (col * SQUARE_SIZE + MARGIN, row * SQUARE_SIZE + MARGIN)) #put image for every position in board[row][col], defined in function above
# Function to draw the labels 
def draw_labels():
    font = pygame.font.SysFont(None, 36)
    
    # Letters (A to H) - for top and bottom
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for i in range(8):
        # Draw at the top
        text = font.render(letters[i], True, WHITE)
        screen.blit(text, (i * SQUARE_SIZE + MARGIN + SQUARE_SIZE // 2 - text.get_width() // 2, MARGIN // 4)) #idk
        
        # Draw at the bottom
        screen.blit(text, (i * SQUARE_SIZE + MARGIN + SQUARE_SIZE // 2 - text.get_width() // 2, SCREEN_HEIGHT - MARGIN // 1.5)) #idk
    
    # Numbers (1 to 8) - for left and right
    for i in range(8):
        text = font.render(str(8 - i), True, WHITE)
        
        # Draw on the left
        screen.blit(text, (MARGIN // 4, i * SQUARE_SIZE + MARGIN + SQUARE_SIZE // 2 - text.get_height() // 2)) #idk
        
        # Draw on the right
        screen.blit(text, (SCREEN_WIDTH - MARGIN // 1.5, i * SQUARE_SIZE + MARGIN + SQUARE_SIZE // 2 - text.get_height() // 2)) #idk

# Handle clicks and piece movement
def handle_click(pos):
    global selected_piece, selected_pos, valid_moves

    # Check if the click is within the actual board area, considering margins
    if MARGIN <= pos[0] < SCREEN_WIDTH - MARGIN and MARGIN <= pos[1] < SCREEN_HEIGHT - MARGIN:
        # Convert pixel position to board coordinates (row, col), taking margin into account
        col = (pos[0] - MARGIN) // SQUARE_SIZE
        row = (pos[1] - MARGIN) // SQUARE_SIZE

        # Mapping to get chess notation for clicked square
        letter_2_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
        num_2_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
        clicked_square = f'{num_2_letter[col]}{8 - row}'

        # If no piece is selected, attempt to select one
        if selected_piece is None:
            piece = board[row][col]
            if piece:  # A piece exists at this position
                selected_piece = piece
                selected_pos = (row, col)
                valid_moves = get_valid_moves(game_board, clicked_square)  # Retrieve valid moves for the selected piece
                print(f"Piece selected at {clicked_square}. Valid moves: {[move[-2:] for move in valid_moves]}")
        else:  # If a piece is already selected, attempt to move it
            target_square = f'{num_2_letter[col]}{8 - row}'
            move_uci = None

            # Check if the target square is in valid moves for the selected piece
            for move in valid_moves:
                if move.endswith(target_square):
                    move_uci = move
                    break

            # If a valid move exists to the target square, execute the move
            if move_uci:
                move_piece(selected_pos, (row, col), move_uci)
            selected_piece = None  # Deselect the piece after the move attempt
            selected_pos = None
            valid_moves = []  # Reset valid moves


# Get valid moves using python-chess
def get_valid_moves(board, position):
    square = chess.parse_square(position)
    piece = board.piece_at(square)
    if piece is None:
        return []  # No piece at the position

    # Get all legal moves for the piece on that square
    legal_moves = [move for move in board.legal_moves if move.from_square == square]
    return [move.uci() for move in legal_moves]

def move_piece(from_pos, to_pos, move_uci):
    global game_board

    from_row, from_col = from_pos
    to_row, to_col = to_pos

    # Convert `from_pos` and `to_pos` to chess notation for debugging
    num_2_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
    from_square = f"{num_2_letter[from_col]}{8 - from_row}"
    to_square = f"{num_2_letter[to_col]}{8 - to_row}"


    # Update the chess library board state
    move = chess.Move.from_uci(move_uci)
    if move in game_board.legal_moves:
        game_board.push(move)
        
        # Move the piece on the visual board
        board[to_row][to_col] = board[from_row][from_col]
        board[from_row][from_col] = None
        print(f"moved piece from {from_square} to {to_square}")


#function for castling 

#function for promoting

def promoting():
    return


#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################




class Module(nn.Module):
    def __init__(self, hidden_size):
        super(Module, self).__init__()
        self.conv1 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)          #2 convolutional layers = extract feautres, patterns
        self.conv2 = nn.Conv2d(hidden_size, hidden_size, 3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_size)                                            # batch normalisation layers = reduce overfitting and improve the generalization ability of the model
        self.bn2 = nn.BatchNorm2d(hidden_size)
        self.activation1 = nn.SELU()                                                      #activation function for to add non-linearity into the model,
        self.activation2 = nn.SELU()

    def forward(self, x):
        x_input = torch.clone(x)                                                          # Clone input for skip connection
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = x + x_input                                                                   # Residual connection, should improve learning
        x = self.activation2(x)
        return x

class ChessNet(nn.Module):
    def __init__(self, hidden_layers=4, hidden_size=200):
        super(ChessNet, self).__init__()
        self.hidden_layers = hidden_layers
        self.input_layer = nn.Conv2d(6, hidden_size, 3, stride=1, padding=1)
        self.module_list = nn.ModuleList([Module(hidden_size) for i in range(hidden_layers)]) #ModuleList connected with above "Module"
        self.output_layer = nn.Conv2d(hidden_size, 2, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.input_layer(x)
        x = F.relu(x)

        for i in range(self.hidden_layers):
            x = self.module_list[i](x)

        x = self.output_layer(x)

        return x



def NN():
    model = ChessNet(hidden_layers=4, hidden_size=200)
    model.load_state_dict(torch.load('chess_net.pth', map_location='cpu', weights_only=True))
    model.eval()





def NN():
    model = ChessNet(hidden_layers=4, hidden_size=200)
    model.load_state_dict(torch.load('chess_net.pth', map_location='cpu', weights_only=True))
    model.eval()

    board_tensor = board_2_rep(game_board)

    with torch.no_grad():
        output = model(board_tensor.unsqueeze(0))

    valid_move_found = False
    attempt_count = 0

    # Get a list of all valid moves from the current board state
    legal_moves = list(game_board.legal_moves)

    while not valid_move_found and attempt_count < 10:
        from_square, to_square = interpret_nn_output(output)

        if from_square is None and to_square is None:
            print("No valid moves found in the interpretation. Retrying...")
            attempt_count += 1
            continue  # Retry if no valid move was found

        move = chess.Move(chess.parse_square(from_square), chess.parse_square(to_square))

        # Check if the move is legal
        if move in legal_moves:
            game_board.push(move)

            # Update the visual board
            from_row, from_col = chess.square_rank(move.from_square), chess.square_file(move.from_square)
            to_row, to_col = chess.square_rank(move.to_square), chess.square_file(move.to_square)
            board[to_row][to_col] = board[from_row][from_col]
            board[from_row][from_col] = None

            print(f"Neural Network moved piece from {from_square} to {to_square}")
            valid_move_found = True
        else:
            print(f"Invalid move by NN: {from_square} to {to_square}, trying next best move.")
            attempt_count += 1  # Increment attempt count

    if not valid_move_found:
        print("No valid moves found after several attempts.")





def board_2_rep(board):   #board object from chess package
  pieces = ['p', 'r', 'n', 'b', 'q', 'k']
  layers = []
  for piece in pieces:
    layers.append(create_rep_layer(board, piece)) #create feature map for each chess type
  board_rep = np.stack(layers) #transform feature maps into 3D-tensor
  board_rep_tensor = torch.tensor(board_rep, dtype=torch.float32)
  return board_rep_tensor


def interpret_nn_output(output):
    move_probs = output.view(-1)  # Flatten output to 1D tensor

    # Generate a mask for legal moves: set probability of illegal moves to -inf
    legal_mask = torch.full_like(move_probs, -float('inf'))
    for move in game_board.legal_moves:
        # Calculate the index for each legal move in move_probs
        from_square = move.from_square
        to_square = move.to_square
        move_index = from_square * 8 + to_square
        legal_mask[move_index] = move_probs[move_index]

    # Find the highest-probability legal move
    max_legal_index = legal_mask.argmax()
    from_square = max_legal_index // 8
    to_square = max_legal_index % 8

    return chess.square_name(from_square), chess.square_name(to_square)



# Get valid moves using python-chess
def get_valid_moves(board, position):
    square = chess.parse_square(position)
    piece = board.piece_at(square)
    if piece is None:
        return []  # No piece at the position

    # Get all legal moves for the piece on that square
    legal_moves = [move for move in board.legal_moves if move.from_square == square]
    return [move.uci() for move in legal_moves]

def move_piece(from_pos, to_pos, move_uci):
    global game_board

    from_row, from_col = from_pos
    to_row, to_col = to_pos

    # Convert from_pos and to_pos to chess notation for debugging
    num_2_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
    from_square = f"{num_2_letter[from_col]}{8 - from_row}"
    to_square = f"{num_2_letter[to_col]}{8 - to_row}"


    # Update the chess library board state
    move = chess.Move.from_uci(move_uci)
    if move in game_board.legal_moves:
        game_board.push(move)
        
        # Move the piece on the visual board
        board[to_row][to_col] = board[from_row][from_col]
        board[from_row][from_col] = None
        print(f"moved piece from {from_square} to {to_square}")
































#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################
 

# Game loop
def game_loop():
    init_board()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                handle_click(pygame.mouse.get_pos())

        # Redraw the board and pieces
        screen.fill((0, 0, 0))
        draw_labels()
        draw_board()
        draw_pieces()
        promoting()
        

        # Update the screen
        pygame.display.flip()

# Start the game loop
game_loop()
