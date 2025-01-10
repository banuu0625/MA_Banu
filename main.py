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
import asyncio

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


#column index mapping - change letters into numbers and vice versa
letter_2_num = {'a':0, 'b':1, 'c':2, 'd':3, 'e':4, 'f':5, 'g':6, 'h':7}
num_2_letter = {0:'a', 1:'b', 2:'c', 3:'d', 4:'e', 5:'f', 6:'g', 7:'h'}

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
        screen.blit(text, (i * SQUARE_SIZE + MARGIN + SQUARE_SIZE // 2 - text.get_width() // 2, MARGIN // 4)) 
        
        # Draw at the bottom
        screen.blit(text, (i * SQUARE_SIZE + MARGIN + SQUARE_SIZE // 2 - text.get_width() // 2, SCREEN_HEIGHT - MARGIN // 1.5)) 
    
    # Numbers (1 to 8) - for left and right
    for i in range(8):
        text = font.render(str(8 - i), True, WHITE)
        
        # Draw on the left
        screen.blit(text, (MARGIN // 4, i * SQUARE_SIZE + MARGIN + SQUARE_SIZE // 2 - text.get_height() // 2)) 
        
        # Draw on the right
        screen.blit(text, (SCREEN_WIDTH - MARGIN // 1.5, i * SQUARE_SIZE + MARGIN + SQUARE_SIZE // 2 - text.get_height() // 2))


def handle_click(pos, color=chess.WHITE, promotion_choice=None):
    global selected_piece, selected_pos, valid_moves, initial_square, move_uci

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
            if piece and piece.color == 'white':  # Ensure correct player's turn
                selected_piece = piece
                selected_pos = (row, col)
                valid_moves = get_valid_moves(game_board, clicked_square)  # Retrieve valid moves for the selected piece
                print(f"Piece selected at {clicked_square}. Valid moves: {[move[-2:] for move in valid_moves]}")
                initial_square = clicked_square
        else:
            # Try moving the selected piece
            target_square = clicked_square
            if target_square in [move[-2:] for move in valid_moves]:  # Check if move is valid
                # Make the move in game logic
                game_move = [move for move in valid_moves if move.endswith(target_square)][0]
                game_board.push_uci(game_move)
                print(f"Moved piece from {game_move[:2]} to {target_square}.")

                # Handle castling
                if game_move == "e1g1":  # White kingside castling
                    board[7][6] = board[7][4]  # Move king
                    board[7][5] = board[7][7]  # Move rook
                    board[7][4] = None
                    board[7][7] = None
                    print("Castled King Side")
                elif game_move == "e1c1":  # White queenside castling
                    board[7][2] = board[7][4]  # Move king
                    board[7][3] = board[7][0]  # Move rook
                    board[7][4] = None
                    board[7][0] = None
                    print("Castled Queen Side")
                
                elif target_square[-2:] == ('a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8') and board[row][col] == ChessPiece('white', 'pawn', 'Chessgame/images/white_pawn.png'):

                    if promotion_choice is None:
                        promotion_choice = 'q'  # Default to queen
                    elif promotion_choice.lower() not in ['q', 'r', 'b', 'n']:
                        print("Invalid promotion choice. Defaulting to queen.")
                        promotion_choice = 'q'
                    
                    # Modify the move with the promotion choice
                    game_move = game_move[:-1] + promotion_choice.lower()

                    # Push the move to the game board
                    game_board.push_uci(game_move)
                    print(f"Moved piece from {game_move[:2]} to {target_square} with promotion to {promotion_choice.upper() if promotion_choice else 'Q'}.")

                    # Update visual board
                    board[row][col] = promotion_choice.upper() if game_move.endswith(('q', 'r', 'b', 'n')) else selected_piece
                    selected_row, selected_col = selected_pos
                    board[selected_row][selected_col] = None

                    # Reset selection
                    selected_piece = None
                    selected_pos = None
                    valid_moves = []              

                else:
                    # Update visual board for normal moves
                    board[row][col] = selected_piece
                    selected_row, selected_col = selected_pos
                    board[selected_row][selected_col] = None

                # Reset selection
                selected_piece = None
                selected_pos = None
                valid_moves = []
            else:
                print("Invalid move. Select another square.")
                # Reset selection if invalid move
                selected_piece = None
                selected_pos = None
                valid_moves = []







# Get valid moves using python-chess
def get_valid_moves(board, position):
    square = chess.parse_square(position)
    piece = board.piece_at(square)
    if piece is None:
        return []  # No piece at the position

    # Get all legal moves for the piece on that square
    legal_moves = [move for move in board.legal_moves if move.from_square == square]
    return [move.uci() for move in legal_moves]

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

#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################
#################################################################################################################################################################################

model = ChessNet(hidden_layers=4, hidden_size=200)
model.load_state_dict(torch.load('NNs\chess_net250.pth', map_location='cpu', weights_only=True))
model.eval()




def check_mate_single(board):
  board = board.copy()
  legal_moves = list(board.legal_moves)
  for move in legal_moves:
    board.push_uci(str(move))
    if board.is_checkmate():
      move = board.pop()
      return move
    _ = board.pop()


def distribution_over_moves(vals):
  probs = np.array(vals)
  probs = np.exp(probs)
  probs = probs/probs.sum()
  probs = probs ** 3
  probs = probs /probs.sum()
  return probs

# Predict function
def predict(x):
     # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient calculations for inference
        output = model(x)
    return output

def choose_move(board, color=chess.BLACK):
    legal_moves = list(board.legal_moves)

    # Check for immediate checkmate
    move = check_mate_single(board)
    if move is not None:
        return move

    # Prepare the input for the model
    x = torch.Tensor(board_2_rep(board)).float()  # Use CPU instead
    if color == chess.BLACK:
        x *= -1
    x = x.unsqueeze(0)  # Add batch dimension
    move = predict(x)  # Model prediction, should return (1, 2, 8, 8)

    # Move output to CPU for NumPy operations
    move = move.cpu()

    vals = []
    froms = [str(legal_move)[:2] for legal_move in legal_moves]
    froms = list(set(froms))

    for from_ in froms:
        rank = 8 - int(from_[1])  # Convert rank to index (0-7)
        file = letter_2_num[from_[0]]  # Convert file (a-h) to index (0-7)

        # Ensure indexing is correct for the "from" predictions
        val = move[0, 0, rank, file].item()  # Convert tensor to scalar for compatibility with NumPy
        vals.append(val)

    # Convert values to probability distribution
    probs = distribution_over_moves(vals)

    # Choose a "from" square based on probabilities
    choosen_from = str(np.random.choice(froms, size=1, p=probs)[0])[:2]

    vals = []
    for legal_move in legal_moves:
        from_ = str(legal_move)[:2]
        if from_ == choosen_from:
            to = str(legal_move)[2:]
            rank_to = 8 - int(to[1])  # Convert rank to index
            file_to = letter_2_num[to[0]]  # Convert file to index

            # Ensure indexing is correct for the "to" predictions
            val = move[0, 1, rank_to, file_to].item()  # Convert tensor to scalar for compatibility with NumPy
            vals.append(val)
        else:
            vals.append(0)

    # Choose the final move based on the highest value in 'to' position predictions
    choosen_move = legal_moves[np.argmax(vals)]
    return choosen_move.uci()
    

def NN_move():
    """
    Executes the AI's chosen move, updating both the visual board and the game logic board.
    """
    global game_board, board

    # Get the move chosen by the AI
    move_uci = choose_move(game_board, chess.BLACK)
    if not move_uci:
        print("AI could not make a move.")
        return
    
    
    # Convert UCI move to source and destination squares
    move = chess.Move.from_uci(move_uci)
    from_square = move.from_square
    to_square = move.to_square

    # Convert to (row, col) for the visual board
    from_row, from_col = 7 - (from_square // 8), from_square % 8
    to_row, to_col = 7 - (to_square // 8), to_square % 8



    if move_uci == "e8g8":  # Black kingside castling
        game_board.push(move)
        board[0][6] = board[0][4]  # Move king
        board[0][5] = board[0][7]  # Move rook
        board[0][4] = None
        board[0][7] = None
        print("Castled King Side")
    
    elif move_uci == "e8c8":  # Black queenside castling
        game_board.push(move)
        board[0][2] = board[0][4]  # Move king
        board[0][3] = board[0][0]  # Move rook
        board[0][4] = None
        board[0][0] = None
        game_board.push(move)
        print("Castled Queen Side")
        
    else:
        # Update python-chess board
        game_board.push(move)

        # Update visual board
        board[to_row][to_col] = board[from_row][from_col]  # Move piece
        board[from_row][from_col] = None  # Clear original square

        # Debug information
        print(f"AI moved piece from ({from_row}, {from_col}) to ({to_row}, {to_col}) using UCI: {move_uci}")





# Game loop
async def game_loop():


    init_board()
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                handle_click(pygame.mouse.get_pos())  # White's turn (human)
                if game_board.turn == chess.BLACK:  # If it's now Black's turn
                    NN_move()   # Make the NN move for black 
                  
        # Redraw the board and pieces
        screen.fill((0, 0, 0))
        draw_labels()
        draw_board()
        draw_pieces()
        # Update the screen
        pygame.display.flip()
        await asyncio.sleep(0)
          

asyncio.run(game_loop())