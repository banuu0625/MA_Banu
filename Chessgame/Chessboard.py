import pygame
import sys #module which works with the interpreter
import chess

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
        # Convert pixel position to board coordinates (x,y), taking margin into account
        col = (pos[0] - MARGIN) // SQUARE_SIZE
        row = (pos[1] - MARGIN) // SQUARE_SIZE

        # Make sure row and col are within the board's 8x8 grid, actually not necessary
        if 0 <= col < 8 and 0 <= row < 8:
            letter_2_num = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h': 7}
            num_2_letter = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
            clicked_square = f'{num_2_letter[col]}{8 - row}'

            if selected_piece is None:  # Select a piece, there should be a board[row][col] and the rest is None
                piece = board[row][col]
                if piece:               #the selected piece detected with the click
                    selected_piece = piece
                    selected_pos = (row, col)
                    valid_moves = get_valid_moves(game_board, clicked_square) #check for valid moves
            else:  # Move the piece to the clicked square if it's a valid move
                target_square = f'{num_2_letter[col]}{8 - row}'
                move_uci = None

                # Look for a move in the valid_moves that matches the target
                for move in valid_moves:
                    if move.endswith(target_square): #if the move to the target square is valid
                        move_uci = move
                        break

                if move_uci:  # If the move is valid, proceed to move the piece
                    move_piece(selected_pos, (row, col), clicked_square, target_square, move_uci)
                selected_piece = None #Reset
                selected_pos = None #Reset
                valid_moves = [] #Reset

#function for castling 

#function for promoting

def promoting():
    for col in range(8):
        if board[1][col] == ChessPiece('white', 'pawn', 'Chessgame/images/white_pawn.png') and board[6][col] == ChessPiece('black', 'pawn', 'Chessgame/images/black_pawn.png'):
            print("Ye")
#function for the CNN to make the corresponding move

# Get valid moves using python-chess
def get_valid_moves(board, position):
    square = chess.parse_square(position)
    piece = board.piece_at(square)
    if piece is None:
        return []  # No piece at the position

    # Get all legal moves for the piece on that square
    legal_moves = [move for move in board.legal_moves if move.from_square == square]
    return [move.uci() for move in legal_moves]

def move_piece(from_pos, to_pos, from_square, to_square, move_uci):
    global game_board

    from_row, from_col = from_pos
    to_row, to_col = to_pos

    # Update the chess library board state
    move = chess.Move.from_uci(move_uci)
    if move in game_board.legal_moves:
        game_board.push(move)
        
        # Move the piece on the visual board
        board[to_row][to_col] = board[from_row][from_col]
        board[from_row][from_col] = None
        print(f"Moved piece from {from_square} to {to_square}")

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
