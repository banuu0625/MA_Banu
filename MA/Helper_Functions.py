from preprocessing import board_2_rep, letter_2_num
from Training import model
import numpy as np
import torch
import chess
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
    model.eval()  
    with torch.no_grad():  
        output = model(x)
    return output
def choose_move(board, player, color):
    legal_moves = list(board.legal_moves)
    # Check for immediate checkmate
    move = check_mate_single(board)
    if move is not None:
        return move
    # Prepare the input for the model
    x = torch.Tensor(board_2_rep(board)).float().to('cuda')
    if color == chess.BLACK:
        x *= -1
    x = x.unsqueeze(0) 
    move = predict(x)  
    # Move output to CPU for NumPy operations
    move = move.cpu()
    vals = []
    froms = [str(legal_move)[:2] for legal_move in legal_moves]
    froms = list(set(froms))
    for from_ in froms:
        rank = 8 - int(from_[1])  
        file = letter_2_num[from_[0]]  
        # Ensure indexing is correct for the "from" predictions
        val = move[0, 0, rank, file].item()  
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
            rank_to = 8 - int(to[1])  
            file_to = letter_2_num[to[0]]  
            # Ensure indexing is correct for the "to" predictions
            val = move[0, 1, rank_to, file_to].item() 
            vals.append(val)
        else:
            vals.append(0)
    choosen_move = legal_moves[np.argmax(vals)]
    return choosen_move
def self_play_evaluation(board, model, color=chess.WHITE):
    # Reset the board and play a game
    board.reset()
    player = 1 if color == chess.WHITE else -1
    for move_num in range(100):  # Simulate up to 100 moves
        move = choose_move(board, player, color)  
        if move is not None:
            board.push(move)  
        else:
            print(f"No legal moves available. Game over after {move_num} moves.")
            break
        if board.is_game_over():
            print(f"Game over after {move_num} moves. Result: {board.result()}")
            break
        # Switch player and color for the next move
        player *= -1
        color = chess.BLACK if color == chess.WHITE else chess.WHITE
    return board