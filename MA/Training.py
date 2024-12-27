from preprocessing import chess_data
from CNN_configuration import ChessNet, ChessDataset, DataLoader, data_train_loader
from Helper_Functions import self_play_evaluation
import torch.nn as nn
import torch.optim as optim
import chess

# Make sure the model is on the GPU
model = ChessNet(hidden_layers=4, hidden_size=200).to('cuda')  
x, y = next(iter(data_train_loader)) 
x, y = x.float().to('cuda'), y.to('cuda') 
# Forward pass through the model
output = model(x)
# Loss calculation
metric_from = nn.CrossEntropyLoss()
metric_to = nn.CrossEntropyLoss()
# Calculating the loss for "from" and "to" positions
loss_from = metric_from(output[:,0,:], y[:,0,:])
loss_to = metric_to(output[:,1,:], y[:,1,:])
loss = loss_from + loss_to
data_train = ChessDataset(chess_data['AN'])                                                
data_train_loader = DataLoader(data_train, batch_size = 32, shuffle = True, drop_last=True)
# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)  # model is your neural network
num_epochs = 100
# After each epoch
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for i, (x, y) in enumerate(data_train_loader):
        x, y = x.float().to('cuda'), y.to('cuda')
        optimizer.zero_grad()
        output = model(x)
        loss_from = metric_from(output[:, 0, :], y[:, 0])
        loss_to = metric_to(output[:, 1, :], y[:, 1])
        loss = loss_from + loss_to
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 100 == 99:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(data_train_loader)}], Loss: {running_loss / 100:.4f}')
            running_loss = 0.0
    # After each epoch, run self-play for evaluation
    print(f"Evaluating model after epoch {epoch + 1}")
    board = chess.Board()
    final_board = self_play_evaluation(board, model, color=chess.WHITE)
    print(final_board)


