import npdataset as d
import model as m
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

# Configure batch size
batch_size = 32

# Create train and validation sets from dataset
full_dataset = d.load_dataset('puzzle.npy','solution.npy')
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_set, test_set = torch.utils.data.random_split(full_dataset, [train_size, test_size])


# Set default device as gpu, if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Generate constraint mask for puzzles
constraint_mask = d.create_constraint_mask().to(device)


# Training data loader
dataloader_ = data.DataLoader(train_set,
                              batch_size=batch_size,
                              pin_memory=True,
                              shuffle=True)

# Validation data loader
dataloader_val_ = data.DataLoader(test_set,
                                  batch_size=batch_size,
                                  pin_memory=True,
                                  shuffle=True)


# Define the loss function
loss = nn.MSELoss()

# Define the model
sudoku_solver = m.SudokuSolver(constraint_mask)
sudoku_solver.to(device)

# Define the optimizer
optimizer = optim.Adam(sudoku_solver.parameters(),
                       lr=0.001,
                       weight_decay=0.000)

# Initialize training metrics
epochs = 10
loss_train = []
loss_val = []

# Training and validation steps
for e in range(epochs):
    for i_batch, ts_ in enumerate(dataloader_):
        puz = ts_[0].to(device, non_blocking=True)
        sol = ts_[1].to(device, non_blocking=True)
        sudoku_solver.train()
        optimizer.zero_grad()
        pred, mat = sudoku_solver(puz)
        ls = loss(pred, sol)
        ls.backward()
        optimizer.step()
        if i_batch%2==0:
          print("Epoch " + str(e) + " batch " + str(i_batch)
              + ": " + str(ls.item()))
        sudoku_solver.eval()

        # Run validation test every 20 iterations
        if i_batch%20==0:
          with torch.no_grad():
            test=next(iter(dataloader_val_))
            puz_t=test[0].to(device, non_blocking=True)
            sol_t=test[1].to(device, non_blocking=True)
 
            rows = torch.randperm(puz_t.shape[0])
            test_pred, test_fill = sudoku_solver(puz_t[rows])
            errors = test_fill.max(dim=2)[1]\
                != sol_t[rows].max(dim=2)[1]
            loss_val.append(errors.sum().item())
            print("Cells in error: " + str(errors.sum().item()))

    # Save the model after every epoch
    torch.save(sudoku_solver,'model.pth')


