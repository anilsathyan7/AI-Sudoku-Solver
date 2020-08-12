import npdataset as d
import model as m
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim

# Configure the device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the test dataset
test_set = d.load_dataset('fulltest_puzzle.npy','fulltest_solution.npy')
test_loader = data.DataLoader(test_set, batch_size=60, pin_memory=True,shuffle=True)
constraint_mask = d.create_constraint_mask().to(device)

# Load the trained model 
model=torch.load('model.pth').to(device)

# Initialize evaluation metrics
loss_val = []
num_puzzles = len(test_set)
solved=0

with torch.no_grad():
           
           # Load the entire test dataset
           for puz, sol in test_loader:
             puz=puz.to(device, non_blocking=True)
             sol=sol.to(device, non_blocking=True)

           # Infer solution sequentailly
           for i in range(num_puzzles):
            rows = torch.tensor([i])
            test_pred, test_fill = model(puz[rows])
            errors = test_fill.max(dim=2)[1]\
               != sol[rows].max(dim=2)[1]
            loss_val.append(errors.sum().item())
            
            # Count the number of solved puzzles
            if errors.sum().item()==0:
              solved+=1
            x=puz[rows]

            # Print the results
            print("Puzzle number: ", rows.item())
            print("Cells in error: " + str(errors.sum().item()))
            print("Total unfilled cells", (x.sum(dim=2) == 0).sum(dim=1).sum().item())
            print("\n")

print("Total solved: {}/{}".format(solved, num_puzzles))


