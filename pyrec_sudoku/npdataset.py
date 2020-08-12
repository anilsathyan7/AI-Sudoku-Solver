from torch.utils.data import DataLoader,Dataset
import numpy as np
import torch

class Sudoku_Dataset(Dataset):

    # One hot encode inputs and labels
    def one_hot_encode(self,s):
      zeros = torch.zeros((81, 9), dtype=torch.float32)
      for a in range(81):
          zeros[a, int(s[a]) - 1] = 1 if int(s[a]) > 0 else 0
      return zeros

    def __init__(self,quiz,solution):
        self.quiz=torch.from_numpy(quiz).to(torch.uint8)
        self.solution=torch.from_numpy(solution).to(torch.uint8)
       
    def __getitem__(self, index):
         x=self.one_hot_encode(self.quiz[index]).float()
         y=self.one_hot_encode(self.solution[index]).float()
         return x, y

    def __len__(self):
        return len(self.quiz)

def create_constraint_mask():
    constraint_mask = torch.zeros((81, 3, 81), dtype=torch.float)
    # Row constraints
    for a in range(81):
        r = 9 * (a // 9)
        for b in range(9):
            constraint_mask[a, 0, r + b] = 1

    # Column constraints
    for a in range(81):
        c = a % 9
        for b in range(9):
            constraint_mask[a, 1, c + 9 * b] = 1

    # Box constraints
    for a in range(81):
        r = a // 9
        c = a % 9
        br = 3 * 9 * (r // 3)
        bc = 3 * (c // 3)
        for b in range(9):
            r = b % 3
            c = 9 * (b // 3)
            constraint_mask[a, 2, br + bc + r + c] = 1

    return constraint_mask

def load_dataset(train_path, solution_path):

  # Load numpy data
  puzzle=np.load(train_path).reshape(-1,81) 
  solution=np.load(solution_path).reshape(-1,81) 

  # Create dataset
  dataset = Sudoku_Dataset(puzzle, solution)

  return dataset
