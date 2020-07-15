
'''Convert csv data into numpy array format'''

import numpy as np

puzzle, solution = [], []
data = np.genfromtxt('sudoku.csv',dtype='S81',delimiter=',', skip_header=1)

# Extract and save the puzzle characters as numpy arrays 
for row in data:
   
   puz = np.frombuffer(row[0], np.int8) - 48
   puz = puz.reshape((9,9))
   puzzle.append(puz)
   
np.save('puzzle.npy', np.array(puzzle))

del puzzle

# Extract and save the solution characters as numpy arrays  
for row in data:
   
   sol = np.frombuffer(row[1], np.int8) - 48
   sol = sol.reshape((9,9))
   solution.append(sol)

np.save('solution.npy', np.array(solution))

