
'''Convert numpy array data into csv format'''

import numpy as np

def array2csv(puzzle, solution):

  # Load the entire numpy array into memory
  puzzle=np.load(puzzle).reshape(-1,81)
  solution=np.load(solution).reshape(-1,81)
   
  # Convert array elements into a list of strings
  puzlist=["".join(item) for item in puzzle.astype(str)]
  sollist=["".join(item) for item in solution.astype(str)]

  # Save combined data in csv format
  np.savetxt('sudoku.csv', np.column_stack((puzlist, sollist)), delimiter=',', fmt='%s', header='quizzes,solutions', comments='')

array2csv(puzzle='puzzles.npy',solution='solutions.npy')
