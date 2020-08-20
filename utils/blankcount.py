
import numpy as np

counts=[]
puz=np.load('puzzle.npy')

# Count number of blanks in each puzzle
for i in range(len(puz)):
    counts.append(np.count_nonzero(puz[i]==0))
 
# Count number of puzzles with unique counts
blank_count=np.array(np.unique(counts, return_counts=True)).T

for i in range(len(blank_count)):
    print("Blanks: {}, Puzzles: {}".format(blank_count[i,0], blank_count[i,1]))
