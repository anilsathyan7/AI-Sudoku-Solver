# AI-Sudoku-Solver
Solving Sudoku Puzzles With Computer Vision And Neural Networks

**Solving Sudoku Puzzle Using Neural Network**

The **classic sudoku** is a number placing puzzle game with a grid of **9 rows and 9 columns**, partly filled with **numbers 1..9** . We have to fill up the remaining positions such that each **row, columns and 3x3 sub grids** contains numbers 1..9, **without repeatation**.

Here our **input** is an **image of sudoku** puzzle and we need to produce a corresponding **output image** by filling the remaining positions of the input. The pipeline for the **solution** consists of the following steps.

1. Preprocess the input image and remove the background
2. Crop ROI's containing digits from the grid
3. Predict numbers from image crops using neural network
4. Predict solution using neural network in an iterative manner
5. Verify the solution and plot the resuts on the input image

![screenshot](https://drive.google.com/uc?export=view&id=1VnRccmRBtTRFhk3Oq5ThfZbJQ92RYAkB)

 We use **tensorflow-keras** library for training(prediction) the neural network and **opencv** library for image processing.
 The **input** sudoku puzzeles are assumed to be **images** of printed version of the puzzle.
 ### Dataset
 
 The **digit recognition** model was trained using the entire **SVHN dataset**(train, test and extra) in grayscale mode. It is used to classify digits 0 to 9.
 The **sudoku solver** model was trained using a dataset of **10 million** puzzles. The inputs for this model contains **9x9 arrays** of integers representing the puzzles, such that **zeros** represent the **unfilled** positions.
 
 ### Algorithm
 
 A **single iteration** of the model, as such does not seem to produce correct results for all the positions in the input.
 So, we follow a **iterative approach** of feeding the partial solution of one iteration as input to next iteration.
 
* The input is a sudoku matrix of 9x9 with numbers 0...9 as input(i.e 'puzzle').
* Zeros represents the blank spaces in the original puzzle.
* Each iteration produces an output array of 9x9 with numbers 1...9 (i.e 'out').
* For each such output array, 'maxp'(9x9) contains the corresponding probability values.
* For each filled(non-zero) element in input array we set corresponding probability in 'maxp' o -1.
* Now, find the maximum elements in 'maxp' and set the corresponding positions of input with corresponding values from current output.
* Repeat the iterations with modified input(i.e 'puzzle'), until all elements are filled (ie. no zeros).

The algoritm takes **atmost 81 iterations** for solving the entire puzzle.
 
 ### Models
