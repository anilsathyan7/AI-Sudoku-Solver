# AI-Sudoku-Solver
Solving Sudoku Puzzles With Computer Vision And Neural Networks

**Solving Sudoku Puzzle Using Neural Network**

The **classic sudoku** is a number placing puzzle game with a grid of **9 rows and 9 columns**, partly filled with **numbers 1..9** . We have to fill up the remaining positions such that each **row, columns and 3x3 sub grids** contains numbers 1..9, **without repeatation**.

Here our **input** is an **image of sudoku** puzzle and we need to produce a corresponding **output image** by filling the remaining positions of the input. The pipeline for the **solution** consists of the following steps.

1. Preprocess the input image and remove the background
2. Crop subregions containing digits from the grid
3. Predict numbers from image crops using neural network
4. Predict solution using neural network in an iterative manner
5. Verify the solution and plot the resuts on the input image

![picture](https://drive.google.com/uc?export=view&id=1VnRccmRBtTRFhk3Oq5ThfZbJQ92RYAkB)



**Note:** We use **tensorflow-keras** library for training(prediction) the neural network and **opencv** library for image processing.
