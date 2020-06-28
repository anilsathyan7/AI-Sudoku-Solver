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
 
1. **Digit Recognition**

The model was trained using **Adam optimizer** with learning rate 0.001 ~ 0.000001.
 ```
 _________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 30, 30, 32)        320       
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 28, 28, 64)        18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 14, 14, 64)        0         
_________________________________________________________________
dropout_4 (Dropout)          (None, 14, 14, 64)        0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 12544)             0         
_________________________________________________________________
dense_4 (Dense)              (None, 128)               1605760   
_________________________________________________________________
dropout_5 (Dropout)          (None, 128)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 10)                1290      
=================================================================
Total params: 1,625,866
Trainable params: 1,625,866
Non-trainable params: 0
_________________________________________________________________
```
* Loss: 0.14, Accuracy: 96%
* Epochs: 196, Size: 19.6MB

2. **Sudoku Solver**

The model was trained using **Adam optimizer** with learning rate 0.001 ~ 0.000001.
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 9, 9, 81)          810       
_________________________________________________________________
batch_normalization (BatchNo (None, 9, 9, 81)          324       
_________________________________________________________________
p_re_lu (PReLU)              (None, 9, 9, 81)          6561      
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 9, 9, 81)          59130     
_________________________________________________________________
batch_normalization_1 (Batch (None, 9, 9, 81)          324       
_________________________________________________________________
p_re_lu_1 (PReLU)            (None, 9, 9, 81)          6561      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 9, 9, 81)          59130     
_________________________________________________________________
batch_normalization_2 (Batch (None, 9, 9, 81)          324       
_________________________________________________________________
p_re_lu_2 (PReLU)            (None, 9, 9, 81)          6561      
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 9, 9, 162)         13284     
_________________________________________________________________
batch_normalization_3 (Batch (None, 9, 9, 162)         648       
_________________________________________________________________
p_re_lu_3 (PReLU)            (None, 9, 9, 162)         13122     
_________________________________________________________________
flatten (Flatten)            (None, 13122)             0         
_________________________________________________________________
dense (Dense)                (None, 1458)              19133334  
_________________________________________________________________
p_re_lu_4 (PReLU)            (None, 1458)              1458      
_________________________________________________________________
dense_1 (Dense)              (None, 729)               1063611   
_________________________________________________________________
reshape (Reshape)            (None, 9, 9, 9)           0         
_________________________________________________________________
softmax (Softmax)            (None, 9, 9, 9)           0         
=================================================================
Total params: 20,365,182
Trainable params: 20,364,372
Non-trainable params: 810
_________________________________________________________________
```
* Loss: 0.24, Accuracy: 90%
* Epochs: 245, Size: 244.5MB
