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

![screenshot](https://docs.google.com/drawings/d/e/2PACX-1vRMOwiXYg9cQ4OGfQxvuF2rxkJRHLOoJYJfAFnC2X_Ggxh2BACz1e4M2OV7h3iSjC8vzJp8SR9LP9sY/pub?w=921&h=606)

 We use **tensorflow-keras** library for training(prediction) the neural network and **opencv** library for image processing.
 The **input** sudoku puzzles are assumed to be **images** of printed version of the puzzle.
 ### Training Datasets
 
 The **digit recognition** model was trained using the entire **SVHN dataset**(train, test and extra) in grayscale mode. It is used to classify digits 0 to 9.
 The **sudoku solver** model was trained using a dataset of **10 million** puzzles. The inputs for this model contains **9x9 arrays** of integers representing the puzzles, such that **zeros** represent the **unfilled** positions.
 
 The **numpy dataset** used for training was created by combining the following **two datasets in csv** formats.
 
 1. [One Million Sudoku Puzzles](https://www.kaggle.com/bryanpark/sudoku)
 2. [Nine Million Sudoku Puzzles](https://www.kaggle.com/rohanrao/sudoku)
 
 ### Sudoku Solver Algorithm
 
 A **single iteration** of the model, as such does not seem to produce correct results for all the positions in the input.
 So, we follow a **iterative approach** of feeding the partial solution of one iteration as input to next iteration.
 
* The input is a sudoku matrix of 9x9 with numbers 0...9 as input(i.e 'puzzle').
* Zeros represents the blank spaces in the original puzzle.
* Each iteration produces an output array of 9x9 with numbers 1...9 (i.e 'out').
* For each such output array, 'maxp'(9x9) contains the corresponding probability values.
* For each filled(non-zero) element in input array we set corresponding probability in 'maxp' o -1.
* Now, find the maximum element(single) in 'maxp' and set the corresponding position of input with corresponding values from current output.
* Repeat the iterations with modified input(i.e 'puzzle'), until all elements are filled (ie. no zeros).

The algoritm takes N iterations for solving the entire puzzle, where N represenets the number of unfilled positions.

### Digit Recognition Inputs

* The input puzzle should be a **grayscale or rgb** image. 
* The images should **not be blurry or shaky**.
* It should be a **close-up** image of the puzzle from a **flat** surface.
* The puzzle should be in **printed format** eg.: paper or screen
* The puzzle image should not contain **marks, stains** or unnecessary patterns.
 
 ### Sudoku Models
 
1. **Digit Recognition**

The model was trained using **Adam optimizer** with a learning rate 0.001 ~ 0.000001 and **SCCE loss** function.
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

A **heavy dense and heavy conv** model was trained using the the same dataset.
The following sections shows the overall **summary** of the model and their training results.

**a) Dense model**

The model was trained using **Adam optimizer** with a learning rate 0.001 ~ 0.000001 and **SCCE loss** function.
Here, most of the parameters are contributed by **dense layers** and conv layers are light weight.
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

**b) Conv Model**

The model was trained using **Adam optimizer** with a learning rate 0.001 and **SCCE loss** function.
Here, there are **no dense layers** and conv layers are heavy(filters).

```_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 9, 9, 512)         5120      
_________________________________________________________________
batch_normalization (BatchNo (None, 9, 9, 512)         2048      
_________________________________________________________________
re_lu (ReLU)                 (None, 9, 9, 512)         0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 9, 9, 512)         2359808   
_________________________________________________________________
batch_normalization_1 (Batch (None, 9, 9, 512)         2048      
_________________________________________________________________
re_lu_1 (ReLU)               (None, 9, 9, 512)         0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 9, 9, 512)         2359808   
_________________________________________________________________
batch_normalization_2 (Batch (None, 9, 9, 512)         2048      
_________________________________________________________________
re_lu_2 (ReLU)               (None, 9, 9, 512)         0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 9, 9, 512)         2359808   
_________________________________________________________________
batch_normalization_3 (Batch (None, 9, 9, 512)         2048      
_________________________________________________________________
re_lu_3 (ReLU)               (None, 9, 9, 512)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 9, 9, 512)         2359808   
_________________________________________________________________
batch_normalization_4 (Batch (None, 9, 9, 512)         2048      
_________________________________________________________________
re_lu_4 (ReLU)               (None, 9, 9, 512)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 9, 9, 512)         2359808   
_________________________________________________________________
batch_normalization_5 (Batch (None, 9, 9, 512)         2048      
_________________________________________________________________
re_lu_5 (ReLU)               (None, 9, 9, 512)         0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 9, 9, 512)         2359808   
_________________________________________________________________
batch_normalization_6 (Batch (None, 9, 9, 512)         2048      
_________________________________________________________________
re_lu_6 (ReLU)               (None, 9, 9, 512)         0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 9, 9, 512)         2359808   
_________________________________________________________________
batch_normalization_7 (Batch (None, 9, 9, 512)         2048      
_________________________________________________________________
re_lu_7 (ReLU)               (None, 9, 9, 512)         0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 9, 9, 512)         2359808   
_________________________________________________________________
batch_normalization_8 (Batch (None, 9, 9, 512)         2048      
_________________________________________________________________
re_lu_8 (ReLU)               (None, 9, 9, 512)         0         
_________________________________________________________________
conv2d_9 (Conv2D)            (None, 9, 9, 512)         2359808   
_________________________________________________________________
batch_normalization_9 (Batch (None, 9, 9, 512)         2048      
_________________________________________________________________
re_lu_9 (ReLU)               (None, 9, 9, 512)         0         
_________________________________________________________________
conv2d_10 (Conv2D)           (None, 9, 9, 9)           4617      
=================================================================
Total params: 21,268,489
Trainable params: 21,258,249
Non-trainable params: 10,240
_________________________________________________________________
```
* Loss: 0.10, Accuracy: 96%
* Epochs: 20, Size: 255.3MB


**c) Recurrent Model**

The model was trained using **Adam optimizer** with a learning rate 0.001 and **MSE loss** function.
Here, there are **no conv layers** and it heavily uses dense layers. 

* The **inputs are one-hot-encoded** version of the puzzle **constraints** (row, column and block) and they have **varaible lengths**, proportional to number of unfilled positions. 
* During training, the model sequentially finds the **most probalble output digit** and feeds this partially filled puzzle to the next iteration in  a **recurrent** fashion.
* The **loss** is computed only after the **final step**, when all the remaining positions are filled.

```-----------------------------------------------------------------------
      Layer (type)         Input Shape         Param #     Tr. Param #
=======================================================================
          Linear-1             [1, 27]          14,336          14,336
            ReLU-2            [1, 512]               0               0
          Linear-3            [1, 512]         262,656         262,656
            ReLU-4            [1, 512]               0               0
          Linear-5            [1, 512]         262,656         262,656
            ReLU-6            [1, 512]               0               0
          Linear-7            [1, 512]         262,656         262,656
            ReLU-8            [1, 512]               0               0
          Linear-9            [1, 512]         262,656         262,656
           ReLU-10            [1, 512]               0               0
         Linear-11            [1, 512]         262,656         262,656
           ReLU-12            [1, 512]               0               0
         Linear-13            [1, 512]         262,656         262,656
           ReLU-14            [1, 512]               0               0
         Linear-15            [1, 512]         262,656         262,656
           ReLU-16            [1, 512]               0               0
         Linear-17            [1, 512]         262,656         262,656
           ReLU-18            [1, 512]               0               0
         Linear-19            [1, 512]           4,617           4,617
        Softmax-20              [1, 9]               0               0
=======================================================================
Total params: 2,120,201
Trainable params: 2,120,201
Non-trainable params: 0
-----------------------------------------------------------------------
```
* Loss: 0.005, Accuracy: 94%
* Epochs: 4, Size: 8.6MB

## Testing Sudoku Solver Models

The test dataset consists of **30** puzzles from website: https://1sudoku.com and **30** random puzzles from **newspapers**. They mostly contain difficulties ranging from **easy to medium**. A puzzle is considered to be solved only if **all its elements are predicted correctly** .

| Model Type     |  Performance |
| -------------- | --------- |
| Dense Model |       40/60 |
| Conv Model |      50/60 |
| Recurrent Model |  25/60 |  

## References

* https://github.com/Kyubyong/sudoku
* https://github.com/shivaverma/Sudoku-Solver
* https://github.com/modulai/pytorch_sudoku
* https://github.com/neeru1207/AI_Sudoku
* https://keras.io/examples/vision/mnist_convnet
* https://www.tensorflow.org/datasets/keras_example
* https://aishack.in/tutorials/sudoku-grabber-opencv-plot
* [OpenCV: Geometric Transformations of Images](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_geometric_transformations/py_geometric_transformations.html)
