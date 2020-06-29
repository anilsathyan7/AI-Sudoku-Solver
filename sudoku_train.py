
'''Sudoku solver training'''

import numpy as np
import random
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, BatchNormalization, PReLU, Input, Flatten, Dense, Reshape, AveragePooling2D

# Load the inputs and labels as UINT8 numpy array
puzzle=np.load('puzzle.npy')
solution=np.load('solution.npy')

# Add channel axis and subtract one from elements of solution
x_train, y_train = puzzle[...,np.newaxis], solution[..., np.newaxis] - 1

# Delete original arrays from memory
del puzzle
del solution

# Print the shape and unique values of both arrays
print("x_train:", x_train.shape, "y_train:", y_train.shape)
print("x values:", np.unique(x_train))
print("y_values:",np.unique(y_train))

# Configure number of epochs, batch size and shuffle buffer size
EPOCHS=300
BATCH_SIZE = 128
SHUFFLE_BUFFER_SIZE = 1000

# Configure paths for checkpoints and training logs
CHECKPOINT="/content/ckpt/sudoku-{epoch:02d}-{loss:.2f}.hdf5"
LOGS='./logs'

# Get number of elements in the dataset
num_train=len(x_train)

# Runtime data augmentations
def random_aug(image, label):

  image, label= (tf.cast(image, tf.float32) / 9.)-0.5, label

  aug=np.random.choice([0,1])
  if aug:
    image, label = rotate_and_flip(image, label)
    image, label = inter_block_shuffle(image, label)
    image, label = intra_block_shuffle(image, label)

  return image, label

# Random rotation and flip
@tf.function
def rotate_and_flip(image, label):
 
  seed1,seed2=1,2

  num = np.random.choice([0,1,2,3])
  image=tf.image.rot90(image, k=num)
  label=tf.image.rot90(label, k=num)

  image = tf.image.random_flip_left_right(image, seed=seed1)
  label = tf.image.random_flip_left_right(label, seed=seed1)

  image = tf.image.random_flip_up_down(image, seed=seed2)
  label = tf.image.random_flip_up_down(label, seed=seed2)

  return image, label

# Shuffle the positions of 3x9 row blocks or 9x3 column blocks
def inter_block_shuffle(image, label):
 
  shifts = np.random.choice([0,3,6])
  axes   = np.random.choice([0,1])

  image = tf.roll(image, shift=shifts, axis=axes)
  label = tf.roll(label, shift=shifts, axis=axes)

  return image, label

# Shuffle the row positions within each 3x9 row blocks
def intra_block_shuffle(image, label):
   
  iarray=np.arange(9)
  random.shuffle(iarray[:3])
  random.shuffle(iarray[3:6])
  random.shuffle(iarray[6:])
  indices=iarray.reshape((9,1))

  image, label = tf.squeeze(image), tf.squeeze(label)

  image = tf.expand_dims(tf.scatter_nd(indices,image,shape=(9,9)),axis=-1)
  label = tf.expand_dims(tf.scatter_nd(indices,label,shape=(9,9)), axis=-1)

  return image, label

# Normalize the input to -0.5 to 0.5 range
def normalize_img(image, label):
  return (tf.cast(image, tf.float32) / 9.)-0.5, label

# Configure the data loader for training
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(random_aug, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE).shuffle(buffer_size=1000).repeat().batch(BATCH_SIZE)


# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=81, kernel_size=3, padding='same', input_shape=(9,9,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.PReLU(),

    tf.keras.layers.Conv2D(filters=81, kernel_size=3, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.PReLU(),

    tf.keras.layers.Conv2D(filters=81, kernel_size=3, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.PReLU(),

    tf.keras.layers.Conv2D(filters=81*2, kernel_size=1, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.PReLU(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=81*18),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Dense(units=81*9),
    
    tf.keras.layers.Reshape((9, 9 , 9)),
    tf.keras.layers.Softmax()
])

# Compile the model with adam optimizer and scce loss
model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

# Show the model summary
model.summary()

# Configure the training callbacks 
checkpoint = ModelCheckpoint(CHECKPOINT, monitor='loss', verbose=1, save_weights_only=False , save_best_only=True, mode='min') 
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.60, patience=3, min_lr=0.000001, verbose=1)
tensorboard = TensorBoard(log_dir=LOGS, histogram_freq=0,
                          write_graph=True, write_images=True)

callbacks_list = [checkpoint, tensorboard, reduce_lr]


# Train the model
model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=num_train//BATCH_SIZE,
                          callbacks=callbacks_list)
