
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
CHECKPOINT="cnn_ckpt/sudoku_cnn-{epoch:02d}-{loss:.2f}.hdf5"
LOGS='./cnn_logs'

# Get number of elements in the dataset
num_train=len(x_train)

# Normalize the input to -0.5 to 0.5 range
def normalize_img(image, label):
  return (tf.cast(image, tf.float32) / 9.)-0.5, label

# Configure the data loader for training
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.map(normalize_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE).shuffle(buffer_size=1000).repeat().batch(BATCH_SIZE)


# Define the model architecture
model = tf.keras.Sequential([

    tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same', input_shape=(9,9,1)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='same'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.ReLU(),

    tf.keras.layers.Conv2D(filters=9, kernel_size=1, padding='same'),
])

# Compile the model with adam optimizer and scce loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['sparse_categorical_accuracy'])

# Show the model summary
model.summary()

# Configure the training callbacks 
checkpoint = ModelCheckpoint(CHECKPOINT, monitor='sparse_categorical_accuracy', verbose=1, save_weights_only=False , save_best_only=True, mode='max') 
reduce_lr = ReduceLROnPlateau(monitor='sparse_categorical_accuracy', factor=0.60, patience=3, min_lr=0.000001, verbose=1, mode='max')
tensorboard = TensorBoard(log_dir=LOGS, histogram_freq=0,
                          write_graph=True, write_images=True)


callbacks_list = [checkpoint, tensorboard, reduce_lr]

# Train the model
model_history = model.fit(train_dataset, epochs=EPOCHS,
                          steps_per_epoch=num_train//BATCH_SIZE,
                          callbacks=callbacks_list)
