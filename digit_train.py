
'''Digit Recognition With SVHN Dataset'''

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

import numpy as np
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, Callback, ReduceLROnPlateau


# Create your input pipeline

(ds_train, ds_test, ds_extra), ds_info = tfds.load(
    'svhn_cropped',
    split=['train','test','extra'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)
ds_full=ds_train.concatenate(ds_test).concatenate(ds_extra) # Combine all the datasets


# Build training pipeline

def preprocess_img(image, label):

  image = tf.image.rgb_to_grayscale(image) # Convert images to grayscale
  return tf.cast(image, tf.float32) / 255., label # Normalize the images

ds_full = ds_full.map(
    preprocess_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_full = ds_full.cache()
ds_full = ds_full.shuffle(1000)
ds_full = ds_full.batch(128)
ds_full = ds_full.prefetch(tf.data.experimental.AUTOTUNE)


# Create and train the model

CHECKPOINT="/content/drive/My Drive/digit_svhn/digit-{epoch:02d}-{loss:.2f}.hdf5"
LOGS='./logs'

checkpoint = ModelCheckpoint(CHECKPOINT, monitor='loss', verbose=1, save_weights_only=False , save_best_only=True, mode='min')
reduce_lr = ReduceLROnPlateau(factor=0.50, monitor='loss', patience=3, min_lr=0.000001, verbose=1)
tensorboard = TensorBoard(log_dir=LOGS, histogram_freq=0,
                          write_graph=True, write_images=True)

callbacks_list = [checkpoint, tensorboard, reduce_lr] # Configure training callbacks

model = tf.keras.models.Sequential([
 
Conv2D(32, kernel_size=(3, 3),activation='relu', input_shape=(32,32,1)),
Conv2D(64, (3, 3), activation='relu'),
MaxPooling2D(pool_size=(2, 2)),
Dropout(0.25),
Flatten(),
Dense(128, activation='relu'),
Dropout(0.5),
Dense(10)

])
model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.keras.optimizers.Adam(0.001),
    metrics=['accuracy'],
)

model.fit(
    ds_full,
    epochs=500,
    callbacks=callbacks_list
)
