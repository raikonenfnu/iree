from iree import runtime as ireert
from iree.tf.support import module_utils
from iree.compiler import tf as tfc
import sys
from absl import app

import numpy as np
import os
import tempfile
import tensorflow as tf

BATCH_SIZE = 1
NUM_CLASSES = 10
NUM_ROWS, NUM_COLS = 28, 28
IMG_DIM = 1

class EdgeDetectionModule(tf.Module):

  @tf.function(input_signature=[tf.TensorSpec([1, 128, 128, 1], tf.float32)])
  def predict(self, image):
    # https://en.wikipedia.org/wiki/Sobel_operator
    sobel_x = tf.constant([[-1.0, 0.0, 1.0],
                           [-2.0, 0.0, 2.0],
                           [-1.0, 0.0, 1.0]],
                          dtype=tf.float32, shape=[3, 3, 1, 1])
    sobel_y = tf.constant([[ 1.0,  2.0,  1.0],
                           [ 0.0,  0.0,  0.0],
                           [-1.0, -2.0, -1.0]],
                          dtype=tf.float32, shape=[3, 3, 1, 1])
    gx = tf.nn.conv2d(image, sobel_x, 1, "SAME")
    gy = tf.nn.conv2d(image, sobel_y, 1, "SAME")
    return tf.math.sqrt(gx * gx + gy * gy)

class TrainableDNN(tf.Module):

  def __init__(self):
    super().__init__()

    # Create a Keras model to train.
    initializer = tf.keras.initializers.GlorotNormal()
    inputs = tf.keras.layers.Input((NUM_COLS, NUM_ROWS, IMG_DIM))
    # x = tf.keras.layers.Conv2D(filters=5,kernel_size=3,strides=1,activation='relu',kernel_initializer=initializer)(inputs)
    # x = tf.keras.layers.Conv2D(filters=10,kernel_size=3,strides=1,activation='relu',kernel_initializer=initializer)(inputs)
    # # x = tf.keras.layers.Conv2D(filters=15,kernel_size=3,strides=1,activation='relu',kernel_initializer=initializer)(inputs)
    # x = tf.keras.layers.Flatten()(x)
    # x = tf.keras.layers.Dropout(0.5)(x)
    # # x = tf.keras.layers.Dense(20,activation='relu',kernel_initializer=initializer)(x)
    # x = tf.keras.layers.Dense(NUM_CLASSES,kernel_initializer=initializer)(x)
    x = tf.keras.layers.Flatten()(inputs)
    x = tf.keras.layers.Dense(NUM_CLASSES,kernel_initializer=initializer)(x)
    x = tf.keras.layers.Dense(128,activation='relu',kernel_initializer=initializer)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES,kernel_initializer=initializer)(x)

    outputs = tf.keras.layers.Softmax()(x)
    self.model = tf.keras.Model(inputs, outputs)
    print(self.model.summary())

    # Create a loss function and optimizer to use during training.
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy()
    self.optimizer = tf.keras.optimizers.SGD(learning_rate=1e-2)

  @tf.function(input_signature=[
      tf.TensorSpec([BATCH_SIZE, NUM_ROWS, NUM_COLS, IMG_DIM])  # inputs
  ])
  def predict(self, inputs):
    return self.model(inputs, training=False)

  # # We compile the entire training step by making it a method on the model.
  # @tf.function(input_signature=[
  #     tf.TensorSpec([BATCH_SIZE, NUM_ROWS, NUM_COLS, IMG_DIM]),  # inputs
  #     tf.TensorSpec([BATCH_SIZE], tf.int32)  # labels
  # ])
  # def learn(self, inputs, labels):
  #   # Capture the gradients from forward prop...
  #   with tf.GradientTape() as tape:
  #     probs = self.model(inputs, training=True)
  #     loss = self.loss(labels, probs)

  #   # ...and use them to update the model's weights.
  #   variables = self.model.trainable_variables
  #   gradients = tape.gradient(loss, variables)
  #   self.optimizer.apply_gradients(zip(gradients, variables))
  #   return loss

if __name__ == "__main__":
    # Compile the model using IREE
    compiler_module = tfc.compile_module(TrainableDNN(), exported_names = ["predict"], target_backends=["vmla"], import_only=True, save_temp_iree_input=True)
    # Save module as MLIR file in a directory
    ARITFACTS_DIR = "/tmp"
    mlir_path = os.path.join(ARITFACTS_DIR, "model.mlir")
    with open(mlir_path, "wt") as output_file:
        output_file.write(compiler_module.decode('utf-8'))
    print(f"Wrote MLIR to path '{mlir_path}'")