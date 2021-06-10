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

INPUT_SHAPE = [1, 224, 224, 3]

resnet_model = tf.keras.applications.resnet50.ResNet50(
    weights="imagenet", include_top=True, input_shape=tuple(INPUT_SHAPE[1:]))

# Wrap the model in a tf.Module to compile it with IREE.
class ResNetModule(tf.Module):

  def __init__(self):
    super(ResNetModule, self).__init__()
    self.m = resnet_model
    self.m.predict = lambda x: self.m.call(x, training=False)
    self.predict = tf.function(
        input_signature=[tf.TensorSpec(INPUT_SHAPE, tf.float32)])(resnet_model.predict)

class TrainableDNN(tf.Module):
  def __init__(self):
    super().__init__()

    # Create a Keras model to train.
    initializer = tf.keras.initializers.GlorotNormal()
    inputs = tf.keras.layers.Input((NUM_COLS, NUM_ROWS, IMG_DIM))
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

if __name__ == "__main__":
    # Compile the model using IREE
    compiler_module = tfc.compile_module(ResNetModule(), exported_names = ["predict"], target_backends=["vmla"], import_only=True)
    # Save module as MLIR file in a directory
    ARITFACTS_DIR = "/tmp"
    mlir_path = os.path.join(ARITFACTS_DIR, "model.mlir")
    with open(mlir_path, "wt") as output_file:
        output_file.write(compiler_module.decode('utf-8'))
    print(f"Wrote MLIR to path '{mlir_path}'")