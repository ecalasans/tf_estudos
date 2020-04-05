'''
Estudos em TensorFlow
Capítulo:  Keras, A High-Level API for TensorFlow
'''

import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x, test_x = tf.cast(train_x/255., tf.float32), tf.cast(test_x/255., tf.float32)
train_y, test_y = tf.cast(train_y, tf.int64), tf.cast(test_y, tf.int64)

