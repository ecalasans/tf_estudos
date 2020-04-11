'''
Estudos em TensorFlow
Capítulo:  Keras, A High-Level API for TensorFlow
'''

import tensorflow as tf

mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x, test_x = tf.cast(train_x/255., tf.float32), tf.cast(test_x/255., tf.float32)
train_y, test_y = tf.cast(train_y, tf.int64), tf.cast(test_y, tf.int64)

# Construção do train_dataset
batch_size = 32
buffer_size = 10000

train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(32).shuffle(10000)
train_dataset = train_dataset.map(lambda  x, y: (tf.image.random_flip_left_right(x), y))
train_dataset = train_dataset.repeat()

# Construção do test_dataset
test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size).shuffle(10000)

test_dataset = test_dataset.repeat()

steps_per_epoch = len(train_x)//batch_size
optimiser = tf.keras.optimizers.Adam()

