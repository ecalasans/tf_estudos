'''
Estudos em TensorFlow
Capítulo:  Keras, A High-Level API for TensorFlow
'''

import tensorflow as tf

class MyModel(tf.keras.Model):
    def __init__(self, num_classes=10):
        super(MyModel, self).__init__()

        inputs = tf.keras.Input(shape=(28,28))
        self.x0 = tf.keras.layers.Flatten()
        self.x1 = tf.keras.layers.Dense(512, activation='relu', name='d1')
        self.x2 = tf.keras.layers.Dropout(0.2)
        self.predictions = tf.keras.layers.Dense(10, activation=tf.nn.softmax, name='d2')

    def calls(self, inputs):
        x = self.x0(inputs)
        x = self.x1(x)
        x = self.x2(x)

        return self.predictions(x)

# Utilizando a nova classe
mnist = tf.keras.datasets.mnist
(train_x, train_y), (test_x, test_y) = mnist.load_data()

train_x = tf.cast(train_x/255., tf.float32)
test_x = tf.cast(test_x/255., tf.float32)

train_y = tf.cast(train_y, tf.int64)
test_y = tf.cast(test_y, tf.int64)

model4 = MyModel()
batch_size = 32
epochs = 10

steps_per_epoch = len(train_x.numpy()) // batch_size

print(steps_per_epoch)

model4.compile(optimizer=tf.keras.optimizers.Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model4.fit(train_x, train_y, batch_size=batch_size, epochs=epochs)
model4.evaluate(test_x, test_y)
