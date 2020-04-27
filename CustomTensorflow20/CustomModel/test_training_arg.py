# -*- coding: utf-8 -*-
# @Time : 2020/4/26 17:56
# @Author : zdqzyx
# @File : test_training_arg.py
# @Software: PyCharm

import  tensorflow as tf
import numpy as np

params = {
    'dims':32,
}

class SimpleModel(tf.keras.Model):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=None, mask=None):
        print('SimpleModel training ', training)
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        return self.dense2(x)


def create_model(params):
    inputs = tf.keras.layers.Input((params['dims'],), dtype=tf.float32, name='input')
    internal_model = SimpleModel()
    logits = internal_model(inputs, )
    model = tf.keras.Model(inputs, logits)

    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'],
    )

    model.summary()
    return model

model = create_model(params)

#  ============ data ==============
x = np.random.randn(100, params['dims'])
# x = tf.convert_to_tensor(x)
y = np.array([np.random.randint(0, params['dims']) for _ in range(100)])
print(x.shape, y.shape)

print('fit... ')
model.fit(x, y, epochs=3, validation_split=0.2, verbose=0)

print('evalue... ')
model.evaluate(x, y, verbose=2)

print('predict...')
pred = model.predict(x[:10])