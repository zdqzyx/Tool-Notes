# -*- coding: utf-8 -*-
# @Time : 2020/4/26 17:56
# @Author : zdqzyx
# @File : test_training_arg_2.py
# @Software: PyCharm

import  tensorflow as tf
import numpy as np

params = {
    'dims':32,
}

#  ============ data ==============
print('data...')
x = np.random.randn(100, params['dims'])
# x = tf.convert_to_tensor(x)
y = np.array([np.random.randint(0,3) for _ in range(100)])
print(x.shape, y.shape)
print()

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


def create_model(params, training):
    inputs = tf.keras.layers.Input((params['dims'],), dtype=tf.float32, name='input')
    internal_model = SimpleModel()
    logits = internal_model(inputs, training)
    model = tf.keras.Model(inputs, logits)
    # model.compile(
    #     optimizer='adam',
    #     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    #     metrics=['accuracy'],
    # )
    # model.summary()
    return model

model = create_model(params)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels, training):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels, training):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)


print('fit... ')
EPOCHS = 5

for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()

    print('train...')
    train_step(x, y)

    print('test...')
    test_step(x, y)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print (template.format(epoch+1,
                         train_loss.result(),
                         train_accuracy.result()*100,
                         test_loss.result(),
                         test_accuracy.result()*100))
