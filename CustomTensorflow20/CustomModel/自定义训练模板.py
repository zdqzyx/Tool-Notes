# -*- coding: utf-8 -*-
# @Time : 2020/4/26 17:56
# @Author : zdqzyx
# @File : test_training_arg_2.py
# @Software: PyCharm

# 关于checkpoints
#https://code5.cn/so/python-3.x/2479816

import  tensorflow as tf
import numpy as np

params = {
    'dims':32,
    'class_nums':3,
    'epochs':10,
    'checkpoint_path':"./checkpoints/train",
    'checkpoint_do_delete':False
}

#  ============ data ==============
print('data...')
x = np.random.randn(100, params['dims'])
# x = tf.convert_to_tensor(x)
y = np.array([np.random.randint(0, params['class_nums']) for _ in range(100)])
print(x.shape, y.shape)
print()

def checkout_dir(dir_path, do_delete=False):
    import shutil, os
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)

class SimpleModel(tf.keras.Model):

    def __init__(self):
        super(SimpleModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(4, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(params['class_nums'], activation=tf.nn.softmax)
        self.dropout = tf.keras.layers.Dropout(0.5)

    def call(self, inputs, training=None, mask=None):
        print('SimpleModel training ', training)
        x = self.dense1(inputs)
        if training:
            x = self.dropout(x, training=training)
        return self.dense2(x)


class ModelHelper:

    def __init__(self):
        self.model = SimpleModel()
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
        self.optimizer = tf.keras.optimizers.Adam()
        # 主要为了累计一个epoch中的batch的loss，最后求平均，得到一个epoch的loss
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        # 主要为了累计一个epoch中的batch的acc，最后求平均，得到一个epoch的acc
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

        # 检查点 params['checkpoint_path']如果不存在，则创建对应目录；如果存在，且checkpoint_do_delete=True时，则先删除目录在创建
        checkout_dir(dir_path=params['checkpoint_path'], do_delete=params['checkpoint_do_delete'])
        # 检查点
        ckpt = tf.train.Checkpoint(model=self.model,
                                   optimizer=self.optimizer)
        self.ckpt_manager = tf.train.CheckpointManager(ckpt, params['checkpoint_path'], max_to_keep=5)
        # 如果检查点存在，则恢复最新的检查点。
        if self.ckpt_manager.latest_checkpoint:
            status = ckpt.restore(self.ckpt_manager.latest_checkpoint)
            # status = ckpt.restore(self.ckpt_manager.latest_checkpoint).expect_partial() # 当恢复模型只用于预测时，这样就不会有警告
            # status.assert_existing_objects_matched()
            print('Latest checkpoint restored!!')

    @tf.function
    def train_step(self, x, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(x, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, x, labels):
        predictions = self.model(x, training=False)
        t_loss = self.loss_object(labels, predictions)
        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    def train(self, x, y, validation_data):
        for epoch in range(params['epochs']):
            # 在下一个epoch开始时，重置评估指标
            self.train_loss.reset_states()
            self.train_accuracy.reset_states()
            self.test_loss.reset_states()
            self.test_accuracy.reset_states()

            print('train...')
            self.train_step(x, y)

            print('test...')
            self.test_step(validation_data[0], validation_data[1])

            template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
            print(template.format(epoch + 1,
                                  self.train_loss.result(),
                                  self.train_accuracy.result() * 100,
                                  self.test_loss.result(),
                                  self.test_accuracy.result() * 100))

            if (epoch + 1) % 5 == 0:
                ckpt_save_path = self.ckpt_manager.save()
                print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,ckpt_save_path))

    def predict(self, x):
        output = self.model(x, training=False)
        predictions = tf.argmax(output, axis=-1)
        return predictions


    def evaluation(self, x, y):
        self.test_accuracy.reset_states()
        self.test_loss.reset_states()
        self.test_step(x, y)
        template = 'Evaluation || Loss: {}, Accuracy: {}'
        print(template.format(self.test_loss.result(), self.test_accuracy.result() * 100))



model_helper = ModelHelper()
print('Train...')
model_helper.train(x, y, validation_data=(x, y))

print('Evaluation...')
model_helper.evaluation(x, y)

print('Prediction...')
pred = model_helper.predict(x)
print('labels : ', y[:10])
print('pred :', pred[:10])


