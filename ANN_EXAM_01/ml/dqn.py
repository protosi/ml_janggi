'''
Created on 2017. 7. 27.

@author: 3F8VJ32
'''
import tensorflow as tf
import numpy as np

class DQN:
    def __init__(self, session, input_size, output_size, name="main"):
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name
        
        self._build_network()
        
    def _build_network(self):
        with tf.variable_scope(self.net_name):
            
            self._X = tf.placeholder(tf.float32, [None, 10, 9, 2], name="input_x")
            
            net = tf.layers.conv2d(self._X, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.conv2d(net, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.conv2d(net, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.conv2d(net, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.conv2d(net, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.conv2d(net, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.conv2d(net, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.conv2d(net, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.conv2d(net, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.conv2d(net, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.max_pooling2d(net, pool_size=[2, 2], strides=1)
            net = tf.reshape(net, [-1, 576])
            net = tf.layers.dense(net, 10, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, 10, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, 10, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, self.output_size, activation=tf.nn.softmax, kernel_initializer=tf.contrib.layers.xavier_initializer(),  name="input_y")
            
            
            self._Qpred = net
            
            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)
            
            #optimizer = tf.train.AdamOptimizer(learning_rate=self._learn_rate)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1)
            self._train = optimizer.minimize(self._loss)
            
    def predict(self, state) :
        feed = {self._X: state}
        return self.session.run(self._Qpred, feed_dict=feed)
    
    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        feed = {self._X: x_stack, self._Y: y_stack}
        return self.session.run([self._loss, self._train], feed)