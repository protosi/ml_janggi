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
        
    def _build_network(self, h_size=25, l_rate=0.01):
        with tf.variable_scope(self.net_name):
            
            
            self._XChoMap = tf.placeholder(tf.float32, [None, None], name="input_x_cho_map")
            self._XHanMap = tf.placeholder(tf.float32, [None, None], name="input_x_han_map")
            self._XPos = tf.placeholder(tf.float32, [None, self.input_size], name="input_x_pos")
            
            # ?*10*9*1 형태의 4차원 배열로 reshape한다.
            choMapReshape = tf.reshape(self._XChoMap, [-1, 10, 9, 1])
            hanMapReshape = tf.reshape(self._XHanMap, [-1, 10, 9, 1])
            
            # cho Convoluation
            choConv2d1 = tf.layers.conv2d(choMapReshape, filters=8, kernel_size=[3,3], padding="same", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            choConv2d2 = tf.layers.conv2d(choConv2d1, filters=1, kernel_size=[3,3], padding="same", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            choMaxPool3 = tf.layers.max_pooling2d(choConv2d2, pool_size=[3,3], strides=1, padding="same")
            choMapFlat = tf.reshape(choMaxPool3[-1, 90])
            choDense = tf.layers.dense(choMapFlat, 100, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            # han Convoluation
            hanConv2d1 = tf.layers.conv2d(hanMapReshape, filters=8, kernel_size=[3,3], padding="same", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            hanConv2d2 = tf.layers.conv2d(hanConv2d1, filters=1, kernel_size=[3,3], padding="same", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            hanMaxPool3 = tf.layers.max_pooling2d(hanConv2d2, pool_size=[3,3], strides=1, padding="same")
            hanMapFlat = tf.reshape(hanMaxPool3[-1, 90])
            hanDense = tf.layers.dense(hanMapFlat, 100, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            mapSum = tf.add(choDense, hanDense);
            
            # Pos 
            pos = tf.contrib.layers.one_hot_encoding(self._XPos, 10)
            posFlat = tf.reshape(pos, [-1, 40])
            posDense = tf.layers.dense(posFlat, 100, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            posSum = tf.add(mapSum, posDense)
            
            sumDense1 = tf.layers.dense(posSum, 50, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            sumDense2 = tf.layers.dense(sumDense1, 20, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(sumDense2, 1, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="output_y")

            self._Qpred = net
            
            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)
            optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self._train = optimizer.minimize(self._loss)
            
    def predict(self, choMap, hanMap, pos) :
        feed = {self._XChoMap: choMap, self._XHanMap: hanMap, self._XPos: pos}
        return self.session.run(self._Qpred, feed_dict=feed)
    
    def update(self, choMap, hanMap, pos, y_stack):
        feed = {self._XChoMap: choMap, self._XHanMap: hanMap, self._XPos: pos, self._Y: y_stack}
        return self.session.run([self._loss, self._train], feed_dict=feed)