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
            
            self._learn_rate = tf.placeholder(tf.float32, [], name="input_learning_rate")
            self._XChoMap = tf.placeholder(tf.float32, [None, 10, 9], name="input_x_cho_map")
            self._XHanMap = tf.placeholder(tf.float32, [None, 10, 9], name="input_x_han_map")
            self._XPos = tf.placeholder(tf.int32, [None, self.input_size], name="input_x_pos")
            self._XFlag = tf.placeholder(tf.int32, [None, 1], name="input_x_flag")
            # ?*10*9*1 형태의 4차원 배열로 reshape한다.
            choMapReshape = tf.reshape(self._XChoMap, [-1, 10, 9, 1])
            hanMapReshape = tf.reshape(self._XHanMap, [-1, 10, 9, 1])
            
            # x flag
            flag = tf.contrib.layers.one_hot_encoding(self._XFlag, 2)
            flagFlat = tf.reshape(flag, [-1, 2])
            #flagDense = tf.layers.dense(flagFlat, 100, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            #flagDense10 = tf.layers.dense(flagFlat, 10, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            # cho Convoluation
            choConv2d1 = tf.layers.conv2d(choMapReshape, filters=1, kernel_size=[3,3], padding="same", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            choConv2d2 = tf.layers.conv2d(choConv2d1, filters=1, kernel_size=[3,3], padding="same", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            choMaxPool3 = tf.layers.max_pooling2d(choConv2d2, pool_size=[3,3], strides=1, padding="same")
            choMapFlat = tf.reshape(choMaxPool3,[-1, 90])
            choDense1 = tf.concat([choMapFlat, flagFlat], 1)
            choDense = tf.layers.dense(choDense1, 90, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            # han Convoluation
            hanConv2d1 = tf.layers.conv2d(hanMapReshape, filters=1, kernel_size=[3,3], padding="same", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            hanConv2d2 = tf.layers.conv2d(hanConv2d1, filters=1, kernel_size=[3,3], padding="same", activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            hanMaxPool3 = tf.layers.max_pooling2d(hanConv2d2, pool_size=[3,3], strides=1, padding="same")
            hanMapFlat = tf.reshape(hanMaxPool3,[-1, 90])
            hanDense1 = tf.concat([hanMapFlat, flagFlat], 1)
            hanDense = tf.layers.dense(hanDense1, 90, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            mapSum1 = tf.concat([choDense, hanDense],1);
            mapSum = tf.layers.dense(mapSum1, 180, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            # Pos 
            pos = tf.contrib.layers.one_hot_encoding(self._XPos, 10)
            posFlat = tf.reshape(pos, [-1, 40])
            posDense = tf.layers.dense(posFlat, 40, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())

            
            posSum = tf.concat([posDense, mapSum], 1)
            
            sumDense = tf.layers.dense(posSum, 220, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            net = tf.layers.dense(sumDense, 1, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer(), name="output_y")

            self._Qpred = net
            
            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)
            #optimizer = tf.train.AdamOptimizer(learning_rate=self._learn_rate)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self._learn_rate)
            self._train = optimizer.minimize(self._loss)
            
    def predict(self, choMap, hanMap, pos, flag) :
        feed = {self._XChoMap: choMap, self._XHanMap: hanMap, self._XPos: pos, self._XFlag: flag}
        return self.session.run(self._Qpred, feed_dict=feed)
    
    def update(self, choMap, hanMap, pos, flag,  y_stack, learn_rate):
        feed = {self._XChoMap: choMap, self._XHanMap: hanMap, self._XPos: pos, self._XFlag: flag, self._Y: y_stack, self._learn_rate: learn_rate}
        return self.session.run([self._loss, self._train], feed_dict=feed)