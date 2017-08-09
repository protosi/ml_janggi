'''
Created on 2017. 8. 7.

@author: 3F8VJ32

맵 정보를 받아 좌표정보를 내놓는 인공지능
'''
import tensorflow as tf
import numpy as np

class DQN2:
    def __init__(self, session, output_size, name="main"):
        self.session = session
        self.output_size = output_size
        self.net_name = name
        
        self._build_network()
        
    def _build_network(self):
        with tf.variable_scope(self.net_name):
            self.learn_rate = tf.constant(0.1, dtype=tf.float32)
            
            '''
            이미지 처리 부분, output은 총 576개 (모양은 다름)
            '''
            self._MAP = tf.placeholder(tf.float32, [None, 10, 9, 3], name="input_map")
            net_map = tf.layers.conv2d(self._MAP, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=8, kernel_size=[2,2], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.max_pooling2d(net_map, pool_size=[2, 2], strides=1)
            net_map = tf.reshape(net_map, [-1, 576])
            
            '''
            POS 입력부분 [x1, y1, x2, y2] 로 입력 받는다.
            (-1, 4) 형태는 (-1, 4, 10) 형태로 변환된다.
            이를 (-1, 40)으로 선형시킨다. 
            '''
            '''
            self._POS = tf.placeholder(tf.float32, [None, 4], name="input_pos")
            net_pos = tf.contfib.layers.one_hot_encoding(self._POS, 10)
            net_pos = tf.reshape(net_pos, [-1, 40])
                        
            net_pos = tf.layers.dense(net_pos, 40, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_pos = tf.layers.dense(net_pos, 40, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_pos = tf.layers.dense(net_pos, 40, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            '''
            '''
            두 선형 결과를 결합한다.
            576 + 40 = 616
            '''
            '''
            net = tf.concat([net_map, net_pos], axis=1)
            '''
            net = tf.layers.dense(net_map, 576, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, 576, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, 576, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, self.output_size, activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(),  name="output_pos")
            
            
            self._Qpred = net
            
            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)
            self._predicted = tf.cast(self._Qpred > 0.5 , tf.float32)
            self._accuracy = tf.reduce_mean(tf.cast(tf.equal(self._predicted, self._Y), dtype=tf.float32))
            #optimizer = tf.train.AdamOptimizer(learning_rate=self._learn_rate)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learn_rate)
            self._train = optimizer.minimize(self._loss)
            
    def accuracy_check(self, state, pos):
        feed = {self._MAP: state, self._Y: pos}
        return self.session.run([self._Qpred, self._accuracy], feed_dict=feed)
            
    def predict(self, state) :
        feed = {self._MAP: state}
        return self.session.run(self._Qpred, feed_dict=feed)
    
    def update(self, state: np.ndarray, values: np.ndarray) -> list:
        feed = {self._MAP: state, self._Y: values}
        return self.session.run([self._loss, self._train], feed)