'''
Created on 2017. 8. 8.

@author: 3F8VJ32

위치 좌표 인공지능 새로 작성
맵정보와 액션을 받으면, 그에 대한 Qvalue가 나오는 신경망

state, action => value 
'''
import tensorflow as tf
import numpy as np

class ChessMoveAI:
    def __init__(self, session, name="main"):
        self.session = session
        self.output_size = 1
        self.net_name = name
        
        self._build_network()
        
    def _build_network(self):
        with tf.variable_scope(self.net_name):
            self.learn_rate = tf.constant(0.1, dtype=tf.float32)
            
            '''
            이미지 처리 부분, output은 총 576개 (모양은 다름)
            5개층이 너무 적지는 않을까... 고민
            일단 10개로 늘려보고 많으면 줄이자.
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
            
            net_map = tf.layers.dense(net_map, 576, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.dense(net_map, 576, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.dense(net_map, 576, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            '''
            POS 입력부분 [x1, y1, x2, y2] 로 입력 받는다.
            (-1, 4) 형태는 (-1, 4, 10) 형태로 변환된다.
            이를 (-1, 40)으로 선형시킨다. 
            '''
            
            self._POS = tf.placeholder(tf.int32, [None, 4], name="input_pos")
            net_pos = tf.contrib.layers.one_hot_encoding(self._POS, num_classes=10, on_value=1.0, off_value=0.0)
            net_pos = tf.reshape(net_pos, [-1, 40])
            #self._ONE_HOT = net_pos             
            net_pos = tf.layers.dense(net_pos, 40, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_pos = tf.layers.dense(net_pos, 40, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_pos = tf.layers.dense(net_pos, 40, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            self._ONE_HOT = net_pos 
            '''
            두 선형 결과를 결합한다.
            576 + 40 = 616
            '''
            net = tf.concat([net_map, net_pos], axis=1)
            
            net = tf.layers.dense(net, 100, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, 100, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, 100, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            '''
            결과값 QValue는 -1 ~ 1 사이의 값이다.
            '''
            net = tf.layers.dense(net, self.output_size, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer(),  name="output_pos")
            
            
            self._Qpred = net
            
            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)
            self._predicted = tf.cast(self._Qpred > 0.5 , tf.float32)
            self._accuracy = tf.reduce_mean(tf.cast(tf.equal(self._predicted, self._Y), dtype=tf.float32))
            #optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learn_rate)
            self._train = optimizer.minimize(self._loss)
            
    def accuracy_check(self, state, pos, values):
        feed = {self._MAP: state, self._POS: pos,  self._Y: values}
        return self.session.run([self._Qpred, self._accuracy], feed_dict=feed)
    def predict_test(self, state, pos) :
        feed = {self._MAP: state, self._POS: pos}
        return self.session.run([self._Qpred, self._ONE_HOT], feed_dict=feed)
    
    def predict(self, state, pos) :
        feed = {self._MAP: state, self._POS: pos}
        return self.session.run(self._Qpred, feed_dict=feed)
    
    def update(self, state: np.ndarray, pos: np.ndarray, values: np.ndarray) -> list:
        feed = {self._MAP: state, self._POS: pos, self._Y: values}
        return self.session.run([self._loss, self._train], feed)