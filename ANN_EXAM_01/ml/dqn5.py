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
            self._MAP = tf.placeholder(tf.float32, [None, 10, 9, 4], name="input_x1")
            self._EXPECTED_VALUE = tf.placeholder(tf.float32, [None, 1], name="input_x2")
            net_map = tf.layers.conv2d(self._MAP, filters=32, kernel_size=[4,4], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=32, kernel_size=[4,4], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=32, kernel_size=[4,4], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=32, kernel_size=[4,4], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=32, kernel_size=[4,4], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            #net_map = tf.layers.conv2d(net_map, filters=32, kernel_size=[4,4], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            #net_map = tf.layers.conv2d(net_map, filters=32, kernel_size=[4,4], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            #net_map = tf.layers.conv2d(net_map, filters=32, kernel_size=[4,4], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            #net_map = tf.layers.conv2d(net_map, filters=32, kernel_size=[4,4], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            #net_map = tf.layers.conv2d(net_map, filters=32, kernel_size=[4,4], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            
            
            #net_map = tf.layers.max_pooling2d(net_map, pool_size=[2, 2], strides=1)
            net = tf.reshape(net_map, [-1, 2880])
            net = tf.concat([net, self._EXPECTED_VALUE], axis=1)
            net = tf.layers.dense(net, 720, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, 360, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, 180, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            net = tf.layers.dense(net, self.output_size, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer(),  name="output_y")
            
            
            self._Qpred = net
            
            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)
            self._predicted = tf.cast(self._Qpred > 0.5 , tf.float32)

            #optimizer = tf.train.AdamOptimizer(learning_rate=self.learn_rate)
            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learn_rate)
            self._train = optimizer.minimize(self._loss)
            
    
    def predict(self, state) :
        
        expected_value = np.zeros((len(state), 1))
        for i in range(len(state)):
            for y in range(len(state[i])):
                for x in range(len(state[i][y])):
                    # 이동 가능한 좌표일 때
                    if state[i][y][x][2] > 0:
                        expected_value[i][0] =  state[i][y][x][0] + state[i][y][x][1]  
                    
        
        
        feed = {self._MAP: state, self._EXPECTED_VALUE: expected_value}
        return self.session.run(self._Qpred, feed_dict=feed)
    
    def update(self, state: np.ndarray,  values: np.ndarray) -> list:
        
        expected_value = np.zeros((len(state), 1))
        for i in range(len(state)):
            for y in range(len(state[i])):
                for x in range(len(state[i][y])):
                    # 이동 가능한 좌표일 때
                    if state[i][y][x][2] > 0:
                        expected_value[i][0] =  state[i][y][x][0] + state[i][y][x][1]  
                        values[i][0] += expected_value[i][0] / 29200.0 
                        
        feed = {self._MAP: state, self._EXPECTED_VALUE: expected_value, self._Y: values}
        return self.session.run([self._loss, self._train], feed)