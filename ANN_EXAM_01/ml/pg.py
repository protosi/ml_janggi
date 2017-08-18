'''
Created on 2017. 8. 8.

@author: 3F8VJ32

위치 좌표 인공지능 새로 작성
맵정보와 액션을 받으면, 그에 대한 Qvalue가 나오는 신경망

state, action => value 
'''
import tensorflow as tf
import numpy as np

class GameAI:
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
            self._MAP = tf.placeholder(tf.float32, [None, 10, 9, 4], name="input_x")
            net_map = tf.layers.conv2d(self._MAP, filters=32, kernel_size=[4,4], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=32, kernel_size=[4,4], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=32, kernel_size=[4,4], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=32, kernel_size=[4,4], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net_map = tf.layers.conv2d(net_map, filters=32, kernel_size=[4,4], padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())
            

            net = tf.reshape(net_map, [-1, 2880])

            net = tf.layers.dense(net, 720, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, 360, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            net = tf.layers.dense(net, 180, activation=tf.nn.tanh, kernel_initializer=tf.contrib.layers.xavier_initializer())
            
            net = tf.layers.dense(net, self.output_size, activation=tf.nn.sigmoid, kernel_initializer=tf.contrib.layers.xavier_initializer(),  name="output_y")
            
            
            self._Qpred = net
            
            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            
            self._ADV = tf.placeholder(tf.float32, name="reward_signal")
            log_lik = -self._Y * tf.log(self._Qpred + 1e-7) - (1 - self._Y) * tf.log(1 - self._Qpred + 1e-7)
            self._loss = tf.reduce_sum(log_lik * self._ADV)


            optimizer = tf.train.AdadeltaOptimizer(learning_rate=self.learn_rate)
            self._train = optimizer.minimize(self._loss)
            
    def normalize_discount_reward(self,  r, gamma = 0.99):
        dr = self.discount_reward(r, gamma)
        dr = (dr - dr.mean()) / (dr.std() + 1e-7)
        return dr
    
    def discount_reward(self, r, gamma = 0.99):
        discounted_r = np.zeros_like(r, dtype=np.float32)
        running_add = 0
        for t in reversed(range(len(r))):
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r
    
    def predict(self, state) :
     
        
        feed = {self._MAP: state, }
        return self.session.run(self._Qpred, feed_dict=feed)
    
    def update(self, state: np.ndarray, adv: np.ndarray,  values: np.ndarray) -> list:

                        
        feed = {self._MAP: state,  self._ADV: adv, self._Y: values}
        return self.session.run([self._loss, self._train], feed)