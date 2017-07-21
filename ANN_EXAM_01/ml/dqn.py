'''
Created on 2017. 7. 19.

@author: 3F8VJ32
'''

import tensorflow as tf
import numpy as np

class DQN:

    def __init__(self, session: tf.Session, input_size: int, output_size: int, name: str="main") -> None:
        """DQN Agent can
        1) Build network
        2) Predict Q_value given state
        3) Train parameters
        Args:
            session (tf.Session): Tensorflow session
            input_size (int): Input dimension
            output_size (int): Number of discrete actions
            name (str, optional): TF Graph will be built under this name scope
        """
        self.session = session
        self.input_size = input_size
        self.output_size = output_size
        self.net_name = name

        self._build_network()

    def _build_network(self, h_size=100, l_rate=0.001) -> None:
        """DQN Network architecture (simple MLP)
        Args:
            h_size (int, optional): Hidden layer dimension
            l_rate (float, optional): Learning rate
        """
        with tf.variable_scope(self.net_name):
            self._X = tf.placeholder(tf.float32, [1, self.input_size], name="input_x")
            net = self._X
            net = tf.layers.dense(net, h_size, activation=tf.nn.relu)
            net = tf.layers.dense(net, h_size, activation=tf.nn.relu)
            net = tf.layers.dense(net, self.output_size)
            #net = tf.contrib.layers.softmax(net)
            
            '''
            W1 = tf.Variable(tf.random_normal([self.input_size, h_size]))
            B1 = tf.Variable(tf.zeros([h_size]))
            L1  = tf.nn.relu(tf.matmul(self._X , W1) + B1)
            
            W2 = tf.Variable(tf.random_normal([h_size, h_size]))
            B2 = tf.Variable(tf.zeros([h_size]))
            L2  = tf.nn.relu(tf.matmul(L1 , W2) + B2)
            
            W3 = tf.Variable(tf.random_normal([h_size, 4 * self.output_size]))
            B3 = tf.Variable(tf.zeros([4 * self.output_size]))
            L3  = tf.nn.softmax(tf.matmul(L2 , W3) + B3)
            
            #net = tf.reshape(L3, [-1, 10])
            '''
            
            
            self._Qpred = net

            self._Y = tf.placeholder(tf.float32, shape=[None, self.output_size])
            self._loss = tf.losses.mean_squared_error(self._Y, self._Qpred)

            optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)
            self._train = optimizer.minimize(self._loss)

    def predict(self, state: np.ndarray) -> np.ndarray:
        """Returns Q(s, a)
        Args:
            state (np.ndarray): State array, shape (n, input_dim)
        Returns:
            np.ndarray: Q value array, shape (n, output_dim)
        """
        x = np.reshape(state, [1, self.input_size])
        
        return self.session.run(self._Qpred, feed_dict={self._X: x})

    def update(self, x_stack: np.ndarray, y_stack: np.ndarray) -> list:
        """Performs updates on given X and y and returns a result
        Args:
            x_stack (np.ndarray): State array, shape (n, input_dim)
            y_stack (np.ndarray): Target Q array, shape (n, output_dim)
        Returns:
            list: First element is loss, second element is a result from train step
        """
        feed = {
            self._X: x_stack,
            self._Y: y_stack
        }
        return self.session.run([self._loss, self._train], feed)