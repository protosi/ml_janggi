'''
Created on 2017. 8. 8.

@author: 3F8VJ32

맵정보를 가지고 좌표정보 [x1, y1, x2, y2]를 뽑는 인공신경망의 학습 로직부분

'''
from collections import deque
import os
import random
from time import sleep
from typing import List
from Game import Game
from JsonParsorClass import JsonParsorClass
from dqn2 import DQN2
import numpy as np
import tensorflow as tf

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
REPLAY_MEMORY = 2000000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
DISCOUNT_RATE = 0.99
# output은 [x1, y1, x2, y2] 일게 나온다.
OUTPUT_SIZE = 4

'''
    convertToOneHot
    one_hot_encoding 을 수행하는 함수
    0, 1, 2를 3으로 one_hot_encoding 하면
    [1, 0, 0], [0, 1, 0], [0, 0, 1] 이 된다.
'''
def convertToOneHot(vector, num_classes=None):
    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0
    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)
    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

'''
    test_from_db
    db에 있는 자료로 테스트를 하는 함수
'''
def test_from_db(mainDQN: DQN2, train_batch) :
    # data form
    '''
    {'state': array(...)
    , 'next_state': array(...)
    , 'moveUnit': 'R_JOL'
    , 'pre_x': 8, 'pre_y': 6
    , 'new_x': 7, 'new_y': 6
    , 'turnCount': 48
    , 'turnFlag': 2 #1 cho, 2 han
    , 'win': 2}
    '''
    states = np.vstack([[x['state']] for x in train_batch])       
        
    pre_x = np.vstack([x['pre_x'] for x in train_batch])
    pre_y = np.vstack([x['pre_y'] for x in train_batch])
    new_x = np.vstack([x['new_x'] for x in train_batch])
    new_y = np.vstack([x['new_y'] for x in train_batch])
    pos = np.concatenate([pre_x, pre_y, new_x, new_y], axis=1)
        
    pred, acc = mainDQN.accuracy_check(states, pos)
    return pred, acc

def train_dqn_from_db(mainDQN: DQN2, targetDQN: DQN2, train_batch):
    states = np.vstack([[x['state']] for x in train_batch])       
    
    pre_x = np.vstack([x['pre_x'] for x in train_batch])
    pre_y = np.vstack([x['pre_y'] for x in train_batch])
    new_x = np.vstack([x['new_x'] for x in train_batch])
    new_y = np.vstack([x['new_y'] for x in train_batch])
    pos = np.concatenate([pre_x, pre_y, new_x, new_y], axis=1)
    
    return pos, mainDQN.update(states, pos)
    
def get_replay_deque_from_db(list_size= 10000, turn_rate=0.0):
    jParsor = JsonParsorClass()
    rt_deque = deque(maxlen=REPLAY_MEMORY);
    if turn_rate > 0.01:
        panlist = jParsor.getRandomPanList(list_size, turn_rate)
    else:
        gamelist = jParsor.getGameList();
        rt_deque = deque(maxlen=REPLAY_MEMORY);
        for x in gamelist:
            panlist = jParsor.getPanList(x['idx']);
            for i in range(len(panlist)):
                row = panlist[i]
                rt_deque.append(row)    
    for x in panlist:
        rt_deque.append(x)
    return rt_deque    

def get_copy_var_ops(*, dest_scope_name: str, src_scope_name: str) -> List[tf.Operation]:

    # Copy variables src_scope to dest_scope
    op_holder = []
    src_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))
    return op_holder


def learn_from_db(learning_episodes = 100000000):
    
    print ("####################")
    print ("load learning data")
    print ("####################")

    replay_buffer = get_replay_deque_from_db()

    print ("####################")
    print ("load learning data done!")
    print ("####################")
    
    with tf.Session() as sess:
        mainDQN = DQN2(sess, OUTPUT_SIZE, name="main_pos")
        targetDQN = DQN2(sess, OUTPUT_SIZE, name="target_pos")
        
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess, CURRENT_PATH + "/pos/model.ckpt")
        
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        weight = sess.run(copy_ops)
        print(weight)
        
        print ("####################")
        print ("learning process start")
        print ("####################")  
    
    
        for i in range(learning_episodes):
            if len(replay_buffer) > BATCH_SIZE:
                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                if i % 1000 == 0 and i != 0:
                    pred, acc = test_from_db(mainDQN, minibatch)
                    print("{} steps accuracy: {}".format(i, acc))
                pos, loss = train_dqn_from_db(mainDQN, targetDQN, minibatch)

                if i % 1000 == 0 and i != 0:
                    pred, acc = test_from_db(mainDQN, minibatch)
                    print("{} steps accuracy: {}, and loss: {}".format(i, acc, loss[0]))
                    saver.save(sess, CURRENT_PATH + "/pos/model.ckpt") 
                if i % 10000 == 0 and i != 0:
                    print (pred)
                    print(pos)
                    print ("####################")
                    print ("load learning data")
                    print ("####################")
                    replay_buffer.clear()
                    replay_buffer = get_replay_deque_from_db(100000, 0.0)
                    print ("####################")
                    print ("load learning data done!")
                    print ("####################")
                    
            if i % TARGET_UPDATE_FREQUENCY == 0:
                w = sess.run(copy_ops)
                if i % 10000 == 0 and i != 0:
                    print (w)
        
        print ("####################")
        print ("learning process complete")
        print ("####################")       
        
        
learn_from_db()