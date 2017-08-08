'''
Created on 2017. 8. 8.

@author: 3F8VJ32

맵정보와 이동할 좌표정보를 가지고 Q-Value를 구하는 인공신경망의 학습부분
pos_ai / dqn2 와 달리 DQN로직을 사용한다.

'''

from collections import deque
import os
import random
from time import sleep
from typing import List
from Game import Game
from JsonParsorClass import JsonParsorClass
from dqn3 import DQN3
import numpy as np
import tensorflow as tf

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
REPLAY_MEMORY = 2000000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
DISCOUNT_RATE = 0.9


def train_dqn(mainDQN: DQN3, targetDQN: DQN3, train_batch):
    states = np.vstack([[x['state']] for x in train_batch])  
    pos = np.vstack([[x['pos']] for x in train_batch])

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
    get_replay_deque_from_db 
    DB에서 저장된 기보 데이터를 불러오는 함수
    turn_rate 가 0.01 보다 작으면 game에서 랜덤 1000개를...
    turn_rate가 0.01 보다 크면, pan에서 무작위 추출을 한다.
    조회속도에서 차이가 있다.
'''
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

'''
    get_copy_var_ops
    targetDQN을 복사할 때 쓰는 함수
'''
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

