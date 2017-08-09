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

def get_min_qvalue(targetDQN: DQN3, env: Game, next_state, turnFlag):
    map, _ = env.getCustomMapByState(next_state)
    poslist = env.getPossibleMoveListfromCustomMap(map, turnFlag)
    minvalue = 1
    temp = []
    for pos in poslist:
        value = targetDQN.predict([next_state], [pos])
        temp.append(value[0][0])
        if value[0][0] < minvalue:
            minvalue = value[0][0]
    return minvalue

def get_max_qvalue(targetDQN: DQN3, env: Game, next_state, turnFlag):
    map, _ = env.getCustomMapByState(next_state)
    poslist = env.getPossibleMoveListfromCustomMap(map, turnFlag)
    maxvalue = -1
    temp = []
    for pos in poslist:

        value = targetDQN.predict([next_state], [pos])
        temp.append(value[0][0])
        if value[0][0]  > maxvalue:
            maxvalue =  value[0][0] 
    return maxvalue

def replay_train(mainDQN: DQN3, targetDQN: DQN3, env, train_batch):
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
    next_states = np.vstack([[x['next_state']] for x in train_batch])
    turnFlags = np.array([x['turnFlag'] for x in train_batch])
    winFlags = np.array([x['win'] for x in train_batch])
    done = np.array([x['done'] for x in train_batch])
    pre_x = np.vstack([x['pre_x'] for x in train_batch])
    pre_y = np.vstack([x['pre_y'] for x in train_batch])
    new_x = np.vstack([x['new_x'] for x in train_batch])
    new_y = np.vstack([x['new_y'] for x in train_batch])
    done = np.array([x['new_y'] for x in train_batch])
    
    # pos array를 만든다.
    pos = np.concatenate([pre_x, pre_y, new_x, new_y], axis=1)
    loss = []
    for i in range(len (winFlags)):
        _loss , _ = train_dqn(mainDQN, targetDQN, env, states[i], pos[i], next_states[i], turnFlags[i], done[i], winFlags[i]);
        loss.append(loss)
        
    return loss
    
    

def train_dqn(mainDQN: DQN3, targetDQN: DQN3, env: Game, state, action, next_state, turnFlag,  done, winFlag):
    reward = 0
    if done:
        if turnFlag == winFlag:
            # 승리시 1점
            reward = 1
        else:
            # 패배시 -1점
            reward = -1     
    
    next_flag = 0
    if turnFlag == 1:
        next_flag = 2
    elif turnFlag == 2:
        next_flag = 1
    '''
        이후 인공신경망의 추가적인 학습 규칙은 여기서 reward값을 변경하면 된다.
        가령 공격적인 성향을 갖게끔 하려면, 50 수 이전에 게임이 끝나면 
        reward값에 가점을 주어 공격적인 성향을 갖게 하거나,
        상대방보다 점수가 높은 상태라면 가점을 주어서 방어적인 성격을 
        갖게 하면 된다.
    '''    
    if done != True:
        QValue = reward - get_min_qvalue(targetDQN, env, next_state, next_flag) - get_max_qvalue(targetDQN, env, next_state, next_flag)

    else:
        QValue = reward
    return mainDQN.update([state], [action], [[QValue]])
    
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


def learning_from_db(learning_episodes = 100000000):
    
    print ("####################")
    print ("load learning data")
    print ("####################")

    replay_buffer = get_replay_deque_from_db()

    print ("####################")
    print ("load learning data done!")
    print ("####################")
    
    with tf.Session() as sess:
        mainDQN = DQN3(sess, name="main_pos_dqn")
        targetDQN = DQN3(sess, name="target_pos_dqn")
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        
        game = Game()
        #saver.restore(sess, CURRENT_PATH + "/pos/model.ckpt")
        
        copy_ops = get_copy_var_ops(dest_scope_name="main_pos_dqn", src_scope_name="target_pos_dqn")
        weight = sess.run(copy_ops)
        print(weight)
        print ("####################")
        print ("learning process start")
        print ("####################")  
        
        for i in range(learning_episodes):
            if len(replay_buffer) > BATCH_SIZE:
                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                
                replay_train(mainDQN, targetDQN, game, minibatch)
                
                if i % TARGET_UPDATE_FREQUENCY == 0:
                    w = sess.run(copy_ops)
                    
                    if i % 10000 == 0 and i != 0:
                        print (w)
                
                if i % 100 == 0 and i != 0:
                    #pred, acc = test_from_db(mainDQN, minibatch)
                    #print("{} steps accuracy: {}, and loss: {}".format(i, acc, loss[0]))
                    print("{} steps")
                    saver.save(sess, CURRENT_PATH + "/pos_dqn/model.ckpt") 
                    
                    
                if i % 10000 == 0 and i != 0:

                    print ("####################")
                    print ("load learning data")
                    print ("####################")
                    replay_buffer.clear()
                    replay_buffer = get_replay_deque_from_db(100000, 0.0)
                    print ("####################")
                    print ("load learning data done!")
                    print ("####################")
        print ("####################")
        print ("learning process complete")
        print ("####################")       
        
learning_from_db()