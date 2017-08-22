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
from Game2 import Game
from GameAI import GameAI
import numpy as np
import tensorflow as tf
import time
import copy

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
REPLAY_MEMORY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
DISCOUNT_RATE = 0.99


def replay_train(mainDQN: GameAI, targetDQN: GameAI, env, train_batch):
    # data form
    '''
        {
            "state": state_array,
            "reward": reward, 
            "next_state": next_state_array,
            "action": action, 
            "turn_flag": turn_flag, 
            "done": done
        }
    '''
    state = np.array([x['state'] for x in train_batch])
    next_state = np.array([x['next_state'] for x in train_batch])
    turn_flag = np.array([x['turn_flag'] for x in train_batch])
    action = np.array([x['action'] for x in train_batch])
    done = np.array([x['done'] for x in train_batch])
    reward = np.array([x['reward'] for x in train_batch])

    loss = deque()
    for i in range(len(done)):
        y = mainDQN.predict(state[i])
        if done[i]:
            Q_target = reward[i]
        else:
            Q_target = reward[i] + DISCOUNT_RATE * np.max(targetDQN.predict(next_state[i]), axis=0) * ~ done[i] 
        y[action[i]] = Q_target
        loss.append(mainDQN.update(state[i], y))
    return loss
    
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

def learn_from_play(EPISODES = 10000):
    
    sess = tf.InteractiveSession()
    mainDQN = GameAI(sess, name="main")
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess, CURRENT_PATH + "/pos_flexible_action/model.ckpt")
    targetDQN = GameAI(sess, name="target")
    copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
    weight = sess.run(copy_ops)
    
    env = Game()
    replay_buffer = deque( maxlen=REPLAY_MEMORY)
    
    for i in range(EPISODES):
        env.initGame()
        done = False
        ML = i % 2 
        step = 0
        while not done:
            step+=1
            turn_flag = env.getTurn()
            
            # 인공지능이 아닌 경우에
            if turn_flag != ML:
                poslist = env.getPossibleMoveList(turn_flag)
                pos = env.getMinMaxPos()
                action = -1
                statelist = deque()
                for i in range(len(poslist)):
                    state = env.getState(poslist[i])
                    statelist.append(state)
                    if pos[0] == poslist[i][0] and pos[1] == poslist[i][1] and pos[2] == poslist[i][2] and pos[3] == poslist[i][3]:
                        action = i
                state_array =  np.array([x for x in statelist]) 
                statelist.clear()       
            # 인공지능인 경우에
            else:
                # 이동 가능한 리스트를 가져온다.
                poslist = env.getPossibleMoveList(turn_flag)
                # state를 저장할 deque를 만든다.
                statelist = deque()
                for pos in poslist:
                    state = env.getState(pos)
                    statelist.append(state)
                # deque에 저장한 것을 np.array로 변환한다.
                state_array =  np.array([x for x in statelist])
                result = mainDQN.predict(state_array)
                statelist.clear()
                
                # action을 구한다.
                action = np.argmax(result)
                pos = poslist[action]
                print("ML thinks pos", pos, "is best!" , np.max(result), action)
                print(result)
                
            pre_score = env.choScore# - env.hanScore
            reward, done = env.doGame(pos)
            env.printMap()
            # 게임이 안끝났으면 상대방 말을 처리한다.
            if not done:
                pos = env.getMinMaxPos()
                reward, done = env.doGame(pos)
                env.printMap()
                
            # next_states 를 처리한다.    
            nextposlist = env.getPossibleMoveList(turn_flag)
            nextstates = deque()
            for nextpos in nextposlist:    
                next_state = env.getState(nextpos)
                nextstates.append(next_state)
            next_state_array = np.array([x for x in nextstates])
            nextstates.clear()
            
            next_score = env.choScore# - env.hanScore
            
            reward = next_score - pre_score
            print("reward:", reward, "pre_score:", pre_score, "next_score:", next_score)
            replay_buffer.append({"state": state_array,"reward": reward, 
                                  "next_state": next_state_array,
                                  "action": action, "turn_flag": turn_flag, 
                                  "done": done})
            
            if len(replay_buffer) > BATCH_SIZE:
                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                loss = replay_train(mainDQN, targetDQN, env, minibatch)
            if step % TARGET_UPDATE_FREQUENCY == 0:
                w = sess.run(copy_ops)
        saver.save(sess, CURRENT_PATH + "/pos_flexible_action/model.ckpt")
learn_from_play()