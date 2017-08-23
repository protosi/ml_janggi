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
from math import sqrt

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
REPLAY_MEMORY = 50000
BATCH_SIZE = 512
TARGET_UPDATE_FREQUENCY = 5
DISCOUNT_RATE = 0.9
MINIMAX_DEPTH = 1

def minimaxseq(env:Game, states, flag, depth):
    
    score = []
    
    for state in states:
        map, turn, pos = env.getCustomMapByState(state)
        #print(pos)
        score.append(minimax(env, map, pos, flag, depth))
    return score

def minimax(env: Game ,map, action, flag, depth):
    score = 0
    next_map = env.getCustomMoveMap(map, action[0], action[1], action[2], action[3])
    if depth == 0:
        return env.getCustomMapScore(next_map, flag)
    emflag = 0
    if flag == 1: 
        emflag = 2
    elif flag == 2:
        emflag = 1 
    maxscore = -99999
    maxaction = None
    em_actions = env.getPossibleMoveListfromCustomMap(next_map, emflag)
    for em_action in em_actions:
        em_score = minimax(env, next_map, em_action, emflag, depth-1)
        
        if maxscore < em_score:
            maxscore = em_score
            maxaction = em_action
    score = (-1) * maxscore
    return score 

    
def replay_train_with_minimax(mainDQN: GameAI, targetDQN: GameAI, env:Game, train_batch):
    states = np.array([x['state'] for x in train_batch])
    
    for state in states:
        scores = minimaxseq(env, state, 1, MINIMAX_DEPTH)
        print(np.max(scores), np.min(scores))
        mainDQN.update(state, scores)
    

  


def replay_train(mainDQN: GameAI, targetDQN: GameAI, env:Game, train_batch):
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
        ML = (1+i) % 2 
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
            if done:
                reward = (env.choScore) / sqrt(env.turnCount + 1) - sqrt(env.turnCount) - env.hanScore/1000.0
            else:
                reward =(next_score - pre_score) / sqrt(env.turnCount + 1)
            #reward = (next_score - pre_score)/1000 + 0.5 * (env.choScore - env.hanScore)/9200.0
            print("reward:", reward, "pre_score:", pre_score, "next_score:", next_score)
            replay_buffer.append({"state": state_array,"reward": reward, 
                                  "next_state": next_state_array,
                                  "action": action, "turn_flag": turn_flag, 
                                  "done": done})
            
        if len(replay_buffer) > BATCH_SIZE:
            minibatch = random.sample(replay_buffer, BATCH_SIZE)
            loss = replay_train_with_minimax(mainDQN, targetDQN, env, minibatch)
        if EPISODES % TARGET_UPDATE_FREQUENCY == 0:
            w = sess.run(copy_ops)
        saver.save(sess, CURRENT_PATH + "/pos_flexible_action/model.ckpt")
learn_from_play()