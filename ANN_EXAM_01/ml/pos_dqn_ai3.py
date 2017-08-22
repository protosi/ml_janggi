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
from JsonParsorClass2 import JsonParsorClass
from dqn5 import GameAI
import numpy as np
import tensorflow as tf
import time
from tensorflow.contrib.labeled_tensor import batch
import copy

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
REPLAY_MEMORY = 2000000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
DISCOUNT_RATE = 0.997

def get_min_max_qvalue(targetDQN: GameAI, env: Game, next_state, turnFlag):
    map, _ = env.getCustomMapByState(next_state)
    poslist = env.getPossibleMoveListfromCustomMap(map, turnFlag)
    state_list = []
    min = 1
    max = -1
    for pos in poslist:
        temp_state = copy.deepcopy(next_state)
        
        pre_x = pos[0]
        pre_y = pos[1]
        new_x = pos[2]
        new_y = pos[3]
        
        temp_state[pre_y][pre_x][2] = -1000
        temp_state[new_y][new_x][2] = 1000
        
        state_list.append(temp_state)

    if len(poslist) <= 0:
        print("poslist is zero")
    elif len(poslist) == len(state_list):
        values = targetDQN.predict(state_list)

        min = np.min(values)
        max = np.max(values)
    state_list.clear()
    

    return min, max
    

def replay_train(mainDQN: GameAI, targetDQN: GameAI, env, train_batch):
    # data form
    '''
    {'state': array(...)
    , 'next_state': array(...)
    , 'moveUnit': 'R_JOL'
    , 'pre_x': 8, 'pre_y': 6
    , 'new_x': 7, 'new_y': 6
    , 'turnCount': 48
    , 'turnFlag': 2 #1 cho, 2 han
    , 'win': 2, 'done' : 0}
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

    
    # pos array를 만든다.
    pos = np.concatenate([pre_x, pre_y, new_x, new_y], axis=1)
    #loss = []
    #for i in range(len (winFlags)):
    #    _loss , _ = train_dqn(mainDQN, targetDQN, env, states[i], pos[i], next_states[i], turnFlags[i], done[i], winFlags[i]);
    #    loss.append(loss)
    
    # 한꺼번에 처리하도록 변경 
    loss , _ = train_dqn(mainDQN, targetDQN, env, states, pos, next_states, turnFlags, done, winFlags);
        
    return loss
    

def train_sl(mainDQN: GameAI, env: Game, state, action):

    map, flag = env.getCustomMapByState(state)
    posslist = env.getPossibleMoveListfromCustomMap(map, flag)
    losslist = []
    if len(posslist) == 0:
        env.printCusomMap(map)
        print(flag, len(posslist))
        
    for pos in posslist:
        value = 0
        if pos[0] == action[0] and pos[1] == action[1] and pos[2] == action[2] and pos[3] == action[3]:
            value = 1
            
        loss, _ = mainDQN.update([state], [[value]])
        losslist.append(loss)
    return np.mean(losslist)


def train_dqn(mainDQN: GameAI, targetDQN: GameAI, env: Game, state, action, next_state, turnFlag,  done, winFlag):
    reward = np.zeros(len(winFlag))
    next_flag = np.zeros(len(winFlag))
    QValue = np.zeros((len(winFlag), 1))
    temp =  np.zeros(len(winFlag))
    for i in range(len(winFlag)):
    
        if done[i]:
            if turnFlag[i] == winFlag[i]:
                # 승리시 1점
                reward[i] = 1
            else:
                # 패배시 -1점
                reward[i] = -1     
                
        else:
            
            if turnFlag[i] == 1:
                next_flag[i] = 2
            elif turnFlag[i] == 2:
                next_flag[i] = 1
            '''
            max = 29200.0
            map, _ = env.getCustomMapByState(next_state[i])
            score = env.getStageScore(map, turnFlag[i]) - env.getStageScore(map, next_flag[i])
            temp[i] = -env.doMinMaxForML(map, env.m_depth-1, score, next_flag[i], next_flag[i]) / max
            '''
            
            cur_sum = np.sum(np.sum(state[i], axis=0),axis=0)
            next_sum = np.sum(np.sum(next_state[i], axis=0),axis=0)
            
            cur_cho = cur_sum[0]
            nex_cho = next_sum[0]
            
            cur_han = cur_sum[1]
            nex_han = next_sum[1]
            
            max = 29200.0
            
            
            
            #if turnFlag[i] == 1:
            #    reward[i] =  ((nex_cho - cur_cho)) / max
            #elif turnFlag[i] == 2:
            #    reward[i] =  ((nex_han - cur_han)) / max
            
            if turnFlag[i] == 1:
                temp[i] =  (cur_han - nex_han) / (cur_han + 1) 
                #temp[i] =  ((max - nex_han) - (max - nex_cho)) / max
            elif turnFlag[i] == 2:
                temp[i] =  (cur_cho - nex_cho) / (cur_cho + 1)
                #temp[i] =  ((max - nex_cho) - (max - nex_han)) / max
            
            '''            
            if turnFlag[i] == 1 and next_sum[0] > next_sum[1]:
                reward[i] += 0.2
            elif turnFlag[i] == 1 and next_sum[0] < next_sum[1]:
                reward[i] -= 0.2
            elif turnFlag[i] == 2 and next_sum[0] < next_sum[1]:
                reward[i] += 0.2
            elif turnFlag[i] == 2 and next_sum[0] > next_sum[1]:
                reward[i] -= 0.2
            '''
        
        
        '''
            이후 인공신경망의 추가적인 학습 규칙은 여기서 reward값을 변경하면 된다.
            가령 공격적인 성향을 갖게끔 하려면, 50 수 이전에 게임이 끝나면 
            reward값에 가점을 주어 공격적인 성향을 갖게 하거나,
            상대방보다 점수가 높은 상태라면 가점을 주어서 방어적인 성격을 
            갖게 하면 된다.
        '''    
        
        
    
        if done[i] != True:
            min, max = get_min_max_qvalue(targetDQN, env, next_state[i], next_flag[i])
            #QValue[i][0] = reward[i] - get_min_qvalue(targetDQN, env, next_state[i], next_flag[i]) - get_max_qvalue(targetDQN, env, next_state[i], next_flag[i])
            QValue[i][0] = reward[i] - (max) * DISCOUNT_RATE
            #QValue[i][0] = reward[i] - (min + max) * DISCOUNT_RATE
        
            if (abs(QValue[i][0]) < abs(temp[i])):
                QValue[i][0] = temp[i]
            
            if i == 0:
                print("train:",action[i], "QValue: ", QValue[i][0],"reward:",  reward[i], "max:",max)

        else:
            QValue[i][0] = reward[i]
            
            
            if i == 0:
                print("train for done:", turnFlag[i], action[i], QValue[i][0])
        
    return mainDQN.update(state, QValue)
    
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

def learn_from_play(EPISODES = 10000):
    
    sess = tf.InteractiveSession()
    
    mainDQN = GameAI(sess, name="main1")
    
    
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess, CURRENT_PATH + "/pos_dqn_play3/model.ckpt")
    targetDQN = GameAI(sess, name="target1")
    copy_ops = get_copy_var_ops(dest_scope_name="target1", src_scope_name="main1")
    weight = sess.run(copy_ops)
    print(weight)
    env = Game()
    replay_buffer = deque()
    for i in range(EPISODES):
        env.initGame()
        done = False
        ML = i % 3
        temp_replay = deque()
        
        while not done:
            
            turnFlag = env.getTurn()
            state = env.getState()
            
            if turnFlag == ML:
                maxvalue = -1
                maxpos = []
                poslist = env.getPossibleMoveList(turnFlag)
                for pos in poslist:
                    temp_state = copy.deepcopy(state)
                    
                    pre_x = pos[0]
                    pre_y = pos[1]
                    new_x = pos[2]
                    new_y = pos[3]
                    
                    temp_state[pre_y][pre_x][2] = - 1000
                    temp_state[new_y][new_x][2] = 1000
                                         
                    value = mainDQN.predict([temp_state])
                    print(pos, value)
                    if maxvalue < value[0][0]:
                        maxvalue = value[0][0]
                        maxpos = pos
                print("ml think best move is ", maxpos, "(", maxvalue , ")")
            else:
                maxpos = env.getMinMaxPos()
            _, done = env.doGame(maxpos)
            env.printMap()
            next_state = env.getState()
            
            pre_x = maxpos[0]
            pre_y = maxpos[1]
            new_x = maxpos[2]
            new_y = maxpos[3]
            state[pre_y][pre_x][2] = -1000
            state[new_y][new_x][2] = 1000
            
            temp_replay.append({'state': state, 'next_state': next_state, 'pos': maxpos, 'turn': turnFlag, 'done': done})
    
            if done:
                winFlag = 0
                if env.choScore > env.hanScore:
                    winFlag = 1
                elif env.hanScore > env.choScore:
                    winFlag = 2
                    
                # replay_buffer 에 temp_replay에 있는 데이터들을 입력한다.
                for row in temp_replay:
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
                    replay_buffer.append({'state': row['state'], 'next_state': row['next_state'], 'pre_x': row['pos'][0], 'pre_y': row['pos'][1], 'new_x': row['pos'][2], 
                                          'new_y': row['pos'][3], 'turnCount': -1, 'turnFlag': row['turn'], 'win':winFlag, 'done': row['done'] })
                
                temp_replay.clear()    
                print ("replay buffer size is " , len(replay_buffer))
                if len(replay_buffer) > BATCH_SIZE:
                    print ("learning start " , len(replay_buffer))
                    for t in range(101):
                        minibatch = random.sample(replay_buffer, BATCH_SIZE)
                        start_time = time.time()
                        loss =replay_train(mainDQN, targetDQN, env, minibatch)
                        end_time = time.time()
                        print("{} steps ({} learns per step) {} seconds spent, loss: {}".format(t, BATCH_SIZE, end_time - start_time, loss))
                        end_time = time.time()
                        if t % TARGET_UPDATE_FREQUENCY == 0:
                            w = sess.run(copy_ops)
                    w = sess.run(copy_ops)
                    print(w)
                    saver.save(sess, CURRENT_PATH + "/pos_dqn_play3/model.ckpt")  
                    print("################")
                    print("model saved")
                    print("################")
                    


def test_play(mainDQN: GameAI):
    env = Game()
    done = False
    ML = np.random.randint(2) + 1
    env.initGame()
    env.printMap()
    while not done:
        
        
        turnFlag = env.getTurn()
        state = env.getState()
        
        if turnFlag == ML:
            poslist = env.getPossibleMoveList(turnFlag)
            maxvalue = -1
            maxpos = []
            for pos in poslist:
                temp_state = copy.deepcopy(state)
                temp_state[pos[1]][pos[0]] = -1
                temp_state[pos[3]][pos[2]] = 1
                value = mainDQN.predict([temp_state])
                print(pos, value)
                if maxvalue < value[0][0]:
                    maxvalue = value[0][0]
                    maxpos = pos

            print("ml think best move is ", maxpos, "(", maxvalue , ")")
        else:
            maxpos = env.getMinMaxPos()
        _, done = env.doGame(maxpos)

        env.printMap()

def super_learning_from_db(learning_episodes = 100000000):
    
    print ("####################")
    print ("load learning data")
    print ("####################")

    replay_buffer = get_replay_deque_from_db()
    minibatch = random.sample(replay_buffer, BATCH_SIZE)
    print ("####################")
    print ("load learning data done!")
    print ("####################")
    
    sess = tf.InteractiveSession()
    
    mainDQN = GameAI(sess, name="main1")
    
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    
    game = Game()
    saver.restore(sess, CURRENT_PATH + "/pos_dqn3/model.ckpt")
        
    print ("####################")
    print ("learning process start")
    print ("####################")  
    
    for i in range(learning_episodes):
        if len(replay_buffer) > BATCH_SIZE:
            minibatch = random.sample(replay_buffer, BATCH_SIZE)
            for row in minibatch:
                print (np.array(row['state']).shape)
                map, turn = game.getCustomMapByState(row['state'])
                game.printCusomMap(map)
                action = [0,0,0,0]
                action[0] = row['pre_x']
                action[1] = row['pre_y']
                action[2] = row['new_x']
                action[3] = row['new_y']
                
                loss = train_sl(mainDQN, game, row['state'], action)
                print("{} steps loss: {}".format(i,  loss))
            if i % 100 == 0:
                test_play(mainDQN)
                saver.save(sess, CURRENT_PATH + "/pos_dqn3/model.ckpt")

            # 1000번마다 데이터를 새로 로드함                
            if i % 1000 == 0 and i != 0:
                print ("####################")
                print ("load learning data")
                print ("####################")
            
                replay_buffer = get_replay_deque_from_db()
                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                print ("####################")
                print ("load learning data done!")
                print ("####################")    
                
                    
                    

learn_from_play()
#super_learning_from_db()