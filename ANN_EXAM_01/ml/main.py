from collections import deque
import copy
import os 
import random
from time import sleep
from typing import List
import pickle
from Game import Game
from JsonParsorClass import JsonParsorClass
from dqn import DQN
import numpy as np
import tensorflow as tf



#import pickle
# https://omid.al/posts/2017-02-20-Tutorial-Build-Your-First-Tensorflow-Android-App.html
CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
REPLAY_MEMORY = 2000000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
DISCOUNT_RATE = 0.99

INPUT_SIZE = 4
OUTPUT_SIZE = 2
MAX_EPISODES = 100000000

# 게임이 끝나면 temp_replay에 든 레코드를 갈무리한다.
def end_game_process(mainDQN, temp_replay, winFlag):
    rt_deque = deque()
    states = np.array([x[0] for x in temp_replay])
    next_states = np.array([x[1] for x in temp_replay])
    actions = np.array([x[2] for x in temp_replay])
    done = np.array([x[3] for x in temp_replay])
    
    
    for i in range(len(actions)):
        reward = -1
        if winFlag-1 == np.argmax(actions[i]):
            reward = 1
        
        print("original :" , states[i].shape, actions[i], reward, next_states[i].shape, done[i], winFlag-1)
        rt_deque.append((states[i], next_states[i], winFlag,  done[i]))
        '''
        # ################################
        # 오리지널의 좌우대칭을 처리한다.
        # ################################
        # y축 j
        inv_states = np.zeros((10, 9 , 2))
        inv_new_states = np.zeros((10, 9 , 2))
        for j in range(len(inv_states)):
            # x축 k
            for k in range(len(inv_states[j])):
                inv_states[j][k][0] = states[i][j][8-k][0]
                inv_states[j][k][1] = states[i][j][8-k][1]
                
                inv_new_states[j][k][0] = states[i][j][8-k][0]
                inv_new_states[j][k][1] = states[i][j][8-k][1]
        inv_action = mainDQN.predict([inv_states]) 
        reward = -1 
        if (winFlag-1) == np.argmax(inv_action[0]):
            reward = 1      
        print("leftright inv :" , inv_states.shape, inv_action[0], reward, inv_new_states.shape, done[i], (winFlag-1))
        rt_deque.append((inv_states, inv_new_states,  winFlag, done[i]))
        
        
        # ################################
        # 한초를 바꾸어 처리한다.
        # ################################
        
        inv_states = np.zeros((10, 9 , 2))
        inv_new_states = np.zeros((10, 9 , 2))
        
        for j in range(len(inv_states)):
            # x축 k
            for k in range(len(inv_states[j])):
                inv_states[j][k][1] = states[i][9-j][k][0]
                inv_states[j][k][0] = states[i][9-j][k][1]
                
                inv_new_states[j][k][1] = next_states[i][9-j][k][0]
                inv_new_states[j][k][0] = next_states[i][9-j][k][1]
        inv_action = mainDQN.predict([inv_states])
        reward = -1
        invWinFlag = winFlag;
        if winFlag == 1:
            invWinFlag = 2
        elif winFlag == 2:
            invWinFlag = 1
        if invWinFlag-1 == np.argmax(inv_action[0]):
            reward = 1
            
        print("flag inv :" , inv_states.shape, inv_action[0], reward, inv_new_states.shape, done[i], invWinFlag-1)
        rt_deque.append((inv_states, inv_new_states, invWinFlag,  done[i]))
    
        # ################################
        # 한초대칭의 좌우대칭을 처리한다.
        # ################################
        inv_rl_states = np.zeros((10, 9 , 2))
        inv_rl_new_states = np.zeros((10, 9 , 2))
        for j in range(len(inv_states)):
            # x축 k
            for k in range(len(inv_states[j])):
                inv_rl_states[j][k][0] = inv_states[j][8-k][0]
                inv_rl_states[j][k][1] = inv_states[j][8-k][1]
                
                inv_rl_new_states[j][k][0] = inv_new_states[j][8-k][0]
                inv_rl_new_states[j][k][1] = inv_new_states[j][8-k][1]
                
        inv_action = mainDQN.predict([inv_rl_states])
        reward = -1
        
        if invWinFlag-1 == np.argmax(inv_action[0]):
            reward = 1
            
        print("flag leftright inv :" , inv_rl_states.shape, inv_action[0], reward, inv_rl_new_states.shape, done[i], invWinFlag-1)
        rt_deque.append((inv_rl_states, inv_rl_new_states, invWinFlag, done[i]))
        '''
        

    return rt_deque
    
def train_dqn_from_db(mainDQN: DQN, targetDQN: DQN, train_batch) :
    states = np.vstack([[x['state']] for x in train_batch])    
    next_states = np.vstack([[x['next_state']] for x in train_batch])
    winFlags = np.array([x['win'] for x in train_batch])
    #turnFlags = np.array([x['turnFlag'] for x in train_batch])
    X = states
    _actions = mainDQN.predict(states)
    rewards = []
    actions = []
    for i in range (len(winFlags)):
        reward = -2
        if(np.argmax(_actions[i]) == (winFlags[i] - 1)):
            reward = 1
        rewards.append(reward)
        actions.append(np.argmax(_actions[i]))
    Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) 
    #Q_target = rewards
    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target
    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y)
    
# replay에서 미니배치 추출한 것으로 학습을 시킨다.
def replay_train(mainDQN: DQN, targetDQN: DQN, train_batch) :
    states = np.vstack([[x[0]] for x in train_batch])
    next_states = np.vstack([[x[1]] for x in train_batch])
    winFlags = np.array([x[2] for x in train_batch])
    done = np.array([x[3] for x in train_batch])
    X = states
    _actions = mainDQN.predict(states)
    rewards = []
    actions = []
    for i in range (len(winFlags)):
        reward = -2
        if(np.argmax(_actions[i]) == (winFlags[i] - 1)):
            reward = 1
        rewards.append(reward)
        actions.append(np.argmax(_actions[i]))
    Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) * ~done
    #Q_target = rewards
    y = mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target
    # Train our network using target and predicted Q values on each episode
    return mainDQN.update(X, y)


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

def learn_temp(learning_episodes, sess, mainDQN, targetDQN):
    
    print ("####################")
    print ("load learning data")
    print ("####################")
    #replay_buffer = get_replay_deque_from_db()
    #with open(CURRENT_PATH+'/cnn/replay_buffer.pickle', 'wb') as handle:
    #    pickle.dump(replay_buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(CURRENT_PATH + '/cnn/replay_buffer.pickle', 'rb') as handle:
        replay_buffer = pickle.load(handle)
    print ("####################")
    print ("load learning data done!")
    print ("####################")
    saver = tf.train.Saver()
    copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")

    

    
    
        
    print ("####################")
    print ("learning process start")
    print ("####################")  
    
    
    for i in range(learning_episodes):
        if len(replay_buffer) > BATCH_SIZE:
            minibatch = random.sample(replay_buffer, BATCH_SIZE)
            loss, _ = replay_train(mainDQN, targetDQN, minibatch)

            if i % 1000 == 0:
                print(i, loss)
                saver.save(sess, CURRENT_PATH + "/cnn/model.ckpt") 
                
        if i % TARGET_UPDATE_FREQUENCY == 0:
            sess.run(copy_ops)
        
    print ("####################")
    print ("learning process complete")
    print ("####################")  

def learn_from_db(learning_episodes = 100000000):
    
    print ("####################")
    print ("load learning data")
    print ("####################")
    replay_buffer = get_replay_deque_from_db()
    #with open(CURRENT_PATH+'/cnn/new_replay_buffer.pickle', 'wb') as handle:
    #    pickle.dump(replay_buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #with open(CURRENT_PATH + '/cnn/replay_buffer.pickle', 'rb') as handle:
    #    replay_buffer = pickle.load(handle)
    print ("####################")
    print ("load learning data done!")
    print ("####################")
    
    with tf.Session() as sess:
        mainDQN = DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        
        
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        #saver.restore(sess, CURRENT_PATH + "/cnn/model.ckpt")
        
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)
        
        print ("####################")
        print ("learning process start")
        print ("####################")  
    
    
        for i in range(learning_episodes):
            if len(replay_buffer) > BATCH_SIZE:
                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                loss, _ = train_dqn_from_db(mainDQN, targetDQN, minibatch)

                if i % 1000 == 0 and i != 0:
                    print(i, loss)
                    saver.save(sess, CURRENT_PATH + "/cnn/model.ckpt") 
                if i % 10000 == 0 and i != 0:
                    print ("####################")
                    print ("load learning data")
                    print ("####################")
                    replay_buffer.clear()
                    replay_buffer = get_replay_deque_from_db()
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

def get_replay_deque_from_db():
    jParsor = JsonParsorClass()
    gamelist = jParsor.getGameList();
    
    rt_deque = deque(maxlen=REPLAY_MEMORY);
    
    
    for x in gamelist:

        panlist = jParsor.getPanList(x['idx']);
        for i in range(len(panlist)):
            row = panlist[i]
            rt_deque.append(row)
        
    return rt_deque

def main():
    replay_buffer = deque(maxlen=REPLAY_MEMORY)

    temp_replay_buffer = deque(maxlen=REPLAY_MEMORY)
    env = Game()
    with tf.Session() as sess:
        
        # 신경망 생성
        mainDQN = DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        
        # 모든 변수를 초기화한다
        sess.run(tf.global_variables_initializer())
        
        
        # 학습된 신경망 로드
        saver = tf.train.Saver()
        saver.restore(sess, CURRENT_PATH + "/cnn/model.ckpt")
        
        # 불러온 신경망을 target network 에 복사
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        w = sess.run(copy_ops)
        
        
        
        # 불러온 가중치(weight)를 확인
        print(w);
        sleep(10)        
        
        #learn_temp(10000, mainDQN=mainDQN, targetDQN=targetDQN, sess=sess)  
        
        # ML을 수행할 플래그 (1 초, 2 한)
        MLFlag = 1
        for episode in range(MAX_EPISODES):
            
            #if MLFlag == 1:
            #    MLFlag = 2
            #elif MLFlag == 2:
            #    MLFlag = 1
            
            temp_replay_buffer.clear()
            #e = 1. / ((episode/10)+1)
            e = 0
            done = False
            step_count = 0
            env.initGame()
            
            env.printMap()
            
            
            
            while not done:
                
                state = env.getState()
                action = mainDQN.predict([state])
                turnFlag = env.getTurn()
                
                # random 값이 e보다 낮으면 MINMAX 수행
                if MLFlag == turnFlag and np.random.rand() > e:
                    maxvalue = 0
                    maxpos = []
                    map = env.getMap()
                    maxresult = []
                    poslist = env.getPossibleMoveList(turnFlag)
                    
                    for pos in poslist:
                        
                        # 현재 맵에서 해당 좌표로 이동한 것을 가정한 맵을 구한다.
                        moved_map = env.getCustomMoveMap(map, pos[0], pos[1], pos[2], pos[3])
                        # 해당 맵으로 인공 신경망에 입력할 state를 만든다.
                        moved_state = env.getCustomState(moved_map)
                        # 신경망에 넣어 결과 값을 얻는다. [[a, b]]
                        result = mainDQN.predict([moved_state])
                        
                        # max값을 갖는 pos를 찾는다.
                        if result[0][turnFlag-1] > maxvalue:
                            # 0은 초(1), 1은 한(2) 이므로
                            maxvalue = result[0][turnFlag-1]
                            maxpos = pos
                            maxresult = result
                    
                    print ("ml thinks pos", maxpos , "is best!", maxvalue, maxresult)
                    pos = maxpos 

                
                else:
                    pos = env.getMinMaxPos()
                    
                _, done = env.doGame(pos)
                next_state = env.getState()
                env.printMap()                
                step_count += 1
                temp_replay_buffer.append((state, next_state, action[0], done))
                
            # End of Single Episode
            winFlag = 0
            if(env.choScore > env.hanScore):
                winFlag = 1
            elif(env.choScore < env.hanScore):
                winFlag = 2

            rt_deque = end_game_process(mainDQN, temp_replay_buffer, winFlag)
            replay_buffer.extend(rt_deque)

            #with open(CURRENT_PATH+'/cnn/replay_buffer.pickle', 'wb') as handle:
            #    pickle.dump(replay_buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            for i in range(10000):
                
                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)

                    if i % 1000 == 0:
                        print(i, loss)
                        
                if i % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops)
            learn_temp(10000, mainDQN=mainDQN, targetDQN=targetDQN, sess=sess)  
            #saver.save(sess, CURRENT_PATH + "/cnn/model.ckpt")
            
        # End of All Episode 
        
        
        
        
        
learn_from_db()        
#main()