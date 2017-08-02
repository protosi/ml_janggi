import tensorflow as tf
import numpy as np
import os 
import copy
import pickle
from collections import deque
from typing import List
import random
from dqn import DQN
from Game import Game
from time import sleep
# https://omid.al/posts/2017-02-20-Tutorial-Build-Your-First-Tensorflow-Android-App.html

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
REPLAY_MEMORY = 1000000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
DISCOUNT_RATE = 0.99

INPUT_SIZE = 4
OUTPUT_SIZE = 3
MAX_EPISODES = 100000

# 게임이 끝나면 temp_replay에 든 레코드를 갈무리한다.
def end_game_process(mainDQN, temp_replay, winFlag):
    rt_deque = deque()
    states = np.array([x[0] for x in temp_replay])
    next_states = np.array([x[1] for x in temp_replay])
    actions = np.array([x[2] for x in temp_replay])
    done = np.array([x[3] for x in temp_replay])
    
    
    for i in range(len(actions)):
        reward = -1
        if winFlag == np.argmax(actions[i]):
            reward = 1
        
        print("original :" , states[i].shape, actions[i], reward, next_states[i].shape, done[i], winFlag)
        rt_deque.append((states[i], next_states[i], winFlag,  done[i]))
        
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
        if winFlag == np.argmax(inv_action[0]):
            reward = 1      
        print("leftright inv :" , inv_states.shape, inv_action[0], reward, inv_new_states.shape, done[i], winFlag)
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
        if invWinFlag == np.argmax(inv_action[0]):
            reward = 1
            
        print("flag inv :" , inv_states.shape, inv_action[0], reward, inv_new_states.shape, done[i], invWinFlag)
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
        
        if invWinFlag == np.argmax(inv_action[0]):
            reward = 1
            
        print("flag leftright inv :" , inv_rl_states.shape, inv_action[0], reward, inv_rl_new_states.shape, done[i], invWinFlag)
        rt_deque.append((inv_rl_states, inv_rl_new_states, invWinFlag, done[i]))
    
        

    return rt_deque
    
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
        reward = -1
        if(np.argmax(_actions[i]) == winFlags[i]):
            reward = 1
        rewards.append(reward)
        actions.append(np.argmax(_actions[i]))

    Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) * ~done
    

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

def main():
    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    with open(CURRENT_PATH + '/cnn/replay_buffer.pickle', 'rb') as handle:
        replay_buffer = pickle.load(handle)
    temp_replay_buffer = deque(maxlen=REPLAY_MEMORY)
    env = Game()
    with tf.Session() as sess:
        mainDQN = DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        #saver.save(sess, CURRENT_PATH + "/cnn/model.ckpt")
        saver.restore(sess, CURRENT_PATH + "/cnn/model.ckpt")
        
        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        sess.run(copy_ops)
        
        print(copy_ops);
        sleep(10)
        
        for i in range(10000):
            if len(replay_buffer) > BATCH_SIZE:
                minibatch = random.sample(replay_buffer, BATCH_SIZE)
                loss, _ = replay_train(mainDQN, targetDQN, minibatch)
    
                if i % 100 == 0:
                    print(i, loss)
                    
            if i % TARGET_UPDATE_FREQUENCY == 0:
                sess.run(copy_ops)
        
        
        MLFlag = 2
        
        for episode in range(MAX_EPISODES):
            
            if MLFlag == 1:
                MLFlag = 2
            elif MLFlag == 2:
                MLFlag = 1
            
            temp_replay_buffer.clear()
            #e = 1. / ((episode/10)+1)
            e = 1
            done = False
            step_count = 0
            env.initGame()
            
            env.printMap()
            
            
            
            while not done:
                
                state = env.getState()
                
                # ml expect
                action = mainDQN.predict([state])
                print("ml thinks ", action)
                
                if np.random.rand() < e:
                    # minMax move 
                    pos = env.getMinMaxPos()
                    
                _, done = env.doGame(pos)
                env.printMap()
                next_state = env.getState()
                
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

            print(len(replay_buffer))
            with open(CURRENT_PATH+'/cnn/replay_buffer.pickle', 'wb') as handle:
                pickle.dump(replay_buffer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            lMax = (episode +1) * 1000
            if lMax > 50000:
                lMax = 50000
            #l_rate = l_rate * 0.95
            for i in range(lMax):
                
                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)

                    if i % 100 == 0:
                        print(i, loss)
                        
                if i % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops)

            saver.save(sess, CURRENT_PATH + "/cnn/model.ckpt")

        # End of All Episode 
        
        
        
        
        
        
main()