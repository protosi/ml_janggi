import tensorflow as tf
import numpy as np
import os 
import copy
from collections import deque
from typing import List
import random
from dqn import DQN
from Game import Game
from time import sleep
# https://omid.al/posts/2017-02-20-Tutorial-Build-Your-First-Tensorflow-Android-App.html

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
REPLAY_MEMORY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
DISCOUNT_RATE = 1

INPUT_SIZE = 4
OUTPUT_SIZE = 1
MAX_EPISODES = 10000

# 게임이 끝나면 temp_replay에 든 레코드를 갈무리한다.
def end_game_process(mainDQN, temp_replay, maxturn, winFlag):
    rt_deque = deque()
    choMap = np.array([x[0] for x in temp_replay])
    hanMap = np.array([x[1] for x in temp_replay])
    pos = np.array([x[2] for x in temp_replay])
    rewards = np.array([[x[3]] for x in temp_replay])
    curTurn = np.array([x[4] for x in temp_replay])
    curFlag = np.array([x[5] for x in temp_replay])
    
    tempRewards = copy.deepcopy(rewards)
    
    for i in range(0, len(curFlag)):
        # 승자와 같은 편이면~ 포인트를 가산한다.
        if(curFlag[i] == winFlag):
            tempRewards[i][0] = rewards[i][0] + 5000 * (float(curTurn[i]) / float(maxturn))
        elif (curFlag[i] != winFlag):
            tempRewards[i][0] = rewards[i][0] - 5000 * (float(curTurn[i]) / float(maxturn))
    
    Q_target = np.tanh(tempRewards / 5000.0)
    for i in range(len(Q_target)):
        if curFlag[i] == 1:
            curFlagName = "초"
        elif curFlag[i] == 2:
            curFlagName = "한"
        if winFlag == 1:
            winFlagName = "초"
        elif winFlag == 2:
            winFlagName = "한"

        predict = mainDQN.predict([choMap[i]], [hanMap[i]], [pos[i]], [[curFlag[i]]])
        print (curFlagName, curTurn[i], pos[i], rewards[i], tempRewards[i], Q_target[i], predict[0],winFlagName )

        rt_deque.append((choMap[i], hanMap[i], pos[i], [curFlag[i]], Q_target[i]))
    sleep(10)
    return rt_deque
    
# replay에서 미니배치 추출한 것으로 학습을 시킨다.
def replay_train(mainDQN, train_batch):
    choMap = np.array([x[0] for x in train_batch])
    hanMap = np.array([x[1] for x in train_batch])
    pos = np.array([x[2] for x in train_batch])
    flag = np.array([x[3] for x in train_batch])
    Q_target = np.array([x[4] for x in train_batch])
    
    
    
    
    return mainDQN.update(choMap, hanMap, pos, flag, Q_target)

def main():
    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    temp_replay_buffer = deque(maxlen=REPLAY_MEMORY)
    env = Game()
    with tf.Session() as sess:
        mainDQN = DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        #saver.save(sess, CURRENT_PATH + "/cnn/model.ckpt")
        #saver.restore(sess, CURRENT_PATH + "/cnn/model.ckpt")
        
        MLFlag = 2
        
        for episode in range(MAX_EPISODES):
            
            if MLFlag == 1:
                MLFlag = 2
            elif MLFlag == 2:
                MLFlag = 1
            
            temp_replay_buffer.clear()
            #e = 1. / ((episode/10)+1)
            e = 0
            done = False
            step_count = 0
            env.initGame()
            
            env.printMap()
            
            while not done:
                
                choMap = env.getUnitMap(1)
                hanMap = env.getUnitMap(2)
                reward = 0
                curTurn =env.turnCount
                curFlag = env.getTurn()
                
                if np.random.rand() < e:
                    pos = env.getMinMaxPos()
                    predict = mainDQN.predict([choMap], [hanMap], [pos], [[curFlag]])
                    print("ml thinks that this position value is ", predict)
                else:
                    if curFlag == MLFlag:
                        maxPredict = -1
                        maxMv = []
                        poslist = env.getPossibleMoveList(curFlag)
                        predictlist = [];
                        for i in range(len(poslist)):
                            pos = poslist[i]
                            predict = mainDQN.predict([choMap], [hanMap], [pos], [[curFlag]])
                            predictlist.append(predict[0][0])
                            if predict[0][0] > maxPredict:
                                maxPredict = predict[0][0]
                                maxMv = pos
                        print("ml thinks that position", maxMv, " value is ", maxPredict, np.min(predictlist), np.mean(predictlist), np.max(predictlist))
                        pos = maxMv
                    else:
                        pos = env.getMinMaxPos()
                        predict = mainDQN.predict([choMap], [hanMap], [pos], [[curFlag]])
                        print("ml thinks that this position value is ", predict)
                    #predict = mainDQN.predict(state)
                    #action = np.argmax(predict)
                   
                   
                reward, done = env.doGame(pos)
                print (pos, reward, done)
                env.printMap()
                temp_replay_buffer.append((choMap, hanMap, pos, reward, curTurn, curFlag))
                if done:
                    winFlag = 0
                    if(env.choScore > env.hanScore):
                        winFlag = 1
                    else:
                        winFlag = 2
                    
                    rt = end_game_process(mainDQN, temp_replay_buffer, curTurn, winFlag)
                    replay_buffer.extend(rt)
                
                
                
                step_count += 1
                
            # End of Single Episode
            
            
            lMax = (episode +1) * 1000
            if lMax > 50000:
                lMax = 50000
            
            for i in range(lMax):
                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss, _ = replay_train(mainDQN, minibatch)
                    

            saver.save(sess, CURRENT_PATH + "/cnn/model.ckpt")

        # End of All Episode 
        
        
        
        
        
        
main()