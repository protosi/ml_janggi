'''
Created on 2017. 8. 18.

@author: 3F8VJ32
'''
import tensorflow as tf
from Game2 import Game
from pg import GameAI
from _operator import pos
import numpy as np
import os
from time import sleep

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

def learn_from_play(episode = 10000):
    sess = tf.Session()
    ai = GameAI(sess, "main")
    env = Game()
    saver = tf.train.Saver()
    saver.restore(sess, CURRENT_PATH + "/pos_pg/model.ckpt")
    sess.run(tf.global_variables_initializer())
    MLFlag = 1
    # 에피소드 시작
    EPISODE_100_REWARD_LIST = []
    for i in range(episode):
        
        env.initGame()
        done = False
        states = np.empty(shape=[0, 10, 9, 4])
        rewards = np.empty(shape=[0, 1])
        actions = np.empty(shape=[0, 1])
        env.printMap()
        p = np.random.rand()
        while not done:
            # 현재 차례 플래그
            turnFlag = env.getTurn()
            
            if MLFlag == turnFlag:
                if p < 0.5:
                    pos = env.getMinMaxPos() 
                    maxstate = env.getState(pos)
                    maxvalue = ai.predict([maxstate])
                    reward, done = env.doGame(pos)
                    print("minmax play! ml thinks the pos", pos ,"is best pos", maxvalue)
                    env.printMap()
                else:
                    poslist = env.getPossibleMoveList(turnFlag)
                
                    # 현재 ai가 가장 좋다고 생각하는 위치를 구한다.
                    maxpos = None
                    maxvalue = 0 
                    maxstate = None   
                    for pos in poslist:
                        state = env.getState(pos)
                        action = ai.predict([state])
                        print(pos, action)
                        if maxvalue < action:
                            maxpos = pos
                            maxvalue = action
                            maxstate = state
                    _, done = env.doGame(maxpos)    
                    print("ml thinks the pos", maxpos ,"is best pos", maxvalue)    
                    env.printMap()
                if not done:
                    action = env.getMinMaxPos() 
                    _, done = env.doGame(action)
                    #reward = reward * (-1)
                    env.printMap()
                reward = env.choScore - env.hanScore * 0.1
                
                rewards = np.vstack([rewards,reward])
                states = np.append(states, [maxstate], axis=0)
                actions = np.vstack([actions, maxvalue])
                if done:
                    dr = ai.normalize_discount_reward(rewards)
                    l, _ = ai.update(states, dr, actions)
        saver.save(sess, CURRENT_PATH + "/pos_pg/model.ckpt") 
        print("[Episode {}] Reward: {} Loss: {}".format(i, reward, l))
        #sleep(5)
                
learn_from_play()
