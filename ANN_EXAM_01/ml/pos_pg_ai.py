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
from typing import List

CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))
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

def learn_from_play(episode = 1000000):
    sess = tf.Session()
    ai = GameAI(sess, "main")
    temp = GameAI(sess, "target")
    env = Game()
    saver = tf.train.Saver()
    
    sess.run(tf.global_variables_initializer())
    
    #saver.restore(sess, CURRENT_PATH + "/pos_pg/model.ckpt")
    copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
    print(sess.run(copy_ops))
    MLFlag = 1
    # 에피소드 시작
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
                    poslist = env.getPossibleMoveList(turnFlag)
                    maxpos = None
                    maxvalue = -999 
                    maxstate = None 
                    for pos in poslist:
                        state = env.getState(pos)
                        action = ai.predict([state])
                        print(pos, action)
                        if maxvalue < action:
                            maxpos = pos
                            maxvalue = action
                            maxstate = state
                    pos = env.getMinMaxPos() 
                    maxstate = env.getState(pos)
                    maxvalue = ai.predict([maxstate])
                    reward, done = env.doGame(pos)
                    print("minmax play! ml thinks the pos", maxpos ,"is best pos", maxvalue)
                    env.printMap()
                else:
                    poslist = env.getPossibleMoveList(turnFlag)
                
                    # 현재 ai가 가장 좋다고 생각하는 위치를 구한다.
                    maxpos = None
                    maxvalue = -999 
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
                reward = env.choScore# - env.hanScore * 0.1
                
                rewards = np.vstack([rewards,reward])
                states = np.append(states, [maxstate], axis=0)
                actions = np.vstack([actions, maxvalue])
                if done:
                    
                    dr = ai.normalize_discount_reward(rewards)
                    print(dr)
                    for j in range(1):
                        l, _ = ai.update(states, dr, actions)
                        print("[Step {} of Episode {}] Reward: {} Loss: {}".format(j, i, reward, l))
        print(sess.run(copy_ops))
        saver.save(sess, CURRENT_PATH + "/pos_pg/model.ckpt") 
        print("[Episode {}] Reward: {} Loss: {}".format(i, reward, l))
        #sleep(5)
                
learn_from_play()
