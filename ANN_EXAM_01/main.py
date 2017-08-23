from collections import deque
import copy
import os 
import random
from time import sleep
from typing import List
import pickle
from Game2 import Game
from JsonParsorClass import JsonParsorClass
from dqn import DQN
import numpy as np
import tensorflow as tf


def minimax(env: Game ,map, action, flag, depth):
    score = 0
    #map, _ = env.getCustomMapByState(state)
    print("===================")
    env.printCusomMap(map)
    next_map = env.getCustomMoveMap(map, action[0], action[1], action[2], action[3])
    print(action)
    env.printCusomMap(next_map)

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

env = Game()

poslist = env.getPossibleMoveList(1)
map = copy.deepcopy(env.map)
for pos in poslist:
    score = minimax(env, map, pos, 1, 2)
    print(score, pos)

