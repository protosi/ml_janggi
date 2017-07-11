'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Game import Game;
from time import sleep
import os
import copy

'''
    좌표계 - 예시
    Y   X      0       1       2       3       4       5       6       7       8
    0    [ B_CHA,   B_MA, B_SANG,   B_SA,      0,   B_SA, B_SANG,   B_MA,  B_CHA],
    1    [     0,      0,      0,      0, B_GUNG,      0,      0,      0,      0],
    2    [     0,   B_PO,      0,      0,      0,      0,      0,   B_PO,      0],
    3    [ B_JOL,      0,  B_JOL,      0,  B_JOL,      0,  B_JOL,      0,  B_JOL],
    4    [     0,      0,      0,      0,      0,      0,      0,      0,      0], 
    5    [     0,      0,      0,      0,      0,      0,      0,      0,      0], 
    6    [ R_JOL,      0,  R_JOL,      0,  R_JOL,      0,  R_JOL,      0,  R_JOL], 
    7    [     0,   R_PO,      0,      0,      0,      0,      0,   R_PO,      0], 
    8    [     0,      0,      0,      0, R_GUNG,      0,      0,      0,      0], 
    9    [ R_CHA,   R_MA, R_SANG,   R_SA,      0,   R_SA, R_SANG,   R_MA,  R_CHA]
    '''
game = Game();
game.printMap()
    

while game.isGame:
    game.printMap()
    game.doMinMax(game.getMap(), game.m_depth, None, game.getTurn(), game.getTurn())



if __name__ == '__main__':
    pass