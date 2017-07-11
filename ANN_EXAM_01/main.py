'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Game import Game;
from time import sleep
import os
import copy


game = Game();
while True:
    game.printMap()
    game.doMinMax(game.getMap(), game.m_depth, None, game.getTurn(), game.getTurn())



if __name__ == '__main__':
    pass