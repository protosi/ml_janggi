'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Game import Game;
game = Game();


game.printMap()

game.setMove(0, 3, 1, 3);
game.printMap()
game.setMove(0, 6, 1, 6);
game.printMap()
game.setMove(0, 0, 0, 9);
game.printMap()

if __name__ == '__main__':
    pass