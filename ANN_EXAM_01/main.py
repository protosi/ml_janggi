'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Game import Game;
game = Game();



map = game.getMap();
print(map);
cha = map[1][4];

print(cha)
print(cha.getPos())
#cha.setPos(3, 0)

rtmap, list = cha.getPossibleMoveList(map)

print(rtmap);
print(list);


if __name__ == '__main__':
    pass