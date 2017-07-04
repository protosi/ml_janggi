'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit
from Pos import Pos

class UnitCha(Unit):


    def __init__(self, flag, Game):
        super(self.__class__, self).__init__(flag, Game)
        self.setName("CHA")
        self.setScore(1000)
        
    
    def getPossibleMoveList(self, maps):
        # 연산을 위해 위치 정보를 획득한다.
        pos = self.getPos()
        now_x = pos.getXPos()
        now_y = pos.getYPos()
        
        # 현재 맵을 불러온다
        current_map = self.game.getMap()
        
        # 리턴할 빈 맵을 하나 생성한다.
        map = self.game.getEmptyMap()
        
        # Pos만 리턴할껀디
        list = [];
        
        # x축부터 처리 - LEFT
        for i in range(now_x -1, -1, -1):
            # 공백인 경우에는 당연히 갈 수 있다.
            if current_map[now_y][i] == 0:
                map[now_y][i] = 1
                #Pos(x, y)
                list.append(Pos(i, now_y))
                
            # 다른 유닛인 경우에 당연히 갈 수 있다.
            elif current_map[now_y][i].getFlag() != self.getFlag():
                map[now_y][i] = 1
                #Pos(x, y)
                list.append(Pos(i, now_y))
                #다른 유닛을 뛰어 넘어서까지 갈 수 없으므로 break한다
                break
            # else 인 경우는 아군 유닛이다.
            else:
                break
                
        # x축부터 처리 - RIGHT
        for i in range(now_x + 1, len(map[now_y])):
            # 공백인 경우에는 당연히 갈 수 있다.
            if current_map[now_y][i] == 0:
                map[now_y][i] = 1
                list.append(Pos(i, now_y))
                
            # 다른 유닛인 경우에 당연히 갈 수 있다.
            elif current_map[now_y][i].getFlag() != self.getFlag():
                map[now_y][i] = 1
                #Pos(x, y)
                list.append(Pos(i, now_y))
                break
            # else 인 경우는 아군 유닛이다.
            else:
                break
            
        # y축 처리 - UP
        for i in range(now_y - 1, -1, -1):
            # 공백인 경우에는 당연히 갈 수 있다
            if current_map[i][now_x] == 0:    
                map[i][now_x] = 1;
                #Pos(x, y)
                list.append(Pos(now_x, i))
            # 다른 유닛인 경우에 당연히 갈 수 있다.
            elif current_map[i][now_x].getFlag() != self.getFlag():
                map[i][now_x] = 1
                #Pos(x, y)
                list.append(Pos(now_x, i))
                break
            # else 인 경우는 아군 유닛이다.
            else:
                break
            
        # y축 처리 - DOWN
        for i in range(now_y+1, len(map)):
            # 공백인 경우에는 당연히 갈 수 있다
            if current_map[i][now_x] == 0:    
                map[i][now_x] = 1;
                #Pos(x, y)
                list.append(Pos(now_x, i))
            # 다른 유닛인 경우에 당연히 갈 수 있다.
            elif current_map[i][now_x].getFlag() != self.getFlag():
                map[i][now_x] = 1
                #Pos(x, y)
                list.append(Pos(now_x, i))
                break
            # else 인 경우는 아군 유닛이다.
            else:
                break
            
        return map, list