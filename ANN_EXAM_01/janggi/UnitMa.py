'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit
from Pos import Pos

class UnitMa(Unit):

    staticScore = 700

    def __init__(self, flag, Game):
        super(self.__class__, self).__init__(flag, Game);
        self.setName("MA");
        self.setScore(self.staticScore);
        
    
    def getPossibleMoveList(self, current_map=None):
        # 연산을 위해 위치 정보를 획득한다.
        pos = self.getPos()
        now_x = pos.getXPos()
        now_y = pos.getYPos()
        
        # 현재 맵을 불러온다
        if current_map == None:
            current_map = self.game.getMap()
        
        # 리턴할 빈 맵을 하나 생성한다.
        map = self.game.getEmptyMap()
        
        # Pos만 리턴할껀디
        list = [];
      
        # 상단 이동
        if now_y > 1:
            if current_map[now_y-1][now_x] == 0:
                if now_x > 0 and (current_map[now_y-2][now_x-1] == 0 or current_map[now_y-2][now_x-1].getFlag() != self.getFlag()):
                    # 좌상 이동
                    map[now_y-2][now_x-1] = 1
                    list.append(Pos(now_x-1, now_y-2))
                    
                if now_x < 8 and (current_map[now_y-2][now_x+1] == 0 or current_map[now_y-2][now_x+1].getFlag() != self.getFlag()):
                    # 우상 이동
                    map[now_y-2][now_x+1] = 1
                    list.append(Pos(now_x+1, now_y-2))
                    
        # 하단 이동
        if now_y < 8:
            if current_map[now_y+1][now_x] == 0:
                if now_x > 0 and (current_map[now_y+2][now_x-1] == 0 or current_map[now_y+2][now_x-1].getFlag() != self.getFlag()):
                    # 좌하 이동
                    map[now_y+2][now_x-1] = 1
                    list.append(Pos(now_x-1, now_y+2))
                    
                if now_x < 8 and (current_map[now_y+2][now_x+1] == 0 or current_map[now_y+2][now_x+1].getFlag() != self.getFlag()):
                    # 우하 이동
                    map[now_y+2][now_x+1] = 1
                    list.append(Pos(now_x+1, now_y+2))
                    
        # 좌측 이동
        if now_x > 1:
            if current_map[now_y][now_x-1] == 0:
                if now_y > 0 and (current_map[now_y-1][now_x-2] == 0 or current_map[now_y-1][now_x-2].getFlag() != self.getFlag()):
                    # 좌상 이동
                    map[now_y-1][now_x-2] = 1
                    list.append(Pos(now_x-2, now_y-1))
                    
                if now_y < 9 and (current_map[now_y+1][now_x-2] == 0 or current_map[now_y+1][now_x-2].getFlag() != self.getFlag()):
                    # 좌하 이동
                    map[now_y+1][now_x-2] = 1
                    list.append(Pos(now_x-2, now_y+1))
        
        # 우측 이동
        if now_x < 7:
            if current_map[now_y][now_x+1] == 0:
                if now_y > 0 and (current_map[now_y-1][now_x+2] == 0 or current_map[now_y-1][now_x+2].getFlag() != self.getFlag()):
                    # 우상 이동    
                    map[now_y-1][now_x+2] = 1
                    list.append(Pos(now_x+2, now_y-1))
                    
                if now_y < 9 and (current_map[now_y+1][now_x+2] == 0 or current_map[now_y+1][now_x+2].getFlag() != self.getFlag()):
                    # 우하 이동
                    map[now_y+1][now_x+2] = 1
                    list.append(Pos(now_x+2, now_y+1))
                    
        return map, list