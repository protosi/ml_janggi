'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit
from Pos import Pos

class UnitJol(Unit):

    staticScore = 400

    def __init__(self, flag, Game):
        super(self.__class__, self).__init__(flag, Game);
        self.setName("JOL");
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
        
        # 좌측 이동 
        if(now_x > 0 and (current_map[now_y][now_x-1] == 0 or current_map[now_y][now_x-1].getFlag() != self.getFlag())):
            map[now_y][now_x-1] = 1
            list.append(Pos(now_x-1, now_y))
        
        # 우측 이동
        if((now_x < len(current_map[now_y] ) -1)  and (current_map[now_y][now_x+1] == 0 or current_map[now_y][now_x+1].getFlag() != self.getFlag())):
            map[now_y][now_x+1] = 1
            list.append(Pos(now_x+1, now_y))
            
        # 초의 쫄은 내려온다.
        if(self.getFlag() == 1):
            if(now_y < len(current_map) -1 and (current_map[now_y+1][now_x] == 0 or current_map[now_y+1][now_x].getFlag() != self.getFlag())):
                map[now_y+1][now_x] = 1
                list.append(Pos(now_x, now_y+1))
            
        # 한의 쫄은 올라간다.
        elif(self.getFlag() == 2):
            if(now_y > 0 and (current_map[now_y-1][now_x] == 0 or current_map[now_y-1][now_x].getFlag() != self.getFlag())):
                map[now_y-1][now_x] = 1
                list.append(Pos(now_x, now_y-1))
        
        # 대각선 처리
        # 상단 - 초의 진형
        if(now_y < 3):    
            if(now_x - now_y == 3):
                if(now_y == 1):
                    if(current_map[now_y-1][now_x-1] == 0 or current_map[now_y-1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x-1] = 1
                        list.append(Pos(now_x-1, now_y-1))
                if(now_y == 2):
                    if(current_map[now_y-1][now_x-1] == 0 or current_map[now_y-1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x-1] = 1
                        list.append(Pos(now_x-1, now_y -1))
            if(now_x+now_y == 5):
                if(now_y == 1):
                    if(current_map[now_y-1][now_x+1] == 0 or current_map[now_y-1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y-1))
                        
                if(now_y == 2):
                    if(current_map[now_y-1][now_x+1] == 0 or current_map[now_y-1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y-1))
                        
        # 하단 - 한의 진형
        elif(now_y > 6):  
            if(now_y - now_x == 4):
                if(now_y == 7):   
                    if(current_map[now_y+1][now_x+1] == 0 or current_map[now_y+1][now_x+1].getFlag() != self.getFlag()):   
                        map[now_y+1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y+1))
                        
                if(now_y == 8):
                    if(current_map[now_y+1][now_x+1] == 0 or current_map[now_y+1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y+1))
                        
            if(now_x + now_y == 12):
                if(now_y == 7):
                    if(current_map[now_y+1][now_x-1] == 0 or current_map[now_y+1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x-1] = 1
                        list.append(Pos(now_x-1, now_y+1)) 
        return map, list