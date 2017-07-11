'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit 
from Pos import Pos

class UnitGung(Unit):


    def __init__(self, flag, Game):
        super(self.__class__, self).__init__(flag, Game);
        self.setName("GUNG");
        self.setScore(20000);
        
    
    def getPossibleMoveList(self):
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
        
        
        # 왼쪽으로 이동
        if(now_x > 3):
            # now_x-1, now_y
            if(current_map[now_y][now_x-1] == 0 or current_map[now_y][now_x-1].getFlag() != self.getFlag()):
                map[now_y][now_x-1] = 1
                list.append(Pos(now_x-1, now_y))
        
        # 오른쪽으로 이동
        if(now_x < 5):
            if(current_map[now_y][now_x+1] == 0 or current_map[now_y][now_x+1].getFlag() != self.getFlag()):
                map[now_y][now_x+1] = 1
                list.append(Pos(now_x+1, now_y))
            
        # 상단
        if(now_y < 3):
            if(now_y < 2):
                # (now_x, now_y+1) 처리
                if(current_map[now_y+1][now_x] == 0 or current_map[now_y+1][now_x].getFlag() != self.getFlag()):
                    map[now_y+1][now_x] = 1
                    list.append(Pos(now_x, now_y+1))
            
            if(now_y > 0):
                # (now_x, now_y-1) 처리
                if(current_map[now_y-1][now_x] == 0 or current_map[now_y-1][now_x].getFlag() != self.getFlag()):
                    map[now_y-1][now_x] = 1
                    list.append(Pos(now_x, now_y-1))
            
            # 우측 대각선 이동    
            if(now_x - now_y == 3):
                if(now_y == 0):
                    # (now_x+1, now_y+1) 처리
                    if(current_map[now_y+1][now_x+1] == 0 or current_map[now_y+1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x+1] = 1;
                        list.append(Pos(now_x+1, now_y+1))
                    
                if(now_y == 1):
                    # (now_x+1, now_y+1) 처리
                    if(current_map[now_y+1][now_x+1] == 0 or current_map[now_y+1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y+1))
                        
                    # (now_x-1, now_y-1) 처리
                    if(current_map[now_y-1][now_x-1] == 0 or current_map[now_y-1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x-1] = 1
                        list.append(Pos(now_x-1, now_y-1))
                    
                if(now_y == 2):
                    # (now_x-1, now_y-1) 처리
                    if(current_map[now_y-1][now_x-1] == 0 or current_map[now_y-1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x-1] = 1
                        list.append(Pos(now_x-1, now_y-1))  
                
            if(now_x + now_y == 5):
                if(now_y == 0):
                    # (now_x-1, now_y+1) 처리
                    if(current_map[now_y+1][now_x-1] == 0 or current_map[now_y+1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x-1] = 1;
                        list.append(Pos(now_x-1, now_y+1))
                
                if(now_y == 1):
                    # (now_x-1, now_y+1) 처리
                    if(current_map[now_y+1][now_x-1] == 0 or current_map[now_y+1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x-1] = 1;
                        list.append(Pos(now_x-1, now_y+1))
                    
                    # (now_x+1, now_y-1) 처리
                    if(current_map[now_y-1][now_x+1] == 0 or current_map[now_y-1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y-1))
                
                if(now_y == 2):
                    # (now_x+1, now_y-1) 처리
                    if(current_map[now_y-1][now_x+1] == 0 or current_map[now_y-1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y-1))
        
        elif (now_y > 6):
            # 아래 이동
            if(now_y < 9):
                # (now_x , now_y+1) 처리
                if(current_map[now_y+1][now_x] == 0 or current_map[now_y+1][now_x].getFlag() != self.getFlag()):
                    map[now_y+1][now_x] = 1
                    list.append(Pos(now_x, now_y+1))
                
            # 상단 이동
            if(now_y > 7):
                # (now_x, now_y-1)
                if(current_map[now_y-1][now_x] == 0 or current_map[now_y-1][now_x].getFlag() != self.getFlag()):
                    map[now_y-1][now_x] = 1
                    list.append(Pos(now_x, now_y-1))
                
            if(now_y - now_x == 4):
                if(now_y == 7):
                    # (now_x+1, now_y+1) 처리
                    if(current_map[now_y+1][now_x+1] == 0 or current_map[now_y+1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y+1))
                
                if(now_y == 8):
                    # (now_x+1, now_y+1) 처리
                    if(current_map[now_y+1][now_x+1] == 0 or current_map[now_y+1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y+1))
                    
                    # (now_x-1, now_y-1) 처리
                    if(current_map[now_y-1][now_x-1] == 0 or current_map[now_y-1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x-1] = 1
                        list.append(Pos(now_x-1, now_y-1))
                    
                if(now_y == 9):
                    # (now_x-1, now_y-1) 처리
                    if(current_map[now_y-1][now_x-1] == 0 or current_map[now_y-1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x-1] = 1
                        list.append(Pos(now_x-1, now_y-1))
                    
            if(now_x + now_y == 12):
                if(now_y == 7):
                    # (now_x-1, now_y+1) 처리
                    if(current_map[now_y+1][now_x-1] == 0 or current_map[now_y+1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x-1] = 1
                        list.append(Pos(now_x-1, now_y+1))
                    
                if(now_y == 8):
                    # (now_x-1, now_y+1) 처리
                    if(current_map[now_y+1][now_x-1] == 0 or current_map[now_y+1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x-1] = 1
                        list.append(Pos(now_x-1, now_y+1))
                    
                    # (now_x+1, now_y-1) 처리
                    if(current_map[now_y-1][now_x+1] == 0 or current_map[now_y-1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y-1))
                    
                if(now_y == 9):
                    # (now_x+1, now_y-1) 처리
                    if(current_map[now_y-1][now_x+1] == 0 or current_map[now_y-1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y-1))
                   
        return map, list 
            
        