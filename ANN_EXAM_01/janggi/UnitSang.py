'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit 
from Pos import Pos

class UnitSang(Unit):


    def __init__(self, flag, Game):
        super(self.__class__, self).__init__(flag, Game);
        self.setName("SANG");
        self.setScore(450);
        
    
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
        
        # 끝이 보인다! 아자아자!
        
        # 상단 이동
        if now_y > 2:
            if current_map[now_y-1][now_x] == 0:
                # 좌상 이동
                if now_x > 1 and current_map[now_y-2][now_x-1] == 0 and ( current_map[now_y-3][now_x-2] == 0 or current_map[now_y-3][now_x-2].getFlag() != self.getFlag()):
                    map[now_y-3][now_x-2] = 1
                    list.append(Pos(now_x-2, now_y-3))
        
                # 우상 이동
                if now_x < 7 and current_map[now_y-2][now_x+1] == 0 and ( current_map[now_y-3][now_x+2] == 0 or current_map[now_y-3][now_x+2].getFlag() != self.getFlag()):
                    map[now_y-3][now_x+2] = 1
                    list.append(Pos(now_x+2, now_y-3))
        
        # 하단 이동            
        if now_y < 7:
            if current_map[now_y+1][now_x] == 0:
                # 좌하 이동
                if now_x > 1 and current_map[now_y+2][now_x-1] == 0 and ( current_map[now_y+3][now_x-2] == 0 or current_map[now_y+3][now_x-2].getFlag() != self.getFlag()):
                    map[now_y+3][now_x-2] = 1
                    list.append(Pos(now_x-2, now_y+3))
                    
                # 우하 이동
                if now_x < 7 and current_map[now_y+2][now_x+1] == 0 and ( current_map[now_y+3][now_x+2] == 0 or current_map[now_y+3][now_x+2].getFlag() != self.getFlag()):
                    map[now_y+3][now_x+2] = 1
                    list.append(Pos(now_x+2, now_y+3))
                    
        # 좌측 이동
        if now_x > 2:
            if current_map[now_y][now_x-1] == 0:
                # 좌상 이동
                if now_y > 1 and current_map[now_y-1][now_x-2] == 0 and ( current_map[now_y-2][now_x-3] == 0 or current_map[now_y-2][now_x-3].getFlag() != self.getFlag()):
                    map[now_y-3][now_x-2] = 1
                    list.append(Pos(now_x-2, now_y-3))
                    
                # 좌하 이동
                if now_y < 8 and current_map[now_y+1][now_x-2] == 0 and ( current_map[now_y+2][now_x-3] == 0 or current_map[now_y+2][now_x-3].getFlag() != self.getFlag()):
                    map[now_y+2][now_x-3] = 1
                    list.append(Pos(now_x-3, now_y+2))
                    
        # 우측 이동
        if now_x < 6:
            if current_map[now_y][now_x+1] == 0:
                # 우상 이동
                if now_y > 0 and current_map[now_y-1][now_x+1] == 0 and ( current_map[now_y-2][now_x+3] == 0 or current_map[now_y-2][now_x+3].getFlag() != self.getFlag()):
                    map[now_y-2][now_x+3] = 1
                    list.append(Pos(now_x+3, now_y-2))
                
                # 우하 이동
                if now_y > 0 and current_map[now_y-1][now_x+1] == 0 and ( current_map[now_y-2][now_x+3] == 0 or current_map[now_y-2][now_x+3].getFlag() != self.getFlag()):
                    map[now_y-2][now_x+3] = 1
                    list.append(Pos(now_x+3, now_y-2))    
                
        return map, list