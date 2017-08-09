'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit
from Pos import Pos

class UnitCha(Unit):
    
    staticScore = 1000


    def __init__(self, flag, Game):
        super(self.__class__, self).__init__(flag, Game);
        self.setName("CHA");
        self.setScore(self.staticScore);
        
    
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
            
        # 상단 대각선 처리
        if(now_y < 3):
            # 우로 대각선 이동
            if(now_x - now_y == 3):
            #{  
                if(now_y == 0):
                    # (now_x+1, now_y+1) 처리
                    if(current_map[now_y+1][now_x+1] == 0 or current_map[now_y+1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y+1))
                        
                    # (now_x+2, now_y+2) 처리
                    if(current_map[now_y+1][now_x+1] == 0 and (current_map[now_y+2][now_x+2] == 0 or current_map[now_y+2][now_x+2].getFlag() != self.getFlag())):
                        map[now_y+2][now_x+2] = 1
                        list.append(Pos(now_x+2, now_y+2))
                        
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
                        
                    # (now_x-2, now_y-2) 처리
                    if(current_map[now_y-1][now_x-1] == 0 and (current_map[now_y-2][now_x-2] == 0 or current_map[now_y-2][now_x-2].getFlag() != self.getFlag())):
                        map[now_y-2][now_x-2] = 1
                        list.append(Pos(now_x-2, now_y-2))
                        
            if(now_x+now_y == 5):
                if(now_y == 0):
                    # (now_x-1, now_y+1) 처리
                    if(current_map[now_y+1][now_x-1] == 0 or current_map[now_y+1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x-1] = 1
                        list.append(Pos(now_x-1, now_y+1))
                        
                    # (now_x-2, now_y+2) 처리
                    if(current_map[now_y+1][now_x-1] == 0 and (current_map[now_y+2][now_x-2] == 0 or current_map[now_y+2][now_x-2].getFlag() != self.getFlag())):
                        map[now_y+2][now_x-2] = 1
                        list.append(Pos(now_x-2, now_y+2))
                        
                if(now_y == 1):
                    # (now_x-1, now_y+1) 처리
                    if(current_map[now_y+1][now_x-1] == 0 or current_map[now_y+1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x-1] = 1
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
                    
                    # (now_x+2, now_y-2) 처리
                    if(current_map[now_y-1][now_x+1] == 0 and (current_map[now_y-2][now_x+2] == 0 or current_map[now_y-2][now_x+2].getFlag() != self.getFlag())):
                        map[now_y-2][now_x+2] = 1
                        list.append(Pos(now_x+2, now_y-2))
            #}
        # 하단 대각선 처리
        elif(now_y > 6):        
            if(now_y - now_x == 4):
                if(now_y == 7):
                #{
                    # (now_x+1, now_y+1) 처리
                    if(current_map[now_y+1][now_x+1] == 0 or current_map[now_y+1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y+1))
                    
                    # (now_x+2, now_y+2) 처리
                    if(current_map[now_y+1][now_x+1] == 0 and(current_map[now_y+2][now_x+2] == 0 or current_map[now_y+2][now_x+2].getFlag() != self.getFlag())):
                        map[now_y+2][now_x+2] = 1
                        list.append(Pos(now_x+2, now_y+2))   
                #}
                if(now_y == 8):
                #{
                    # (now_x+1, now_y+1) 처리
                    if(current_map[now_y+1][now_x+1] == 0 or current_map[now_y+1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y+1))
                    
                    # (now_x-1, now_y-1) 처리
                    if(current_map[now_y-1][now_x-1] == 0 or current_map[now_y-1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x-1] = 1
                        list.append(Pos(now_x-1, now_y-1))
                #}
                if(now_y == 9):
                #{
                    # (now_x-1, now_y-1) 처리
                    if(current_map[now_y-1][now_x-1] == 0 or current_map[now_y-1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x-1] = 1
                        list.append(Pos(now_x-1, now_y-1))
                        
                    # (now_x-2, now_y-2) 처리
                    if(current_map[now_y-1][now_x-1] == 0 and (current_map[now_y-2][now_x-2] == 0 or current_map[now_y-2][now_x-2].getFlag() != self.getFlag())):
                        map[now_y-2][now_x-2] = 1
                        list.append(Pos(now_x-2, now_y-2))
                #}
            if(now_x+now_y == 12):
                if(now_y == 7):
                #{  
                    # (now_x-1, now_y+1) 처리
                    if(current_map[now_y+1][now_x-1] == 0 or current_map[now_y+1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x-1] = 1
                        list.append(Pos(now_x-1, now_y+1))
                        
                    # (now_x-2, now_y+2) 처리
                    if(current_map[now_y+1][now_x-1] == 0 and (current_map[now_y+2][now_x-2] == 0 or current_map[now_y+2][now_x-2].getFlag() != self.getFlag())):
                        map[now_y+2][now_x-2] = 1
                        list.append(Pos(now_x-2, now_y+2)) 
                #}
                if(now_y == 8):
                #{
                    # (now_x-1, now_y+1) 처리
                    if(current_map[now_y+1][now_x-1] == 0 or current_map[now_y+1][now_x-1].getFlag() != self.getFlag()):
                        map[now_y+1][now_x-1] = 1
                        list.append(Pos(now_x-1, now_y+1))
                    # (now_x+1, now_y-1) 처리
                    if(current_map[now_y-1][now_x+1] == 0 or current_map[now_y-1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y-1))
                #}
                if(now_y == 9):
                #{
                    # (now_x+1, now_y-1) 처리
                    if(current_map[now_y-1][now_x+1] == 0 or current_map[now_y-1][now_x+1].getFlag() != self.getFlag()):
                        map[now_y-1][now_x+1] = 1
                        list.append(Pos(now_x+1, now_y-1))
                    
                    # (now_x+2, now_y-2) 처리
                    if(current_map[now_y-1][now_x+1] == 0 and (current_map[now_y-2][now_x+2] == 0 or current_map[now_y-2][now_x+2].getFlag() != self.getFlag())):
                        map[now_y-2][now_x+2] = 1
                        list.append(Pos(now_x+2, now_y-2))
                #}
            
        return map, list
    pass