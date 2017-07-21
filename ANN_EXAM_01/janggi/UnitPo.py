'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit
from Pos import Pos

class UnitPo(Unit):


    def __init__(self, flag, Game):
        super(self.__class__, self).__init__(flag, Game);
        self.setName("PO");
        self.setScore(850);
        
        if(flag == 1):
            self.setId(self.getScore())
        else:
            self.setId(self.getScore() * (-1))
    
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
        
        # 좌측 이동 처리
        if(now_x > 1):
            # 점프 여부를 처리할 플래그
            jump_flag = False
            for i in range(now_x-1, -1, -1):
                # 아직 뛰어 넘지 않았다면
                if(jump_flag == False):
                    # 뛰어 넘을 장기알을 발견했다.
                    if(current_map[now_y][i] != 0):
                        # Name이 같으면, 즉 같은 유닛(포)이면
                        if isinstance(current_map[now_y][i], UnitPo):
                            # 쀍하자.
                            break
                        else:
                            jump_flag = True
                    # 안뛰어 넘었으므로, 이후는 생략하고 이어서 진행한다
                    continue
                
                # 뛰어넘은 후
                if(current_map[now_y][i] == 0):
                    map[now_y][i] = 1
                    list.append(Pos(i, now_y))
                else:
                    if isinstance(current_map[now_y][i], UnitPo) == False  and self.getFlag() != current_map[now_y][i].getFlag():
                        map[now_y][i] = 1
                        list.append(Pos(i, now_y))
                    break
                
        # 우측 이동 처리
        if(now_x < 7):
            jump_flag = False
            for i in range(now_x+1, 9):
                # 아직 뛰어 넘지 않았다면
                if jump_flag == False: 
                    # 뛰어 넘을 장기알을 발견했는지 판별
                    if current_map[now_y][i] != 0:
                        # Name이 같으면, 즉 같은 유닛(Po)이면
                        if isinstance(current_map[now_y][i], UnitPo):
                            # Let's Break
                            break
                        # 포 이외의 유닛이면
                        else:
                            jump_flag = True
                    continue
                
                # 뛰어넘은 후
                if current_map[now_y][i] == 0 :
                    map[now_y][i] = 1
                    list.append(Pos(i, now_y))
                else:
                    if isinstance(current_map[now_y][i], UnitPo) == False and self.getFlag() != current_map[now_y][i].getFlag():
                        map[now_y][i] = 1
                        list.append(Pos(i, now_y))
                    break
                
        # 상측 이동 처리 
        if now_y > 1 : 
            jump_flag = False
            for i in range(now_y-1, -1, -1):
                # 뛰어넘기 전
                if jump_flag == False:
                    # 뛰어 넘을 장기알을 발견했는지 판별 
                    if current_map[i][now_x] != 0:
                        # Name이 같으면, 즉 같은 유닛(Po)이면
                        if isinstance(current_map[i][now_x], UnitPo):
                            # 넘을 수 없으므로 더 이상 진행하지 않는다
                            break
                        else:
                            jump_flag = True
                    continue
                
                # 뛰어넘은 후
                if current_map[i][now_x] == 0:
                    map[i][now_x] = 1
                    list.append(Pos(now_x, i))
                else:
                    if isinstance(current_map[i][now_x], UnitPo) == False and self.getFlag() != current_map[i][now_x].getFlag():
                        map[i][now_x] = 1
                        list.append(Pos(now_x, i))
                    break
                
        # 아래측 이동 처리
        if now_y < 8:
            jump_flag = False
            # 뛰어 넘기 전
            for i in range(now_y+1 , 10):
                if jump_flag == False:
                    if current_map[i][now_x] != 0 :
                        if isinstance(current_map[i][now_x], UnitPo):
                            # 넘을 수 없으므로, 더 이상 진행하지 않는다.
                            
                            break
                        else:
                            
                            jump_flag = True
                    continue
                
                # 뛰어 넘은 후
                if current_map[i][now_x] == 0 :
                    map[i][now_x] = 1
                    list.append(Pos(now_x, i))
                else:
                    if isinstance(current_map[i][now_x], UnitPo) == False and self.getFlag() != current_map[i][now_x].getFlag():
                        map[i][now_x] = 1
                        list.append(Pos(now_x, i))
                    break
                
        # 대각선 이동 처리
        # 상단 - 초 진영에서
        if now_y < 3:
            # 우로 대각 선 이동
            if  now_x - now_y == 3:
                if now_y == 0:
                    if(current_map[now_y+1][now_x+1] != 0 and isinstance(current_map[now_y+1][now_x+1], UnitPo) == False 
                    and (current_map[now_y+2][now_x+2] == 0 
                    or (current_map[now_y+2][now_x+2].getFlag() != self.getFlag() and isinstance(current_map[now_y+2][now_x+2], UnitPo) == False))):
                        map[now_y+2][now_x+2] = 1
                        list.append(Pos(now_x+2, now_y+2))
                        
                if now_y == 2:
                    if(current_map[now_y-1][now_x-1] != 0 and isinstance(current_map[now_y-1][now_x-1], UnitPo) == False
                   and (current_map[now_y-2][now_x-2] == 0 
                    or (current_map[now_y-2][now_x-2].getFlag() != self.getFlag() and isinstance(current_map[now_y-2][now_x-2], UnitPo) == False))):
                        map[now_y-2][now_x-2] = 1
                        list.append(Pos(now_x-2, now_y-2))
                        
            if now_x + now_y == 5:
                if now_y == 0:
                    if(current_map[now_y+1][now_x-1] != 0 and isinstance(current_map[now_y+1][now_x-1], UnitPo) == False
                    and (current_map[now_y+2][now_x-2] == 0
                    or (current_map[now_y+2][now_x-2].getFlag() != self.getFlag() and isinstance(current_map[now_y+2][now_x-2], UnitPo) == False))):
                        map[now_y+2][now_x-2] = 1
                        list.append(Pos(now_x+2, now_y-2))
                        
                if now_y == 2:
                    if(current_map[now_y-1][now_x+1] != 0 and isinstance(current_map[now_y-1][now_x+1], UnitPo) == False
                    and (current_map[now_y-2][now_x+2] == 0
                    or (current_map[now_y-2][now_x+2].getFlag() != self.getFlag() and isinstance(current_map[now_y-2][now_x+2], UnitPo) == False))):
                        map[now_y-2][now_x+2] = 1
                        list.append(Pos(now_x+2, now_y-2))
        
        # 하단 - 한 진영에서
        elif now_y > 6:
            # 우로 대각선 이동
            if now_y - now_x == 4:
                if now_y == 7:
                    if(current_map[now_y+1][now_x+1] != 0 and isinstance(current_map[now_y+1][now_x+1], UnitPo) == False 
                    and (current_map[now_y+2][now_x+2] == 0 
                    or (current_map[now_y+2][now_x+2].getFlag() != self.getFlag() and isinstance(current_map[now_y+2][now_x+2], UnitPo) == False))):
                        map[now_y+2][now_x+2] = 1
                        list.append(Pos(now_x+2, now_y+2))
                        
                if now_y == 9:
                    if(current_map[now_y-1][now_x-1] != 0 and isinstance(current_map[now_y-1][now_x-1], UnitPo) == False 
                    and (current_map[now_y-2][now_x-2] == 0 
                    or (current_map[now_y-2][now_x-2].getFlag() != self.getFlag() and isinstance(current_map[now_y-2][now_x-2], UnitPo) == False))):
                        map[now_y-2][now_x-2] = 1
                        list.append(Pos(now_x-2, now_y-2))
                        
            if now_x + now_y == 12:
                if now_y == 7:
                    if(current_map[now_y+1][now_x-1] != 0 and isinstance(current_map[now_y+1][now_x-1], UnitPo) != False
                    and (current_map[now_y+2][now_x-2] == 0 
                    or (current_map[now_y+2][now_x-2].getFlag() != self.getFlag() and isinstance(current_map[now_y+2][now_x-2], UnitPo) != False))):
                        map[now_y+2][now_x-2] = 1
                        list.append(Pos(now_x-2, now_y+2))
                        
                if now_y == 9:
                    if(current_map[now_y-1][now_x+1] != 0 and isinstance(current_map[now_y-1][now_x+1], UnitPo) != False
                    and (current_map[now_y-2][now_x+2] == 0 
                    or (current_map[now_y-2][now_x+2].getFlag() != self.getFlag() and isinstance(current_map[now_y-2][now_x+2], UnitPo) != False))):
                        map[now_y-2][now_x+2] = 1
                        list.append(Pos(now_x+2, now_y-2))
                        
        return map, list