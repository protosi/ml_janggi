'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit
from UnitCha import UnitCha
from UnitGung import UnitGung
from UnitJol import UnitJol
from UnitMa import UnitMa
from UnitPo import UnitPo
from UnitSa import UnitSa
from UnitSang import UnitSang


class Game():
    
    # 1 = 초, 2 = 한
    turn = 1
    
    # 초, 한의 스코어
    choScore = 0
    hanScore = 0
    
    '''
    좌표계 - 예시
    Y   X      0       1       2       3       4       5       6       7       8
    0    [ B_CHA,   B_MA, B_SANG,   B_SA,      0,   B_SA, B_SANG,   B_MA,  B_CHA],
    1    [     0,      0,      0,      0, B_GUNG,      0,      0,      0,      0],
    2    [     0,      0,   B_PO,      0,      0,      0,      0,   B_PO,      0],
    3    [ B_JOL,      0,  B_JOL,      0,  B_JOL,      0,  B_JOL,      0,  B_JOL],
    4    [     0,      0,      0,      0,      0,      0,      0,      0,      0], 
    5    [     0,      0,      0,      0,      0,      0,      0,      0,      0], 
    6    [ R_JOL,      0,  R_JOL,      0,  R_JOL,      0,  R_JOL,      0,  R_JOL], 
    7    [     0,      0,   R_PO,      0,      0,      0,      0,   R_PO,      0], 
    8    [     0,      0,      0,      0, R_GUNG,      0,      0,      0,      0], 
    9    [ R_CHA,   R_MA, R_SANG,   R_SA,      0,   R_SA, R_SANG,   R_MA,  R_CHA]
    '''
    map = []
    #unitList = [];
    
    

    def __init__(self):
        self.initMap();
        
        # 턴 초기화
        self.turn = 1
        
        # 스코어 초기화
        self.choScore = 0
        self.hanScore = 0
        
        # 1. CHO
            #CHA
        self.addObjToMap(0, 0, UnitCha(1, self));
        self.addObjToMap(8, 0, UnitCha(1, self));
        
            #MA
        self.addObjToMap(1, 0, UnitMa(1, self));
        self.addObjToMap(7, 0, UnitMa(1, self));
        
            #SANG
        self.addObjToMap(2, 0, UnitSang(1, self));
        self.addObjToMap(6, 0, UnitSang(1, self));
        
            #SA
        self.addObjToMap(3, 0, UnitSa(1, self));
        self.addObjToMap(5, 0, UnitSa(1, self));
        
            #GUNG
        self.addObjToMap(4, 1, UnitGung(1, self));
        
            #PO
        self.addObjToMap(2, 2, UnitPo(1, self));
        self.addObjToMap(7, 2, UnitPo(1, self));
        
            #JOL
        self.addObjToMap(0, 3, UnitJol(1, self));
        self.addObjToMap(2, 3, UnitJol(1, self));
        self.addObjToMap(4, 3, UnitJol(1, self));
        self.addObjToMap(6, 3, UnitJol(1, self));
        self.addObjToMap(8, 3, UnitJol(1, self));
        
        
        # 2. HAN
            #CHA
        self.addObjToMap(0, 9, UnitCha(2, self));
        self.addObjToMap(8, 9, UnitCha(2, self));
        
            #MA        
        self.addObjToMap(1, 9, UnitMa(2, self));
        self.addObjToMap(7, 9, UnitMa(2, self));
        
            #SANG        
        self.addObjToMap(2, 9, UnitSang(2, self));
        self.addObjToMap(6, 9, UnitSang(2, self));
        
            #SA        
        self.addObjToMap(3, 9, UnitSa(2, self));
        self.addObjToMap(5, 9, UnitSa(2, self));
        
            #GUNG
        self.addObjToMap(4, 8, UnitGung(2, self));
        
            #PO
        self.addObjToMap(2, 7, UnitPo(2, self));
        self.addObjToMap(7, 7, UnitPo(2, self));
        
            #JOL
        self.addObjToMap(0, 6, UnitJol(2, self));
        self.addObjToMap(2, 6, UnitJol(2, self));
        self.addObjToMap(4, 6, UnitJol(2, self));
        self.addObjToMap(6, 6, UnitJol(2, self));
        self.addObjToMap(8, 6, UnitJol(2, self));
    
    def getTurn(self):
        return self.turn
    
    def setTurn(self, turn):
        self.turn = turn
        
    def changeTurn(self):
        if self.turn == 1:
            self.turn = 2
        elif self.turn == 2:
            self.turn = 1
        
    def getMove(self, x, y):
        obj = self.map[y][x]
        map = self.getEmptyMap()
        list = []
        # obj가 유닛일 때만 정상적인 길을 보여준다.
        if(isinstance(obj, Unit)):
            map, list = obj.getPossibleMoveList(self.map)
        return map, list
    
    def setMove(self, pre_x, pre_y, new_x, new_y):
        # 잘못된 범위 예외처리
        if(pre_y < 0 or  pre_y >= len(self.map)):
            print("pre_y is wrong value")
            return False
        
        if(pre_x < 0 or pre_x >= len(self.map[pre_y])):
            print("pre_x is wrong value")
            return False
        
        if(new_y < 0 or  new_y >= len(self.map)):
            print("new_y is wrong value")
            return False
        
        if(new_x < 0 or new_x >= len(self.map[new_y])):
            print("new_x is wrong value")
            return False
        
        
        obj = self.map[pre_y][pre_x]
        
        # obj가 Unit의 instance가 아니면 False를 리턴한다.
        if(isinstance(obj, Unit) == False):
            print("map[%d][%d] is not Unit" % pre_x, pre_y)
            return False
        
        flag = obj.getFlag()
        
        # 현재 차례와 움직이려는 말이 다른 경우
        if(self.turn != flag):
            print("wrong flag")
            return False
        
        map, _ = obj.getPossibleMoveList()
        
        if(map[new_y][new_x] != 1):
            return False

        target = self.map[new_y][new_x] 
        
        if(isinstance(target, Unit)):
            score = target.getScore()
            print("score is %s" % score)
            # 초인 경우
            if(flag == 1):
                self.choScore += score
            # 한의 경우
            elif(flag == 2):
                self.hanScore += score
        
        obj.setPos(new_x, new_y)
        self.map[new_y][new_x] = obj
        self.map[pre_y][pre_x] = 0
        
        # turn을 바꾼다
        self.changeTurn()
        print("changeTurn %d" % self.turn)
        return True
    
    def printMap(self):
        
        for i in range(0, len(self.map)):
            if(i == 0):
                print("%6s" % "Y\X", end='')
                for k in range(0, len(self.map[i])):
                    print("%6s" % k, end='')
                print('')
            for j in range(0, len(self.map[i])):
                if(j == 0):
                    print("%6s" % i, end='')
                print("%6s" % self.map[i][j], end='')
            print('')
            
    
    def getMap(self):
        return self.map;
    
    def setMap(self, map):
        self.map = map;
    
    def addObjToMap(self, x, y, obj):
        self.map[y][x] = obj;    
        if isinstance(obj, Unit) == True:
            obj.setPos(x, y);

    
    def getEmptyMap(self):
        #y축 부터
        rt = []
        for i in range(0, 10):
            temp = [] 
            for j in range(0, 9):
                temp.append(0)
            rt.append(temp)
        return rt
        
    def initMap(self):
        #y축 부터
        self.map = []
        for i in range(0, 10):
            temp = [] 
            for j in range(0, 9):
                temp.append(0)
            self.map.append(temp)        