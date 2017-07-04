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
        
        # 1. CHO
            #CHA
        self.addObjToMap(0, 0, UnitCha(1, self));
        self.addObjToMap(8, 0, UnitCha(1, self));
        
            #MA
        self.addObjToMap(1, 0, UnitMa(1, self));
        self.addObjToMap(7, 0, UnitMa(1, self));
        
            #SANG
        #self.addObjToMap(2, 0, UnitSang(1, self));
        #self.addObjToMap(6, 0, UnitSang(1, self));
        
            #SA
        self.addObjToMap(5, 1, UnitCha(1, self));
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
        
    def getMap(self):
        return self.map;
    
    def setMap(self, map):
        self.map = map;
    
    def addObjToMap(self, x, y, obj):
        #self.unitList.append(obj);
        self.setObj(x, y, obj);
    
    def setObj(self, x, y, obj):
        self.map[y][x] = obj;    
        if isinstance(obj, Unit) == True:
            obj.setPos(x, y);
        
    #def printMap(self):
    #   for(i in range(0, leng(self.map)))
    
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
        for i in range(0, 10):
            temp = [] 
            for j in range(0, 9):
                temp.append(0)
            self.map.append(temp)        