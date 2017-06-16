'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit
from janggi.UnitCha import UnitCha


class Game():
    '''
    classdocs
    '''
    map = []
    unitList = [];

    def __init__(self):
        self.initMap();
        
        # 1. CHO
        self.addObjToMap(0, 0, UnitCha(1));
        
        # 2. HAN
        self.addObjToMap(0, 9, UnitCha(2));
        
    def getMap(self):
        return self.map;
    
    def setMap(self, map):
        self.map = map;
    
    def addObjToMap(self, x, y, obj):
        self.unitList.append(obj);
        self.setObj(x, y, obj);
    
    def setObj(self, x, y, obj):
        self.map[y][x] = obj;    
        if isinstance(obj, Unit) == True:
            obj.setPos(x, y);
        
    def initMap(self):
        #y축 부터
        for i in range(0, 10):
            temp = [] 
            for j in range(0, 9):
                temp.append(0)
            self.map.append(temp);            