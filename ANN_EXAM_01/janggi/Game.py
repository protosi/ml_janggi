'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit
from janggi.UnitCha import UnitCha
from janggi.UnitGung import UnitGung
from janggi.UnitJol import UnitJol
from janggi.UnitMa import UnitMa
from janggi.UnitPo import UnitPo
from janggi.UnitSa import UnitSa
from janggi.UnitSang import UnitSang


class Game():
    '''
    classdocs
    '''
    map = []
    unitList = [];

    def __init__(self):
        self.initMap();
        
        # 1. CHO
            #CHA
        self.addObjToMap(0, 0, UnitCha(1));
        self.addObjToMap(8, 0, UnitCha(1));
        
            #MA
        self.addObjToMap(1, 0, UnitMa(1));
        self.addObjToMap(7, 0, UnitMa(1));
        
            #SANG
        self.addObjToMap(2, 0, UnitSang(1));
        self.addObjToMap(6, 0, UnitSang(1));
        
            #SA
        self.addObjToMap(3, 0, UnitSa(1));
        self.addObjToMap(5, 0, UnitSa(1));
        
            #GUNG
        self.addObjToMap(4, 1, UnitGung(1));
        
            #PO
        self.addObjToMap(2, 2, UnitPo(1));
        self.addObjToMap(7, 2, UnitPo(1));
        
            #JOL
        self.addObjToMap(0, 3, UnitJol(1));
        self.addObjToMap(2, 3, UnitJol(1));
        self.addObjToMap(4, 3, UnitJol(1));
        self.addObjToMap(6, 3, UnitJol(1));
        self.addObjToMap(8, 3, UnitJol(1));
        
        
        # 2. HAN
            #CHA
        self.addObjToMap(0, 9, UnitCha(2));
        self.addObjToMap(8, 9, UnitCha(2));
        
            #MA        
        self.addObjToMap(1, 9, UnitMa(2));
        self.addObjToMap(7, 9, UnitMa(2));
        
            #SANG        
        self.addObjToMap(2, 9, UnitSang(2));
        self.addObjToMap(6, 9, UnitSang(2));
        
            #SA        
        self.addObjToMap(3, 9, UnitSa(2));
        self.addObjToMap(5, 9, UnitSa(2));
        
            #GUNG
        self.addObjToMap(4, 8, UnitGung(2));
        
            #PO
        self.addObjToMap(2, 7, UnitPo(2));
        self.addObjToMap(7, 7, UnitPo(2));
        
            #JOL
        self.addObjToMap(0, 6, UnitJol(2));
        self.addObjToMap(2, 6, UnitJol(2));
        self.addObjToMap(4, 6, UnitJol(2));
        self.addObjToMap(6, 6, UnitJol(2));
        self.addObjToMap(8, 6, UnitJol(2));
        
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
        
    #def printMap(self):
    #   for(i in range(0, leng(self.map)))
        
        
    def initMap(self):
        #y축 부터
        for i in range(0, 10):
            temp = [] 
            for j in range(0, 9):
                temp.append(0)
            self.map.append(temp);            