'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Pos import Pos as CPos;
from abc import abstractclassmethod
from abc import ABCMeta

class Unit(metaclass=ABCMeta):
        
    # 1: blue (Cho), 2: red (han)
    iFlag = 0;

    # Unit Name
    strName = '';
    
    # Unit Pos
    mPos = CPos();
    
    # Unit Score
    iScore = 0;
    
    

    def __init__(self):
        '''
        Constructor
        '''  
    
    def setScore(self, score):
        self.iScore = score;
        
    def getScore(self):
        self.iScore;
            
    def getFlag(self):    
        return self.iFlag
    
    def setFlag(self, flag):
        self.iFlag = flag;

    def setPos(self, x, y):
        self.mPos.setPos(x, y);
        
    def getPos(self):
        return self.mPos;
    
    def setName(self, name):
        self.strName = name;
        
    def getName(self):
        self.strName;
        
    def __str__(self):
        flag = "";
        if self.iFlag == 1:
            flag = "B_";
        if self.iFlag == 2:
            flag = "R_";
        
        return "%6s" % (flag + self.strName);
    
    def __repr__(self):
        return self.__str__();
    
    @abstractclassmethod
    def getPossibleMoveList(self):
        '''
        .파이썬은 불편 불편해.
        '''