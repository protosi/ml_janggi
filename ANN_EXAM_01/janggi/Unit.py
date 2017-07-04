'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Pos import Pos
from abc import abstractclassmethod
from abc import ABCMeta

class Unit(metaclass=ABCMeta):
        
    # 1: blue (초), 2: red (한)
    iFlag = 0;

    # Unit Name
    strName = '';
    
    # Unit Pos
    mPos = None;
    
    # Unit Score
    iScore = 0;
    
    # Game Stage
    game = None
    

    def __init__(self, flag, Game):
        self.mPos = Pos(0, 0)
        self.iFlag = flag
        # 개별 유닛들이 게임 스테이지를 참조할 수 있게 한다
        self.game = Game;
    
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