'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

class Pos():
    iXPos = 0;
    iYPos = 0;

    # 생성자 함수에서 곧바로 Pos 를 지정하게 변경
    def __init__(self, x, y):
        self.setPos(x, y)
    
    def setPos(self, x, y):
        self.iXPos = x;
        self.iYPos = y;
        
    def setXPos(self, x):
        self.iXPos = x;
    
    def setYPos(self, y):
        self.iYPos = y;
        
    def getXPos(self):
        return self.iXPos;
    
    def getYPos(self):
        return self.iYPos;    
    
    def __str__(self):
        return "(X: " + str(self.iXPos) + ", Y:" + str(self.iYPos) + ")";
    
    def __repr__(self):
        return self.__str__();