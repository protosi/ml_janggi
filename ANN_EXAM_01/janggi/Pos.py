'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

class Pos():
    '''
    classdocs
    '''
    
    iXPos = 0;
    iYPos = 0;


    def __init__(self):
        '''
        Constructor
        '''
    
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
    