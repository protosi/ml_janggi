'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit as CUnit

class UnitSa(CUnit):


    def __init__(self, flag):
        self.setName("SA");
        self.setFlag(flag);
        self.setScore(600);
        
    
    def getPossibleMoveList(self, maps):
        pos = self.getPos();
      
        