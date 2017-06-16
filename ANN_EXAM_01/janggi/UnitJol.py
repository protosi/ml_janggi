'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit as CUnit

class UnitJol(CUnit):


    def __init__(self, flag):
        self.setName("JOL");
        self.setFlag(flag);
        self.setScore(400);
        
    
    def getPossibleMoveList(self, maps):
        pos = self.getPos();
      
        