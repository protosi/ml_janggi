'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit as CUnit

class UnitGung(CUnit):


    def __init__(self, flag):
        self.setName("GUNG");
        self.setFlag(flag);
        self.setScore(20000);
        
    
    def getPossibleMoveList(self, maps):
        pos = self.getPos();
      
        