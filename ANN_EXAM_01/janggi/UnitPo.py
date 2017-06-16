'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit as CUnit

class UnitPo(CUnit):


    def __init__(self, flag):
        self.setName("PO");
        self.setFlag(flag);
        self.setScore(850);
        
    
    def getPossibleMoveList(self, maps):
        pos = self.getPos();
      
        