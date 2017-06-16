'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit as CUnit

class UnitMa(CUnit):


    def __init__(self, flag):
        self.setName("MA");
        self.setFlag(flag);
        
    
    def getPossibleMoveList(self, maps):
        pos = self.getPos();
      
        