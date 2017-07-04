'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit

class UnitPo(Unit):


    def __init__(self, flag, Game):
        super(self.__class__, self).__init__(flag, Game);
        self.setName("PO");
        self.setScore(850);
        
    
    def getPossibleMoveList(self):
        pos = self.getPos();
      
        