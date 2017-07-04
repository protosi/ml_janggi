'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit

class UnitMa(Unit):


    def __init__(self, flag, Game):
        super(self.__class__, self).__init__(flag, Game);
        self.setName("MA");
        self.setScore(700);
        
    
    def getPossibleMoveList(self, maps):
        pos = self.getPos();
      
        