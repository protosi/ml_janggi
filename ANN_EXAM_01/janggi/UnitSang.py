'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit 

class UnitSang(Unit):


    def __init__(self, flag, Game):
        super(self.__class__, self).__init__(flag, Game);
        self.setName("SANG");
        self.setScore(450);
        
    
    def getPossibleMoveList(self):
        pos = self.getPos();
      
        