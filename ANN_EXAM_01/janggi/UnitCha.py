'''
Created on 2017. 6. 15.

@author: 3F8VJ32
'''

from Unit import Unit as CUnit

class UnitCha(CUnit):


    def __init__(self, flag, Game):
        super(self.__class__, self).__init__(flag, Game);
        self.setName("CHA")
        self.setScore(1000)
        
    
    def getPossibleMoveList(self, maps):
        # 연산을 위해 위치 정보를 획득한다.
        pos = self.getPos()
        now_x = pos.getXPos()
        now_y = pos.getYPos()
        
        
        for i in range(0, now_x):
            '''
            '''
        
        