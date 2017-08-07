'''
Created on 2017. 8. 2.

@author: 3F8VJ32
'''
from JsonParsorClass import JsonParsorClass
import numpy as np;


parsor = JsonParsorClass()

a = np.array([ x['turnCount'] for x in parsor.getRandomPanList(5, 0.9)])
print(a)
