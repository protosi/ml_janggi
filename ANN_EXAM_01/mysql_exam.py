'''
Created on 2017. 8. 2.

@author: 3F8VJ32
'''
from JsonParsorClass import JsonParsorClass
import numpy as np;


parsor = JsonParsorClass()

a = parsor.getPanList(4)

b = np.array([x['win'] for x in a]);
print (b);