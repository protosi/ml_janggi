import tensorflow as tf
from Game2 import Game
from pg import GameAI
from _operator import pos
import numpy as np

env = Game()

states = np.empty(shape=(0, 10, 9, 4))

states = np.append(states, [env.getState()], axis=0)

print(states.shape)


