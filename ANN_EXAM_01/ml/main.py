'''
Created on 2017. 7. 20.

@author: 3F8VJ32
'''
import tensorflow as tf
import numpy as np
import random
from collections import deque
from typing import List
from dqn import DQN
from ml import get_copy_var_ops
from ml import replay_train
from Game import Game
from ml import encode
from ml import decode
from ml import convertToOneHot
from time import sleep

INPUT_SIZE = 93 # [9*10 + 3]
OUTPUT_SIZE = 8100 # [9*10*9*10]

DISCOUNT_RATE = 0.99
REPLAY_MEMORY = 50000
BATCH_SIZE = 20
TARGET_UPDATE_FREQUENCY = 5
MAX_EPISODES = 10000

def main():
    env = Game()
    # store the previous observations in replay memory
    replay_buffer = deque(maxlen=REPLAY_MEMORY)

    last_100_game_reward = deque(maxlen=100)

    with tf.Session() as sess:
        mainDQN = DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())

        # initial copy q_net -> target_net
        copy_ops = get_copy_var_ops(dest_scope_name="target",
                                    src_scope_name="main")
        sess.run(copy_ops)

        for episode in range(MAX_EPISODES):
            e = 1. / ((episode / 10) + 1)
            done = False
            step_count = 0
            state = env.reset()
            stage = env.getMap()
            while not done:
                if np.random.rand() < e:
                    print("minmax move")
                    action = env.doMinMaxForML(stage, 2, None, env.getTurn(), env.getTurn())
                    
                    
                    env.printMap()
                else:
                    print("ml move")
                    # Choose an action by greedily from the Q-network
                    action = mainDQN.predict(state)
                    action = np.argmax(action)
                    
                    env.printMap()

                # Get new state and reward from environment
                next_state, reward, done, valid = env.step(action, True)

                if valid == False:
                    # 잘못된 움직임을 리턴하면 제대로 학습이 안된 것이므로
                    # 페널티를 준 후
                    print("########## wrong move ###########")
                    reward = - 1000
                    # 게임을 끝낸다
                    #done = True

                if done:  # Penalty
                    reward = 0

                # Save the experience to our buffer
                replay_buffer.append((state, action, reward, next_state, done))

                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)

                if step_count % TARGET_UPDATE_FREQUENCY == 0:
                    sess.run(copy_ops)

                state = next_state
                step_count += 1

            print("Episode: {}  steps: {}".format(episode, step_count))

            # CartPole-v0 Game Clear Checking Logic
            last_100_game_reward.append(step_count)

            if len(last_100_game_reward) == last_100_game_reward.maxlen:
                avg_reward = np.mean(last_100_game_reward)

                if avg_reward > 199:
                    print("Game Cleared in {episode} episodes with avg reward {avg_reward}")
                    break
                
main()