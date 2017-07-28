import tensorflow as tf
import numpy as np
from collections import deque
from typing import List
import random
from dqn import DQN
from Game import Game
# https://omid.al/posts/2017-02-20-Tutorial-Build-Your-First-Tensorflow-Android-App.html

REPLAY_MEMORY = 50000
BATCH_SIZE = 64
TARGET_UPDATE_FREQUENCY = 5
DISCOUNT_RATE = 1

INPUT_SIZE = 9
OUTPUT_SIZE = 3
MAX_EPISODES = 10000

def replay_train(mainDQN, train_batch):
    choMap = np.vstack([x[0] for x in train_batch])
    hanMap = np.vstack([x[1] for x in train_batch])
    pos = np.array([x[2] for x in train_batch])
    rewards = np.array([[x[3]] for x in train_batch])
    curTurn = np.array([x[4] for x in train_batch])
    maxTurn = np.array([x[5] for x in train_batch])
    curFlag = np.array([x[6] for x in train_batch])
    winFlag = np.array([x[7] for x in train_batch])
    
    for i in range(0, len(curFlag)):
        # 승자와 같은 편이면~ 포인트를 가산한다.
        if(curFlag[i] == winFlag[i]):
            rewards[i][0] += 5000 * (float(curTurn[i] / maxTurn[i]))
        elif (curFlag[i] != winFlag[i]):
            rewards[i][0] -= 5000 * (float(curTurn[i] / maxTurn[i]))
        
        Q_target = np.tanh(rewards / 5000.0)
    return mainDQN.update(choMap, hanMap, pos, Q_target)


def main():
    replay_buffer = deque(maxlen=REPLAY_MEMORY)
    last_100_game_reward = deque(maxlen=100)
    env = Game()
    
    
    
    with tf.Session() as sess:
        mainDQN = DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="main")
        targetDQN = DQN(sess, INPUT_SIZE, OUTPUT_SIZE, name="target")
        sess.run(tf.global_variables_initializer())
        
        saver = tf.train.Saver()
         

        tf.train.write_graph(sess.graph_def, '.', 'tmp/model.pbtxt')  

        copy_ops = get_copy_var_ops(dest_scope_name="target", src_scope_name="main")
        weight = sess.run(copy_ops)
        
        win = 0
        lose = 0
        wrong = 0
        
        temp_win = 0
        temp_lose = 0
        temp_wrong = 0
        
        for episode in range(MAX_EPISODES):
            e = 1. / ((episode/10)+1)

            done = False
            step_count = 0
            env.initGame()
            state = env.getState()
            reward_sum = 0
            
            while not done:
                if np.random.rand() < e:
                    action = env.getMyRandomAction()
                else:
                    predict = mainDQN.predict(state)

                    action = np.argmax(predict)
                    
                reward, done = env.doGame(action)
                next_state = env.getState()
                
                if done :
                    
                    
                    if env.count < 9:
                        wrong += 1
                        temp_wrong += 1
                    else:
                        
                        if env.myPoint > env.emPoint:
                            reward = 100
                            win += 1
                            temp_win += 1
                        elif env.emPoint > env.myPoint:
                            lose += 1
                            temp_lose += 1
                            reward -= 10
                            
                    if episode % 1000 == 0:
                        env.printState()
                        print("episode:", episode, "reward:", reward, "win:", temp_win, "lose:", temp_lose, "wrong:", temp_wrong)
                        temp_win = 0
                        temp_lose = 0
                        temp_wrong = 0
                        saver.save(sess, "tmp/model.ckpt")

                        
                reward_sum += reward
                replay_buffer.append((state, action, reward, next_state, done))
                

                if len(replay_buffer) > BATCH_SIZE:
                    minibatch = random.sample(replay_buffer, BATCH_SIZE)
                    loss, _ = replay_train(mainDQN, targetDQN, minibatch)
                    
                if step_count % TARGET_UPDATE_FREQUENCY == 0:
                    weight = sess.run(copy_ops)

                    
                    
                state = next_state
                step_count += 1

            
            # CartPole-v0 Game Clear Checking Logic
            last_100_game_reward.append(reward_sum)

            if len(last_100_game_reward) == last_100_game_reward.maxlen:
                avg_reward = np.mean(last_100_game_reward)

        print("win:" , win)
        print("lose:", lose)
        print("wrong:", wrong)
        
        saver.save(sess, "tmp/model.ckpt")
        
        
        
main()