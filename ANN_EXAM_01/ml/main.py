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

def replay_train(mainDQN, targetDQN, train_batch):
    states = np.vstack([x[0] for x in train_batch])
    actions = np.array([x[1] for x in train_batch])
    rewards = np.array([x[2] for x in train_batch])
    next_states = np.vstack([x[3] for x in train_batch])
    done = np.array([x[4] for x in train_batch])
    
    X = states
    
    Q_target = rewards + DISCOUNT_RATE * np.max(targetDQN.predict(next_states), axis=1) * ~done
    y =mainDQN.predict(states)
    y[np.arange(len(X)), actions] = Q_target
    return mainDQN.update(X, y)

def get_copy_var_ops( dest_scope_name, src_scope_name):
    op_holder  = []
    src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=src_scope_name)
    dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=dest_scope_name)
    
    for src_var, dest_var in zip(src_vars, dest_vars):
        op_holder.append(dest_var.assign(src_var.value()))
        
    return op_holder

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