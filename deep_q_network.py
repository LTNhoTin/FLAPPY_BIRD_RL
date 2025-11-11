#!/usr/bin/env python
from __future__ import print_function

import os
import logging
import random
import sys
from collections import deque

import cv2
import numpy as np
import tensorflow as tf

sys.path.append("game/")
import wrapped_flappy_bird as game

# Use TF2 with v1-style APIs
tf.compat.v1.disable_eager_execution()

# Configure GPU for best-possible performance if available
try:
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logging.getLogger().info("Enabled GPU memory growth for TensorFlow.")
except Exception:
    pass

GAME = 'bird'  # the name of the game being played for log files
ACTIONS = 2  # number of valid actions
GAMMA = 0.99  # decay rate of past observations
OBSERVE = 100000.  # timesteps to observe before training
EXPLORE = 2000000.  # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001  # final value of epsilon
INITIAL_EPSILON = 0.0001  # starting value of epsilon
REPLAY_MEMORY = 50000  # number of previous transitions to remember
BATCH = 512  # size of minibatch
FRAME_PER_ACTION = 1

# Prepare dirs
CHECKPOINT_DIR = "saved_networks"
LOG_FILE = "log.txt"
VIDEO_DIR = "videos"
VIDEO_TEMPLATE = os.path.join(VIDEO_DIR, "train_step_{step:07d}.mp4")
VIDEO_FPS = 60.0

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs("logs_" + GAME, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# Basic logging to file
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

def weight_variable(shape):
    initial = tf.random.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding="SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool2d(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.compat.v1.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def trainNetwork(s, readout, h_fc1, sess):
    # define the cost function
    a = tf.compat.v1.placeholder("float", [None, ACTIONS])
    y = tf.compat.v1.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), axis=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.compat.v1.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.compat.v1.train.Saver(max_to_keep=None)
    sess.run(tf.compat.v1.global_variables_initializer())
    checkpoint = tf.compat.v1.train.get_checkpoint_state(CHECKPOINT_DIR)
    if checkpoint and checkpoint.model_checkpoint_path:
        try:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            logging.info("Successfully loaded: %s", checkpoint.model_checkpoint_path)
        except Exception as e:
            logging.info("Could not restore checkpoint (%s). Starting fresh.", str(e))
    else:
        logging.info("Could not find old network weights. Starting fresh.")

    # setup video writer
    video_writer = None
    current_video_step = None
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    def start_video_writer(step, frame):
        nonlocal video_writer, current_video_step
        if video_writer is not None:
            return
        try:
            height, width = frame.shape[:2]
            video_path = VIDEO_TEMPLATE.format(step=step)
            video_writer = cv2.VideoWriter(video_path, fourcc, VIDEO_FPS, (width, height))
            current_video_step = step
            logging.info("Started recording video segment: %s", video_path)
        except Exception as e:
            logging.warning("Video writer init failed: %s", str(e))
            video_writer = None
            current_video_step = None

    def close_video_writer():
        nonlocal video_writer, current_video_step
        if video_writer is None:
            return
        try:
            video_writer.release()
            if current_video_step is not None:
                logging.info("Closed video segment for step %s", current_video_step)
        except Exception as e:
            logging.warning("Video writer release failed: %s", str(e))
        finally:
            video_writer = None
            current_video_step = None

    # start training
    epsilon = INITIAL_EPSILON
    t = 0
    try:
        while "flappy bird" != "angry bird":
            # choose an action epsilon greedily
            readout_t = readout.eval(feed_dict={s: [s_t]})[0]
            a_t = np.zeros([ACTIONS])
            action_index = 0
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    logging.info("----------Random Action----------")
                    action_index = random.randrange(ACTIONS)
                    a_t[random.randrange(ACTIONS)] = 1
                else:
                    action_index = np.argmax(readout_t)
                    a_t[action_index] = 1
            else:
                a_t[0] = 1 # do nothing

            # scale down epsilon
            if epsilon > FINAL_EPSILON and t > OBSERVE:
                epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORE

            # run the selected action and observe next state and reward
            x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
            # init video writer lazily after we know frame size
            if video_writer is None:
                start_video_writer(t, x_t1_colored)
            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

            # write video frame (convert RGB -> BGR for OpenCV)
            if video_writer is not None:
                try:
                    bgr = cv2.cvtColor(x_t1_colored, cv2.COLOR_RGB2BGR)
                    video_writer.write(bgr)
                except Exception as e:
                    logging.warning("Video write failed: %s", str(e))

            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            # only train if done observing
            if t > OBSERVE:
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH)

                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []
                readout_j1_batch = readout.eval(feed_dict={s: s_j1_batch})
                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append(r_batch[i])
                    else:
                        y_batch.append(r_batch[i] + GAMMA * np.max(readout_j1_batch[i]))

                # perform gradient step
                train_step.run(feed_dict={
                    y: y_batch,
                    a: a_batch,
                    s: s_j_batch}
                )

            # update the old values
            s_t = s_t1
            t += 1

            # save progress every 10000 iterations
            if t % 10000 == 0:
                saver.save(sess, os.path.join(CHECKPOINT_DIR, GAME + '-dqn'), global_step=t)
                close_video_writer()

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            logging.info("TIMESTEP %d / STATE %s / EPSILON %.6f / ACTION %d / REWARD %.4f / Q_MAX %e",
                         t, state, epsilon, action_index, r_t, np.max(readout_t))
            # write info to files
            '''
            if t % 10000 <= 100:
                a_file.write(",".join([str(x) for x in readout_t]) + '\n')
                h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
                cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
            '''
    finally:
        close_video_writer()

def playGame():
    sess = tf.compat.v1.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    try:
        trainNetwork(s, readout, h_fc1, sess)
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

def main():
    logging.info("Starting training...")
    playGame()

if __name__ == "__main__":
    main()