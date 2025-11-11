#!/usr/bin/env python
from __future__ import print_function

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
sys.path.append("game/")
import wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque
import re
import os
import time
from os import path
import multiprocessing

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
OBSERVE = 100000. # timesteps to observe before training
EXPLORE = 2000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.0001 # final value of epsilon
INITIAL_EPSILON = 0.0001 # starting value of epsilon
REPLAY_MEMORY = 50000 # number of previous transitions to remember
BATCH = 32 # size of minibatch
FRAME_PER_ACTION = int(os.environ.get("FRAME_SKIP", "1"))  # 可通过环境变量 FRAME_SKIP 调整帧跳过
TARGET_UPDATE_FREQ = 10000 # hard update the target network every C steps

# 快速模式：默认开启以提升 2-3 倍速度（减少 IO、可选开启 frame-skip）
FAST_MODE = os.environ.get("FAST_MODE", "1") == "1"
PRINT_EVERY = int(os.environ.get("PRINT_EVERY", "1000" if FAST_MODE else "1"))
ENABLE_VIDEO_LOG = os.environ.get("ENABLE_VIDEO_LOG", "0" if FAST_MODE else "1") == "1"

# 尝试优化 OpenCV 和 TensorFlow 线程
try:
    # 让 OpenCV 避免过度抢占线程；也可以改为 cv2.setNumThreads(multiprocessing.cpu_count())
    cv2.setNumThreads(1 if FAST_MODE else max(1, multiprocessing.cpu_count() // 2))
except Exception:
    pass

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork(scope_name, dueling=True):
    with tf.variable_scope(scope_name):
        # network weights
        W_conv1 = tf.get_variable("W_conv1", shape=[8, 8, 4, 32], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b_conv1 = tf.get_variable("b_conv1", shape=[32], initializer=tf.constant_initializer(0.01))

        W_conv2 = tf.get_variable("W_conv2", shape=[4, 4, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b_conv2 = tf.get_variable("b_conv2", shape=[64], initializer=tf.constant_initializer(0.01))

        W_conv3 = tf.get_variable("W_conv3", shape=[3, 3, 64, 64], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b_conv3 = tf.get_variable("b_conv3", shape=[64], initializer=tf.constant_initializer(0.01))

        W_fc1 = tf.get_variable("W_fc1", shape=[1600, 512], initializer=tf.truncated_normal_initializer(stddev=0.01))
        b_fc1 = tf.get_variable("b_fc1", shape=[512], initializer=tf.constant_initializer(0.01))

        # input layer
        s = tf.placeholder("float", [None, 80, 80, 4], name="state")

        # hidden layers
        h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
        h_pool1 = max_pool_2x2(h_conv1)

        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)

        h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)

        h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

        h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

        if dueling:
            # Dueling heads
            W_value = tf.get_variable("W_value", shape=[512, 1], initializer=tf.truncated_normal_initializer(stddev=0.01))
            b_value = tf.get_variable("b_value", shape=[1], initializer=tf.constant_initializer(0.01))
            W_adv = tf.get_variable("W_adv", shape=[512, ACTIONS], initializer=tf.truncated_normal_initializer(stddev=0.01))
            b_adv = tf.get_variable("b_adv", shape=[ACTIONS], initializer=tf.constant_initializer(0.01))

            value = tf.matmul(h_fc1, W_value) + b_value                    # [batch, 1]
            advantage = tf.matmul(h_fc1, W_adv) + b_adv                    # [batch, A]
            advantage_mean = tf.reduce_mean(advantage, axis=1, keepdims=True)
            readout = value + (advantage - advantage_mean)                 # [batch, A]
        else:
            W_fc2 = tf.get_variable("W_fc2", shape=[512, ACTIONS], initializer=tf.truncated_normal_initializer(stddev=0.01))
            b_fc2 = tf.get_variable("b_fc2", shape=[ACTIONS], initializer=tf.constant_initializer(0.01))
            readout = tf.matmul(h_fc1, W_fc2) + b_fc2

        return s, readout, h_fc1

def _get_vars(scope_name):
    return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope_name)

def _build_target_update_ops(source_scope, target_scope):
    source_vars = sorted(_get_vars(source_scope), key=lambda v: v.name)
    target_vars = sorted(_get_vars(target_scope), key=lambda v: v.name)
    return [tf.assign(t, s) for s, t in zip(source_vars, target_vars)]

def trainNetwork(sess):
    # build online and target networks (dueling by default)
    s_online, readout_online, _ = createNetwork("online", dueling=True)
    s_target, readout_target, _ = createNetwork("target", dueling=True)

    # placeholders for training
    a = tf.placeholder("float", [None, ACTIONS], name="action_one_hot")
    y = tf.placeholder("float", [None], name="td_target")
    readout_action = tf.reduce_sum(tf.multiply(readout_online, a), axis=1)
    # Huber loss for stability
    cost = tf.losses.huber_loss(y, readout_action, reduction=tf.losses.Reduction.MEAN)
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # target network hard update ops
    target_update_ops = _build_target_update_ops("online", "target")

    # open up a game state to communicate with emulator
    game_state = game.GameState()

    # store the previous observations in replay memory
    D = deque()

    # ensure log directories exist
    os.makedirs("logs_" + GAME, exist_ok=True)
    os.makedirs("logs_bird", exist_ok=True)

    # printing
    a_file = open("logs_" + GAME + "/readout.txt", 'w')
    h_file = open("logs_" + GAME + "/hidden.txt", 'w')
    train_log = open("logs_bird/log.txt", 'a', buffering=1)

    # get the first state by doing nothing and preprocess the image to 80x80x4
    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t_color, r_0, terminal = game_state.frame_step(do_nothing)
    # init video writer using first colored frame (仅在启用视频日志时)
    video_writer = None
    if ENABLE_VIDEO_LOG:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_path = os.path.join("logs_bird", "train_" + timestamp + ".mp4")
        height, width = x_t_color.shape[0], x_t_color.shape[1]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (width, height))
        if video_writer is not None:
            video_writer.write(x_t_color)

    # preprocess grayscale stack for state
    x_t = cv2.cvtColor(cv2.resize(x_t_color, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    # initialize target net = online net
    sess.run(target_update_ops)
    # use absolute path for checkpoint directory to avoid CWD issues
    base_dir = path.dirname(path.abspath(__file__))
    project_root = path.dirname(base_dir)
    ckpt_dir = os.environ.get("CKPT_DIR") or path.join(project_root, "saved_networks")
    os.makedirs(ckpt_dir, exist_ok=True)
    print("Checkpoint directory:", ckpt_dir)
    checkpoint = tf.train.get_checkpoint_state(ckpt_dir)
    t = 0
    if checkpoint and checkpoint.model_checkpoint_path:
        try:
            saver.restore(sess, checkpoint.model_checkpoint_path)
            print("Successfully loaded:", checkpoint.model_checkpoint_path)
            # Extract timestep from checkpoint path (e.g., "bird-dqn-2920000" -> 2920000)
            match = re.search(r'-(\d+)$', checkpoint.model_checkpoint_path)
            if match:
                t = int(match.group(1))
                print("Resuming from timestep:", t)
        except Exception as e:
            print("Could not restore from existing checkpoint due to architecture mismatch or corruption:", str(e))
            print("Starting training from scratch with new architecture.")
            t = 0
            # re-init variables and re-sync target
            sess.run(tf.global_variables_initializer())
            sess.run(target_update_ops)
    else:
        print("No existing checkpoints. Starting fresh.")

    # start training
    # Calculate epsilon based on current timestep
    if t > OBSERVE:
        epsilon = max(FINAL_EPSILON, INITIAL_EPSILON - (INITIAL_EPSILON - FINAL_EPSILON) * (t - OBSERVE) / EXPLORE)
    else:
        epsilon = INITIAL_EPSILON
    try:
        while "flappy bird" != "angry bird":
            # choose an action epsilon greedily from online network
            readout_t = readout_online.eval(feed_dict={s_online : [s_t]})[0]
            a_t = np.zeros([ACTIONS])
            action_index = 0
            if t % FRAME_PER_ACTION == 0:
                if random.random() <= epsilon:
                    print("----------Random Action----------")
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
            # write colored frame to video
            if video_writer is not None and ENABLE_VIDEO_LOG:
                video_writer.write(x_t1_colored)

            x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
            ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
            x_t1 = np.reshape(x_t1, (80, 80, 1))
            #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
            s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

            # store the transition in D
            D.append((s_t, a_t, r_t, s_t1, terminal))
            if len(D) > REPLAY_MEMORY:
                D.popleft()

            # only train if done observing and have enough samples in replay memory
            if t > OBSERVE and len(D) >= BATCH:
                # sample a minibatch to train on
                minibatch = random.sample(D, BATCH)

                # get the batch variables
                s_j_batch = [d[0] for d in minibatch]
                a_batch = [d[1] for d in minibatch]
                r_batch = [d[2] for d in minibatch]
                s_j1_batch = [d[3] for d in minibatch]

                y_batch = []
                # Double DQN target: a* from online, value from target
                readout_j1_online = readout_online.eval(feed_dict = {s_online : s_j1_batch})
                readout_j1_target = readout_target.eval(feed_dict = {s_target : s_j1_batch})
                for i in range(0, len(minibatch)):
                    terminal = minibatch[i][4]
                    # if terminal, only equals reward
                    if terminal:
                        y_batch.append(r_batch[i])
                    else:
                        best_action = np.argmax(readout_j1_online[i])
                        y_batch.append(r_batch[i] + GAMMA * readout_j1_target[i][best_action])

                # perform gradient step
                train_step.run(feed_dict = {
                    y : y_batch,
                    a : a_batch,
                    s_online : s_j_batch}
                )

            # update the old values
            s_t = s_t1
            t += 1

            # save progress every 10000 iterations
            if t % 10000 == 0:
                saver.save(sess, path.join(ckpt_dir, GAME + '-dqn'), global_step = t)
                save_msg = "[%s] Saved checkpoint at timestep %d" % (time.strftime("%Y-%m-%d %H:%M:%S"), t)
                print(save_msg)
                train_log.write(save_msg + "\n")

            # update target network
            if t % TARGET_UPDATE_FREQ == 0:
                sess.run(target_update_ops)

            # print info
            state = ""
            if t <= OBSERVE:
                state = "observe"
            elif t > OBSERVE and t <= OBSERVE + EXPLORE:
                state = "explore"
            else:
                state = "train"

            if (t % PRINT_EVERY) == 0:
                log_msg = "TIMESTEP %d / STATE %s / EPSILON %f / ACTION %d / REWARD %s / Q_MAX %e" % (
                    t, state, epsilon, action_index, str(r_t), np.max(readout_t)
                )
                print(log_msg)
                train_log.write(log_msg + "\n")

            # write info to files
            '''
            if t % 10000 <= 100:
                a_file.write(",".join([str(x) for x in readout_t]) + '\n')
                h_file.write(",".join([str(x) for x in h_fc1.eval(feed_dict={s:[s_t]})[0]]) + '\n')
                cv2.imwrite("logs_tetris/frame" + str(t) + ".png", x_t1)
            '''
    finally:
        try:
            if video_writer is not None:
                video_writer.release()
        except:
            pass
        try:
            train_log.close()
        except:
            pass

def playGame():
    # 配置 TF Session 线程与显存策略
    cpu_cores = multiprocessing.cpu_count()
    config = tf.ConfigProto(
        intra_op_parallelism_threads=cpu_cores,
        inter_op_parallelism_threads=max(1, cpu_cores // 2),
        allow_soft_placement=True,
        gpu_options=tf.GPUOptions(allow_growth=True)
    )
    sess = tf.InteractiveSession(config=config)
    trainNetwork(sess)

def main():
    playGame()

if __name__ == "__main__":
    main()
