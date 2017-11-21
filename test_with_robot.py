#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import cv2
import tf.transformations as tr
import urx
from urx.robotiq_two_finger_gripper import Robotiq_Two_Finger_Gripper

NUM_CLASSES = 6
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3

sys.path.append(".")


def inference(images_placeholder, keep_prob):
    """ モデルを作成する関数

    引数: 
      images_placeholder: inputs()で作成した画像のplaceholder
      keep_prob: dropout率のplace_holder

    返り値:
      cross_entropy: モデルの計算結果
    """

    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(x):
        return tf.nn.max_pool(
            x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)

    with tf.name_scope('fc1') as scope:
        W_fc1 = weight_variable(
            [7 * IMAGE_SIZE / 28 * 7 * IMAGE_SIZE / 28 * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(
            h_pool2, [-1, 7 * IMAGE_SIZE / 28 * 7 * IMAGE_SIZE / 28 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y


def transformPose(tcp_pose, obj_pose):
    robot_pose = np.array(tcp_pose)
    ROT = tr.euler_matrix(*tcp_pose[3:6])
    TRNS = tr.translation_matrix(tcp_pose[0:3])
    MATRIX = np.dot(TRNS, ROT)
    print MATRIX

    pos = np.dot(MATRIX, [obj_pose[0], obj_pose[1], obj_pose[2], 1.0])
    pose = np.append(pos, tcp_pose[3:6])
    return pose


if __name__ == '__main__':
    print "python test_with_robot.py /path/to/image 0or1 degree(0 or -45 or 45)"

    test_image = []
    for i in range(1, len(sys.argv)):
        img = cv2.imread(sys.argv[1])
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        test_image.append(img.flatten().astype(np.float32) / 255.0)
    test_image = np.asarray(test_image)

    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
    keep_prob = tf.placeholder("float")
    logits = inference(images_placeholder, keep_prob)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./model/size28/model.ckpt")

    for i in range(len(test_image)):
        pred = logits.eval(
            feed_dict={images_placeholder: [test_image[i]],
                       keep_prob: 1.0})
        print pred

    ###########################
    robothost = "192.168.1.101"
    arm = urx.Robot(robothost, use_rt=True)
    gripper = Robotiq_Two_Finger_Gripper(arm)

    if sys.argv[3] == 0:
        pose_ofs = [
            0.6597890577558689, -0.10476665093445824, 0.05001604571330823,
            -0.02762120685705565, 2.802975718507989, -0.0205745827681476
        ]
    elif sys.argv[3] < 0:
        pose_ofs = [
            0.6731454996623532, -0.07164352150178421, 0.022193872146894514,
            -0.16545389143737882, 2.915767137505513, -0.28818145704404363
        ]
    elif sys.argv[3] > 0:
        pose_ofs = [
            0.6783020621506362, -0.12574664289932758, 0.020887822822653807,
            0.1714879003207263, 2.8870452777892432, 0.2925037102699276
        ]

    home = [0.7, -0.02, 0.03, -0.2, 2.8, -0.04]
    target = home
    arm.movel(home)

    # for i in range(6):
    #     target[i] = pred[0][i]
    target = transformPose(pose_ofs, pred[0])

    print target
    arm.movel(target)
    sys.exit()

    #ipython
    #%run ~/workspace/python-urx/init.py
