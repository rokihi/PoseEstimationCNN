#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

NUM_CLASSES = 6
IMAGE_SIZE = 56  #28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * 3
ADJUST = IMAGE_SIZE / 28
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
        W_fc1 = weight_variable([7 * ADJUST * 7 * ADJUST * 64, 1024])
        b_fc1 = bias_variable([1024])
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * ADJUST * 7 * ADJUST * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])

    y = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return y


if __name__ == '__main__':
    test_image = []
    # for i in range(1, len(sys.argv)):
    #     img = cv2.imread(sys.argv[i])
    #     img = cv2.resize(img, (28, 28))
    #     test_image.append(img.flatten().astype(np.float32) / 255.0)
    # test_image = np.asarray(test_image)

    images_placeholder = tf.placeholder("float", shape=(None, IMAGE_PIXELS))
    labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
    keep_prob = tf.placeholder("float")

    logits = inference(images_placeholder, keep_prob)
    sess = tf.InteractiveSession()
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, "./model/size{}/model500.ckpt".format(IMAGE_SIZE))

    # pred = np.zeros((len(test_image), 6))
    # imgnum = np.zeros(len(test_image))
    np.set_printoptions(precision=5, suppress=True)

    # if len(sys.argv) > 1:
    #     for i in range(len(test_image)):
    #         pred[i] = logits.eval(
    #             feed_dict={images_placeholder: [test_image[i]],
    #                        keep_prob: 1.0})
    #         #print(sys.argv[i + 1], pred[i])
    #         with open("data/listpose.txt","r") as f:
    #             line=line.rstrip()
    #             l=line.split()
    #             if sys.argv[i +1] == l[0]:
    #                 for j in range(6):
    #                     correct[i]=float(l[j+1])
    #                 error[i]=correct[i]-pred[i]

    linenum = 0
    correct = []
    print "[error]"
    print "trained-----------"
    with open("data/train.txt", "r") as ftrained:
        for line in ftrained:
            line = line.rstrip()
            l = line.split()
            img = cv2.imread(l[0])
            print l[0]
            if img is None:
                print "ERROR READING IMAGE"
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            test_image.append(img.flatten().astype(np.float32) / 255.0)
            correct.append(map(float, l[1:7]))
            #print correct[linenum]
            linenum += 1
        print "linenum", linenum
        correct = np.asarray(correct)
        test_image = np.asarray(test_image)
        pred = np.zeros((len(test_image), 6))
        #error = np.zeros((len(test_image), 6))
        for i in range(len(test_image)):
            pred[i] = logits.eval(feed_dict={
                images_placeholder: [test_image[i]],
                keep_prob: 1.0
            })
        error = correct - pred
        print "correct"
        print correct
        print "pred"
        print pred
        print "error"
        print error

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = Axes3D(fig)
    xcorr = correct[:, 0]
    ycorr = correct[:, 1]
    zcorr = correct[:, 2]
    rzcorr = correct[:, 5]
    xpred = pred[:, 0]
    ypred = pred[:, 1]
    zpred = pred[:, 2]
    rzpred = pred[:, 5]

    for i in range(len(xcorr)):
        distance = np.linalg.norm(correct[i, 0:3] - pred[i, 0:3])

        print correct[i, 0:3]
        if rzcorr[i] == 0:
            ax.quiver(
                xcorr[i],
                ycorr[i],
                zcorr[i],
                xpred[i] - xcorr[i],
                ypred[i] - ycorr[i],
                zpred[i] - zcorr[i],
                length=distance,
                color="r")
        elif rzcorr[i] > 0:
            ax.quiver(
                xcorr[i],
                ycorr[i],
                zcorr[i],
                xpred[i] - xcorr[i],
                ypred[i] - ycorr[i],
                zpred[i] - zcorr[i],
                length=distance,
                color="darkred")
        elif rzcorr[i] < 0:
            ax.quiver(
                xcorr[i],
                ycorr[i],
                zcorr[i],
                xpred[i] - xcorr[i],
                ypred[i] - ycorr[i],
                zpred[i] - zcorr[i],
                length=distance,
                color="lightcoral")

    print "test------------------------------"
    test_image = []
    correct = []
    linenum = 0
    with open("data/test.txt", "r") as ftest:
        for line in ftest:
            line = line.rstrip()
            l = line.split()
            img = cv2.imread(l[0])
            img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
            test_image.append(img.flatten().astype(np.float32) / 255.0)
            correct.append(map(float, l[1:7]))
            #print correct[linenum]
            linenum += 1
        correct = np.asarray(correct)
        print "linenum", linenum
        test_image = np.asarray(test_image)
        pred = np.zeros((len(test_image), 6))
        error = np.zeros((len(test_image), 6))

        for i in range(len(test_image)):
            pred[i] = logits.eval(feed_dict={
                images_placeholder: [test_image[i]],
                keep_prob: 1.0
            })
        error = correct - pred
        print "correct"
        print correct
        print "pred"
        print pred
        print "error"
        print error

    # from mpl_toolkits.mplot3d import Axes3D
    # plt.hold(True)
    # fig = plt.figure()
    # ax = Axes3D(fig)
    xcorr = correct[:, 0]
    ycorr = correct[:, 1]
    zcorr = correct[:, 2]
    rzcorr = correct[:, 5]
    xpred = pred[:, 0]
    ypred = pred[:, 1]
    zpred = pred[:, 2]
    rzpred = pred[:, 5]

    for i in range(len(xcorr)):
        distance = np.linalg.norm(correct[i, 0:3] - pred[i, 0:3])
        print distance
        if rzcorr[i] == 0:
            ax.quiver(
                xcorr[i],
                ycorr[i],
                zcorr[i],
                xpred[i] - xcorr[i],
                ypred[i] - ycorr[i],
                zpred[i] - zcorr[i],
                length=-distance,
                color="b")
        elif rzcorr[i] > 0:
            ax.quiver(
                xcorr[i],
                ycorr[i],
                zcorr[i],
                xpred[i] - xcorr[i],
                ypred[i] - ycorr[i],
                zpred[i] - zcorr[i],
                length=distance,
                color="darkblue")
        elif rzcorr[i] < 0:
            ax.quiver(
                xcorr[i],
                ycorr[i],
                zcorr[i],
                xpred[i] - xcorr[i],
                ypred[i] - ycorr[i],
                zpred[i] - zcorr[i],
                length=distance,
                color="lightblue")

    plt.show()
