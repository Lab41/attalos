
"""
A tensorflow implementation of VGG19. Structure was pulled
from the paper.
"""

import tensorflow as tf

class VGG16(object):

    def __init__(self, n_classes=1000, keep_prob=0.5, train=False):
        self.middle_shape = 25088

        self.keep_prob = tf.constant(keep_prob, name="Dropout", dtype=tf.float32)

        self.input_data = tf.placeholder(dtype=tf.float32, name='input_data', shape=[128, 224, 224, 3])

        # need to subtract
        # self.mean = tf.constant([0, 103.939, 116.779, 123.68], dtype=tf.float32)

        # self.mean_subtract = tf.sub(self.input_data, self.mean)

        with tf.variable_scope("group1_64"):
            with tf.variable_scope("conv1"):
                self.weights1 = tf.get_variable('weights',[3, 3, 3, 64], initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias1 = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
                self.convolve1 = tf.nn.conv2d(self.input_data, self.weights1, [1, 1, 1, 1], padding="SAME")
                self.conv1 = tf.nn.relu(tf.nn.bias_add(self.convolve1, self.bias1))
            with tf.variable_scope("conv2"):
                self.weights2 = tf.get_variable('weights', [3, 3, 64, 64],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias2 = tf.get_variable('bias', [64], initializer=tf.constant_initializer(0.0))
                self.convolve2 = tf.nn.conv2d(self.conv1, self.weights2, [1, 1, 1, 1], padding="SAME")
                self.conv2 = tf.nn.relu(tf.nn.bias_add(self.convolve2, self.bias2))

            self.pool1 = tf.nn.max_pool(self.conv2, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pool1')

        with tf.variable_scope("group2_128"):
            with tf.variable_scope("conv3"):
                self.weights3 = tf.get_variable('weights', [3, 3, 64, 128],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias3 = tf.get_variable('bias', [128], initializer=tf.constant_initializer(0.0))
                self.convolve3 = tf.nn.conv2d(self.pool1, self.weights3, [1, 1, 1, 1], padding="SAME")
                self.conv3 = tf.nn.relu(tf.nn.bias_add(self.convolve3, self.bias3))
            with tf.variable_scope("conv4"):
                self.weights4 = tf.get_variable('weights', [3, 3, 128, 128],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias4 = tf.get_variable('bias', [128], initializer=tf.constant_initializer(0.0))
                self.convolve4 = tf.nn.conv2d(self.conv3, self.weights4, [1, 1, 1, 1], padding="SAME")
                self.conv4 = tf.nn.relu(tf.nn.bias_add(self.convolve4, self.bias4))
            self.pool2 = tf.nn.max_pool(self.conv4, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pool2')

        with tf.variable_scope("group3_256"):
            with tf.variable_scope("conv5"):
                self.weights5 = tf.get_variable('weights', [3, 3, 128, 256],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias5 = tf.get_variable('bias', [256], initializer=tf.constant_initializer(0.0))
                self.convolve5 = tf.nn.conv2d(self.pool2, self.weights5, [1, 1, 1, 1], padding="SAME")
                self.conv5 = tf.nn.relu(tf.nn.bias_add(self.convolve5, self.bias5))
            with tf.variable_scope("conv6"):
                self.weights6 = tf.get_variable('weights', [3, 3, 256, 256],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias6 = tf.get_variable('bias', [256], initializer=tf.constant_initializer(0.0))
                self.convolve6 = tf.nn.conv2d(self.conv5, self.weights6, [1, 1, 1, 1], padding="SAME")
                self.conv6 = tf.nn.relu(tf.nn.bias_add(self.convolve6, self.bias6))
            with tf.variable_scope("conv7"):
                self.weights7 = tf.get_variable('weights', [3, 3, 256, 256],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias7 = tf.get_variable('bias', [256], initializer=tf.constant_initializer(0.0))
                self.convolve7 = tf.nn.conv2d(self.conv6, self.weights7, [1, 1, 1, 1], padding="SAME")
                self.conv7 = tf.nn.relu(tf.nn.bias_add(self.convolve7, self.bias7))
            self.pool3 = tf.nn.max_pool(self.conv7, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name='pool3')

        with tf.variable_scope("group4_512"):
            with tf.variable_scope("conv8"):
                self.weights8 = tf.get_variable('weights', [3, 3, 256, 512],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias8 = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.0))
                self.convolve8 = tf.nn.conv2d(self.pool3, self.weights8, [1, 1, 1, 1], padding="SAME")
                self.conv8 = tf.nn.relu(tf.nn.bias_add(self.convolve8, self.bias8))
            with tf.variable_scope("conv9"):
                self.weights9 = tf.get_variable('weights', [3, 3, 512, 512],
                                                initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias9 = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.0))
                self.convolve9 = tf.nn.conv2d(self.conv8, self.weights9, [1, 1, 1, 1], padding="SAME")
                self.conv9 = tf.nn.relu(tf.nn.bias_add(self.convolve9, self.bias9))
            with tf.variable_scope("conv10"):
                self.weights10 = tf.get_variable('weights', [1, 1, 512, 512],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias10 = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.0))
                self.convolve10 = tf.nn.conv2d(self.conv9, self.weights10, [1, 1, 1, 1], padding="SAME")
                self.conv10 = tf.nn.relu(tf.nn.bias_add(self.convolve10, self.bias10))
            self.pool4 = tf.nn.max_pool(self.conv10, [1, 3, 3, 1], [1, 2, 2, 1], 'SAME', name="pool4")

        with tf.variable_scope("group5_512"):
            with tf.variable_scope("conv11"):
                self.weights11 = tf.get_variable('weights', [3, 3, 512, 512],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias11 = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.0))
                self.convolve11 = tf.nn.conv2d(self.pool4, self.weights11, [1, 1, 1, 1], padding="SAME")
                self.conv11 = tf.nn.relu(tf.nn.bias_add(self.convolve11, self.bias11))

            with tf.variable_scope("conv12"):
                self.weights12 = tf.get_variable('weights', [3, 3, 512, 512],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias12 = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.0))
                self.convolve12 = tf.nn.conv2d(self.conv11, self.weights12, [1, 1, 1, 1], padding="SAME")
                self.conv12 = tf.nn.relu(tf.nn.bias_add(self.convolve12, self.bias12))

            with tf.variable_scope("conv13"):
                self.weights13 = tf.get_variable('weights', [3, 3, 512, 512],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias13 = tf.get_variable('bias', [512], initializer=tf.constant_initializer(0.0))
                self.convolve13 = tf.nn.conv2d(self.conv12, self.weights13, [1, 1, 1, 1], padding="SAME")
                self.conv13 = tf.nn.relu(tf.nn.bias_add(self.convolve13, self.bias13))
            self.pool5 = tf.nn.max_pool(self.conv13, [1, 3, 3, 1], [1, 2, 2, 1], "SAME", name="pool5")
            self.fc6 = tf.reshape(self.pool5, [-1, self.middle_shape], 'fc6')

        with tf.variable_scope("group6_fc"):

            with tf.variable_scope("fc1"):
                self.weights14 = tf.get_variable('weights', [self.middle_shape, 4096],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias14 = tf.get_variable('bias', [4096], initializer=tf.constant_initializer(0.0))
                self.fc1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.fc6, self.weights14), self.bias14))
                if train is True:
                    self.fc1 = tf.nn.dropout(self.fc1, self.keep_prob)

            with tf.variable_scope("fc2"):
                self.weights15 = tf.get_variable('weights', [4096, 4096],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias15 = tf.get_variable('bias', [4096], initializer=tf.constant_initializer(0.0))
                self.fc2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(self.fc1, self.weights15), self.bias15))
                if train is True:
                    self.fc2 = tf.nn.dropout(self.fc2, self.keep_prob)

            with tf.variable_scope("logits"):
                self.weights16 = tf.get_variable('weights', [4096, n_classes],
                                                 initializer=tf.random_normal_initializer(stddev=1e-2))
                self.bias16 = tf.get_variable('bias', [n_classes], initializer=tf.constant_initializer(0.0))
                self.logits = tf.nn.bias_add(tf.matmul(self.fc2, self.weights16), self.bias16)

        self.softmax = tf.nn.softmax(self.logits, 'softmax')


if __name__ == "__main__":
    vgg = VGG16(1000)
