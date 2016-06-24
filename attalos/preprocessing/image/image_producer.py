#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Author: Andrew
Github: https://github.com/andrew62
"""
from __future__ import division

import numpy as np
import tensorflow as tf

class ImageProducer(object):

    def __init__(self, img_paths, labels, **kwargs):
        """
        A class for loading input images. Images will be transformed
        to the specified img_shape.

        :param img_paths: an iterable containing jpg paths
        :param labels: an iterable with string labels
        :param n_processes: number of threads to use to run enqueue data. default 4
        :param batch_size: number of samples to load. default 128
        :param img_shape: shape of the input image. Default is (224, 224, 3)
        """

        self.img_paths = img_paths
        self.labels = labels

        # Number of threads to use to load data
        self.n_processes = kwargs.pop('n_processes', 4)

        self.batch_size = kwargs.pop('batch_size', 128)
        self.img_shape = kwargs.pop("img_shape", (224, 224, 3))

        # Queue to hold the input image paths and labels
        self.data_queue = tf.RandomShuffleQueue(capacity=self.num_images, min_after_dequeue=1,
                                                dtypes=[tf.string, tf.string],
                                                name="data_queue")

        # Load the queue
        self.data_enqueue_op = self.data_queue.enqueue_many([self.labels, self.img_paths])

        # Close queue since we won't add addtional items
        self.closed_queue = self.data_queue.close()

        # load and crop input images
        label, processed_image = self.process()

        # Another queue to hold the image data
        process_queue = tf.FIFOQueue(capacity=np.ceil(self.num_images/self.n_processes),
                                     dtypes=[tf.string, tf.float32],
                                     shapes=[(), self.img_shape])

        # put all processed images and labels into another queue
        enqueue_process = process_queue.enqueue([label, processed_image])

        # queue to get data out for models
        self.dequeue_op = process_queue.dequeue_many(self.batch_size)

        # ensure we don't create more processes than we need
        n_processes = min(self.n_processes, self.num_images)

        # object to manage queues
        self.queue_runner = tf.train.QueueRunner(process_queue, [enqueue_process] * n_processes)

    def process(self):
        """
        Method to load image data. If you need additional transforms, this is where is should happen

        Returns: label, image
        """
        label, img_path = self.data_queue.dequeue()
        img = self.load_image(img_path)
        resized = tf.image.resize_images(img, self.img_shape[0], self.img_shape[1])
        return label, tf.to_float(resized)

    def load_image(self, img_path):
        """
        Method to load a jpg. Could add additional logic to load other image types

        Args:
            img_path: path to an image. Should come from a queue

        Returns: decoded jpg.

        """
        raw_data = tf.read_file(img_path)
        return tf.image.decode_jpeg(raw_data, channels=self.img_shape[-1])

    def start(self, sess, coord):
        """
        Starts the queue runners

        :param sess: tensorflow session
        :param coord: tensorflow coordinator
        :return: queue runner threads that need to be joined by a coordinator
        """
        sess.run(self.data_enqueue_op)
        sess.run(self.closed_queue)
        return self.queue_runner.create_threads(sess, coord=coord, start=True)

    def get_batch(self, sess):
        """
        Loads a batch of data

        :param sess: a tensorflow session
        :return: lables, images
        """
        labels, images = sess.run(self.dequeue_op)
        return labels, images

    @property
    def num_images(self):
        return len(self.img_paths)

