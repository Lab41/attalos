#!/usr/bin/python
# -*- coding: UTF-8 -*-
"""
Author: Andrew
Github: https://github.com/andrew62
"""
from __future__ import division

import numpy as np
import tensorflow as tf
from zipfile import ZipFile

class ImageProducer(object):

    def __init__(self, img_paths, labels=None, **kwargs):
        """
        A class for loading input images. Images will be transformed
        to the specified img_shape.

        :param img_paths: an iterable containing jpg paths
        :param labels: an iterable with string labels. Default is None.
        :param n_processes: number of threads to use to run enqueue data. default 4
        :param batch_size: number of samples to load. default 128
        :param img_shape: shape of the input image. Default is (224, 224, 3)
        """

        self.img_paths = img_paths
        self.labels = labels

        # Number of threads to use to load data
        self.n_processes = kwargs.pop('n_processes', 4)

        self.batch_size = kwargs.pop('batch_size', min(self.num_images, 128))
        self.img_shape = kwargs.pop("img_shape", (224, 224, 3))

        # Queue to hold the input image paths and labels
        self.data_queue = tf.FIFOQueue(capacity=self.num_images, dtypes=[tf.int32, tf.string],
                                       name="data_queue")
        idx = tf.range(self.num_images)
        # Load the queue
        self.data_enqueue_op = self.data_queue.enqueue_many([idx, self.img_paths])

        # Close queue since we won't add addtional items
        self.closed_queue = self.data_queue.close()

        # load and crop input images
        img_idx, processed_image = self.process()

        # Another queue to hold the image data
        process_queue = tf.FIFOQueue(capacity=np.ceil(self.num_images/self.n_processes),
                                     dtypes=[tf.int32, tf.float32],
                                     shapes=[(), self.img_shape])

        # put all processed images and labels into another queue
        enqueue_process = process_queue.enqueue([img_idx, processed_image])

        # queue to get data out for models
        self.dequeue_op = process_queue.dequeue_many(self.batch_size)

        # ensure we don't create more processes than we need
        n_processes = min(self.n_processes, self.num_images)

        # object to manage queues
        self.queue_runner = tf.train.QueueRunner(process_queue, [enqueue_process] * n_processes)

    def process(self):
        """
        Method to load image data. If you need additional transforms, this is where is should happen

        Returns: index, image
        """
        idx, img_path = self.data_queue.dequeue()
        img = self.load_image(img_path)
        resized = tf.image.resize_images(img, self.img_shape[0], self.img_shape[1])
        return idx, tf.to_float(resized)

    def load_image(self, img_path):
        """
        Method to load a jpg. Could add additional logic to load other image types

        Args:
            img_path: path to an image. Should come from a queue

        Returns: decoded jpg.

        """
        # TODO Images come in from a zipfile. We don't want to write the to temp space
        # Proposed solution
        """
        from zipfile import ZipFile
        archive = Zipfile("myarchive.zip", "r")
        jpgs = filter(lambda x: x.endswith("jpg"), archive.namelist())
        image_data = archive.read(jpgs[0]) # read one at a time
        return tf.image.decode_jpeg(image_data, channels=3)
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
        :return: lables, images. If no labels are provided then index, images are returned
        """
        indices, images = sess.run(self.dequeue_op)
        if self.labels is not None:
            labels = [self.labels[idx] for idx in indices]
            return labels, images
        return indices, images

    @property
    def num_images(self):
        return len(self.img_paths)


class ZipImageProducer(ImageProducer):
    def __init__(self, archive, *args, **kwargs):
        super(ZipImageProducer, self).__init__(*args, **kwargs)
        self.archive = ZipFile(archive, 'r')

    def load_image(self, img_path):
        raw_data = self.archive.read(img_path)
        raw_data_tensor = tf.convert_to_tensor(raw_data, dtype=tf.string)
        return tf.image.decode_jpeg(raw_data_tensor, channels=self.img_shape[-1])

    def __del__(self):
        self.archive.close()



