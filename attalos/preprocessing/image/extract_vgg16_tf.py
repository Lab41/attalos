
from __future__ import absolute_import

import numpy as np
import tensorflow as tf
from attalos.preprocessing.image.vgg16 import VGG16



graph = tf.Graph()
with graph.as_default():
    model = VGG16(1000)

sess = tf.Session(graph=graph):
