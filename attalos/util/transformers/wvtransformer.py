from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import attalos.util.log.log as l

from attalos.imgtxt_algorithms.correlation.correlation import construct_W


logger = l.getLogger(__name__)


class WVTransformer(object):
    def __init__(self, one_hot, w2v_model, vocab=None, w_file=None):
        self.one_hot = one_hot
        self.w2v_model = w2v_model
        self.vocab = vocab
        if w_file:
            self.load_w(w_file)
        else:
            if vocab:
                logger.debug("Constructing W with custom vocab.")
                self.w = construct_W(self.w2v_model, vocab)
            else:
                logger.debug("Constructing W with W2V vocab.")
                self.w = construct_W(self.w2v_model, self.w2v_model.vocab)
            logger.debug("Finished constructing W.")

    def save_w(self, f):
        np.save(f, self.w)

    def load_w(self, f):
        self.w = np.load(f)
