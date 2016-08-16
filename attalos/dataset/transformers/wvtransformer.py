from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from abc import ABCMeta
from abc import abstractmethod
import numpy as np

from attalos.imgtxt_algorithms.correlation.correlation import construct_W

class WVTransformer(object):
    def __init__(self, one_hot, w2v_model, vocab=None, w_file=None):
        self.one_hot = one_hot
        self.w2v_model = w2v_model
        if w_file:
            self.load_w(f)
        else:
            if vocab:
                self.w = construct_W(self.w2v_model, vocab)
            else:
                self.w = construct_W(self.w2v_model, self.w2v_model.vocab)
        
    def save_w(self, f):
        np.save(f, self.w)
    
    def load_w(self, f):
        self.w = np.load(f)
    
        
    