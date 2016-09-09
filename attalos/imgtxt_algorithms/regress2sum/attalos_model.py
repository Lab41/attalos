import numpy as np
import tensorflow as tf

class AttalosModel(object): 
    def __init__(self):
        self.saver = tf.train.Saver()
    
    def initialize_model(self, sess):
        sess.run(tf.initialize_all_variables())
        
    def save(self, sess, model_output_path):
        self.saver.save(sess, model_output_path)

    def load(self, sess, model_input_path):
        self.saver.restore(sess, model_input_path)
        
    def to_batches(self, dataset, batch_size):
        raise NotImplementedError()
    
    def to_ndarrs(self, dataset):
        raise NotImplementedError()
        
    def predict(self, sess, x, y=None):
        raise NotImplementedError()

    def fit(self, sess, x, y, **kwargs):
        raise NotImplementedError()
