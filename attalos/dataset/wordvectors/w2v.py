from attalos.dataset.wordvectors.wrapper import WordVectorWrapper
import numpy as np


class W2VWrapper(WordVectorWrapper):
    def __init__(self, w2v_model):
        super(W2VWrapper, self).__init__(w2v_model)
        
    def get_vocab(self):
        return self.w2v_model.vocab
    
    def get_vector(self, word):
        return self.w2v_model.get_vector(word)
    
    def closest_words(self, vector, k):
        metrics = np.dot(self.w2v_model.vectors, vector.T)
        closest_idxs = np.argsort(metrics)[::-1][1:k+1]
        return [self.vocab[idx] for idx in closest_idxs]
    
    def get_word_vector_shape(self):
        return self.w2v_model.vectors[0].shape