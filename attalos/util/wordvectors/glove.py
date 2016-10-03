import gzip
import os
import numpy as np
from attalos.util.wordvectors.wrapper import WordVectorWrapper

class GloveWrapper(WordVectorWrapper):
    def __init__(self, w2v):
        self.w2v = w2v
        self.vocab = self.w2v.keys()
    def __getitem__(self, key):
        return self.w2v[key]
    def __contains__(self, key):
        return key in self.w2v
    def get_vector(self, key):
        return self.w2v[key]
    def get_word_vector_shape(self):
        return self.w2v[self.vocab[0]].shape
    def get_vocab(self):
        return self.vocab
    def closest_words(self, vector, k):
        raise NotImplementedError('oops')
        
    @staticmethod
    def load(word_vector_file):
        w2v_lookup = {}
        if os.path.exists(word_vector_file):
            if word_vector_file.endswith('.gz'):
                input_file = gzip.open(word_vector_file)
            else:
                input_file = open(word_vector_file)
        else:
            raise IOError('No word vector file specified')

        for i, line in enumerate(input_file):
            first_word = line[:line.find(' ')]
            line = line.strip().split(' ')
            w2v_vector = np.array([float(j) for j in line[1:]])
            # Normalize vector before storing
            w2v_lookup[line[0]] = w2v_vector
        return w2v_lookup

#class GloveWrapper(WordVectorWrapper):
#    def __init__(self, glove_model):
#        super(GloveWrapper, self).__init__(glove_model)
#        
#    def get_vocab(self):
#        return self.w2v_model.dictionary.keys()
#    
#    def get_vector(self, word):
#        if word not in self.vocab:
#            return None
#        idx = self.w2v_model.dictionary[word]
#        return self.w2v_model.word_vectors[idx]
#    
#    def closest_words(self, vector, k):
#        return self.w2v_model._similarity_query(vector, number=k) 
#    
#    def get_word_vector_shape(self):
#        return self.w2v_model.word_vectors[0].shape
