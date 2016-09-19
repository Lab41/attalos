from abc import ABCMeta, abstractmethod

class WordVectorWrapper:
    __metaclass__ = ABCMeta
    
    def __init__(self, w2v_model):
        self.w2v_model = w2v_model
        self.vocab = self.get_vocab()
        self.word_vector_shape = self.get_word_vector_shape()

    @abstractmethod
    def get_vocab(self):
        pass
    
    @abstractmethod
    def get_vector(self, word):
        pass
    
    @abstractmethod
    def closest_words(self, vector, k):
        pass
    
    @abstractmethod
    def get_word_vector_shape(self):
        pass
    
    def __getitem__(self, item):
        return self.get_vector(item)