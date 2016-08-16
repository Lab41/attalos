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
    
class GloveWrapper(WordVectorWrapper):
    def __init__(self, glove_model):
        super(GloveWrapper, self).__init__(glove_model)
        
    def get_vocab(self):
        return self.w2v_model.dictionary.keys()
    
    def get_vector(self, word):
        if word not in self.vocab:
            return None
        idx = self.w2v_model.dictionary[word]
        return self.w2v_model.word_vectors[idx]
    
    def closest_words(self, vector, k):
        return self.w2v_model._similarity_query(vector, number=k) 
    
    def get_word_vector_shape(self):
        return self.w2v_model.word_vectors[0].shape
    
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
    
"""
class W2VWrapper:
    def __init__(self, glove_model):
        self.glove = glove_model
        self.vocab = self.glove.dictionary.keys()
        
    def get_vector(self, word):
        return self.__getitem__(word)
        
    def __getitem__(self, item):
        if item not in self.glove.dictionary:
            return None
        idx = self.glove.dictionary[item]
        return self.glove.word_vectors[idx]
"""
    
    
    
    