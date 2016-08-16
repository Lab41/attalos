from attalos.dataset.wordvectors.wrapper import WordVectorWrapper


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