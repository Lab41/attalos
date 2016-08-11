class W2VWrapper:
    def __init__(self, glove_model):
        self.glove = glove_model
        self.vocab = self.glove.dictionary.keys()
        
    def get_vector(self, word):
        idx = self.glove.dictionary[word]
        return self.glove.word_vectors[idx]
    
    