import numpy as np

import attalos.util.log.log as l
from attalos.imgtxt_algorithms.correlation.correlation import construct_W
from attalos.util.transformers.transformer import Transformer

logger = l.getLogger(__name__)


class NaiveW2V(Transformer):
    def __init__(self, w_arr):
        self.w_arr = w_arr
        
    def transform(self, multihot_arr):
        return np.dot(multihot_arr, self.w_arr.T)
    
    @staticmethod
    def to_multihot(w2v_model, one_hot, predictions, k=5):
        interpreted = []
        logger.info("Converting predictions to multihot.")
        for row_idx, prediction in enumerate(predictions):
            if row_idx % 100 == 0:
                logger.info("%s of %s." % (row_idx, len(predictions)))
            predicted_words = w2v_model.closest_words(prediction, k)
            multihot_predictions = one_hot.get_multiple(predicted_words)
            interpreted.append(multihot_predictions)
        return np.asarray(interpreted)
    
    def save_to_file(self, f):
        np.save(f, self.wdv_matrix)
        
    @classmethod    
    def load_from_file(cls, f):
        wdv_arr = np.load(f)
        return cls(wdv_arr)

    @classmethod
    def create_from_vocab(cls, w2v_model, one_hot, vocab=None):
        if vocab:
            w = construct_W(w2v_model, vocab)
        else:
            logger.warning("No specified vocabulary, using full w2v_model instead.") 
            logger.warning("This may be slow and/or cause a memory error.")
            w = construct_W(w2v_model, w2v_model.vocab)
        return cls(w)
    
    """
    def get_multiple(self, tags):
        global i
        summed = np.zeros(self.w2v_model.get_word_vector_shape())
        # Filter out tags that are not in the w2v_model's vocab
        tags = filter(lambda x: x in self.w2v_model.vocab, tags)
        for tag in tags:
            summed += self.__getitem__(tag)
        return summed
    
    def interpret(self, predictions, k=5):
        interpreted = []   
        for row_idx, prediction in enumerate(predictions):
            predicted_words = self.w2v_model.closest_words(prediction, k)
            multihot_predictions = self.one_hot.get_multiple(predicted)
            interpreted.append(multihot_predictions)
        return np.asarray(interpreted)
    """