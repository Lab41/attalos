from attalos.dataset.transformers.wvtransformer import WVTransformer
import numpy as np

import attalos.util.log.log as l
import logging


logger = l.getLogger(__name__)


class NaiveW2V(WVTransformer):
    def transform(self, multihot_arr):
        return np.dot(multihot_arr, self.w.T)
    
    def to_multihot(self, predictions, k=5):
        interpreted = []   
        for row_idx, prediction in enumerate(predictions):
            if row_idx % 100 == 0:
                logger.debug("%s of %s." % (row_idx, len(predictions)))
            predicted_words = self.w2v_model.closest_words(prediction, k)
            multihot_predictions = self.one_hot.get_multiple(predicted_words)
            interpreted.append(multihot_predictions)
        return np.asarray(interpreted)
    
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