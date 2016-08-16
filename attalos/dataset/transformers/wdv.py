from attalos.dataset.transformers.wvtransformer import WVTransformer
import numpy as np

import attalos.util.log.log as l
import logging


logger = l.getLogger(__name__)


class WDV(WVTransformer):
    def transform(self, multihot_arr):
        wdv_arr = np.dot(self.w, self.w.T)
        return np.dot(multihot_arr, wdv_arr)
    
    def to_multihot(self, predictions, k=5):
        interpreted = []   
        for row_idx, prediction in enumerate(predictions):
            if row_idx % 100 == 0:
                logger.debug("%s of %s." % (row_idx, len(predictions)))
            predicted_words = self.w2v_model.closest_words(prediction, k)
            multihot_predictions = self.one_hot.get_multiple(predicted_words)
            interpreted.append(multihot_predictions)
        return np.asarray(interpreted)