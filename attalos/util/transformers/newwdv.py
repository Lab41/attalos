import numpy as np

import attalos.util.log.log as l
from attalos.imgtxt_algorithms.correlation.correlation import construct_W, top_n_value_filter, scale3, original_or_top_n_value_filter
from attalos.util.transformers.transformer import Transformer

logger = l.getLogger(__name__)


class WDV(Transformer):
    def __init__(self, wdv_arr):
        self.wdv_arr = wdv_arr
    
        
    def transform(self, multihot_arr, postprocess_fn=lambda x, y: x):
        #logger.debug("Transforming multihot matrix using WDV matrix.")
        transformed = np.dot(multihot_arr, self.wdv_arr)
        #logger.debug("Applying postprocess_fn.")
        transformed = postprocess_fn(transformed, multihot_arr)
        return transformed
    
    @staticmethod
    def preprocess(wdv_arr, top_n=2, suppression_fn=lambda x: 0):
        #logger.debug("Filtering to top %s." % top_n)
        wdv_arr = top_n_value_filter(wdv_arr, top_n, suppression_fn=suppression_fn)
        #logger.debug("Scaling WDV matrix.")
        wdv_arr = scale3(wdv_arr)
        return wdv_arr
    
    @staticmethod
    def postprocess(wdv_arr, multihot_arr, top_n=2):
        #logger.debug("Filtering to original / top %s." % top_n)
        transformed = original_or_top_n_value_filter(wdv_arr, multihot_arr, top_n)
        #logger.debug("Scaling transformed matrix.")
        transformed = scale3(transformed)
        #logger.debug("Replacing NaN in transformed matrix with zero.")
        transformed = np.nan_to_num(transformed)
        return transformed
    
    def save_to_file(self, f):
        np.save(f, self.wdv_arr)
        
        
    @classmethod
    def create_from_vocab(cls, w2v_model, vocab1=None, vocab2=None, preprocess_fn=lambda arr: arr):
        if vocab1:
            logger.debug("Constructing w1.")
            w1 = construct_W(w2v_model, vocab1)
            if vocab2:
                logger.debug("Constructing w2.")
                w2 = construct_W(w2v_model, vocab2)
            else:
                logger.debug("No specified vocab2. Setting w2 = w1.")
                w2 = w1
        else:
            logger.warning("No specified vocabulary, using full w2v_model instead.") 
            logger.warning("This may be slow and/or cause a memory error.")
            w1 = construct_W(w2v_model, w2v_model.vocab)
            w2 = w1
        logger.debug("Multiplying w1 and w2.")
        wdv_arr = np.dot(w1.T, w2)
        logger.debug("Applying preprocess_fn.")
        wdv_arr = preprocess_fn(wdv_arr)
        return cls(wdv_arr)

    
    @classmethod    
    def load_from_file(cls, f):
        wdv_arr = np.load(f)
        return cls(wdv_arr)

"""
class WDV(WVTransformer):
    def transform(self, multihot_arr, top_n=2, suppression_fn=lambda x: 0):
        logger.debug("Creating WDV matrix.")
        wdv_arr = np.dot(self.w.T, self.w)
        logger.debug("Filtering to top %s." % top_n)
        wdv_arr = top_n_value_filter(wdv_arr, top_n, suppression_fn=suppression_fn)
        logger.debug("Scaling WDV matrix.")
        #wdv_arr = scale(wdv_arr)
        wdv_arr = scale2(wdv_arr)
        #wdv_arr = wdv_arr / np.linalg.norm(wdv_arr, 1)
        logger.debug("Transforming multihot matrix using WDV matrix.")
        transformed = np.dot(multihot_arr, wdv_arr)
        logger.debug("Filtering to original / top %s." % top_n)
        transformed = original_or_top_n_value_filter(transformed, multihot_arr, top_n)
        logger.debug("Scaling transformed matrix.")
        #transformed = scale(transformed)
        transformed = scale2(transformed)
        #transformed = transformed / np.linalg.norm(transformed, 1)
        logger.debug("Replacing NaN in transformed matrix with zero.")
        transformed = np.nan_to_num(transformed)
        return transformed

    def to_multihot(self, predictions, k=5):
        interpreted = []
        for row_idx, prediction in enumerate(predictions):
            if row_idx % 100 == 0:
                logger.debug("%s of %s." % (row_idx, len(predictions)))
            top_idxs = prediction.argsort()[::-1][:k]
            multihot_predictions = np.zeros(self.one_hot.vocab_size)
            for idx in top_idxs:
                multihot_predictions[idx] = 1
            interpreted.append(multihot_predictions)
        return np.asarray(interpreted)
"""