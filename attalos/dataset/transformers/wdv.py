from attalos.dataset.transformers.wvtransformer import WVTransformer
from attalos.imgtxt_algorithms.correlation.correlation import top_n_value_filter, scale2, original_or_top_n_value_filter

import numpy as np

import attalos.util.log.log as l


logger = l.getLogger(__name__)


class WDV(WVTransformer):
    def transform(self, multihot_arr, top_n=2, suppression_fn=lambda x: 0):
        logger.debug("Creating WDV matrix.")
        wdv_arr = np.dot(self.w.T, self.w)
        logger.debug("Filtering to top %s." % top_n)
        wdv_arr = top_n_value_filter(wdv_arr, top_n, suppression_fn=suppression_fn)
        logger.debug("Scaling WDV matrix.")
        wdv_arr = scale2(wdv_arr)
        logger.debug("Transforming multihot matrix using WDV matrix.")
        transformed = np.dot(multihot_arr, wdv_arr)
        logger.debug("Filtering to original / top %s." % top_n)
        transformed = original_or_top_n_value_filter(transformed, multihot_arr, top_n)
        logger.debug("Scaling transformed matrix.")
        transformed = scale2(transformed)
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
