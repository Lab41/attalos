import numpy as np
import tensorflow as tf
from scipy.special import expit as sigmoid

from attalos.imgtxt_algorithms.approaches.base import AttalosModel
from attalos.util.transformers.onehot import OneHot
from attalos.imgtxt_algorithms.correlation.correlation import construct_W
from attalos.imgtxt_algorithms.util.negsamp import NegativeSampler

import attalos.util.log.log as l
logger = l.getLogger(__name__)

class NegSamplingFixedModel(AttalosModel):
    """
    This model performs negative sampling.
    """

    def _construct_model_info(self, input_size, output_size, learning_rate, wv_arr,
                              hidden_units=[200,200],
                              use_batch_norm=True):
        model_info = {}
        model_info["input"] = tf.placeholder(shape=(None, input_size), dtype=tf.float32)

        model_info["w2v"] = tf.Variable(wv_arr, dtype=tf.float32)
        model_info["pos_ids"] = tf.placeholder(dtype=tf.int32)
        model_info["neg_ids"] = tf.placeholder(dtype=tf.int32)
        model_info["pos_vecs"] = tf.transpose(tf.nn.embedding_lookup(model_info["w2v"],
                                                                     model_info["pos_ids"]),
                                                   perm=[1,0,2])
        model_info["neg_vecs"] = tf.transpose(tf.nn.embedding_lookup(model_info["w2v"],
                                                                     model_info["neg_ids"]),
                                                   perm=[1,0,2])
        logger.info("Not optimizing word vectors.")

        # Construct fully connected layers
        layers = []
        layer = model_info["input"]
        for i, hidden_size in enumerate(hidden_units[:-1]):
            layer = tf.contrib.layers.relu(layer, hidden_size)
            layers.append(layer)
            if use_batch_norm:
                layer = tf.contrib.layers.batch_norm(layer)
                layers.append(layer)

        # Output layer should always be linear
        layer = tf.contrib.layers.linear(layer, wv_arr.shape[1])
        layers.append(layer)

        model_info["layers"] = layers
        model_info["prediction"] = layer

        def meanlogsig(predictions, truth):
            reduction_indices = 2
            return tf.reduce_mean(tf.log(tf.sigmoid(tf.reduce_sum(predictions * truth, reduction_indices=reduction_indices))))

        pos_loss = meanlogsig(model_info["prediction"], model_info["pos_vecs"])
        neg_loss = meanlogsig(-model_info["prediction"], model_info["neg_vecs"])
        model_info["loss"] = -(pos_loss + neg_loss)

        model_info["optimizer"] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model_info["loss"])

        return model_info

    def __init__(self, wv_model, datasets, **kwargs):
        self.wv_model = wv_model
        self.one_hot = OneHot(datasets, valid_vocab=wv_model.vocab)
        word_counts = NegativeSampler.get_wordcount_from_datasets(datasets, self.one_hot)
        self.negsampler = NegativeSampler(word_counts)
        train_dataset = datasets[0] # train_dataset should always be first in datasets
        self.w = construct_W(wv_model, self.one_hot.get_key_ordering()).T

        self.learning_rate = kwargs.get("learning_rate", 0.0001)
        self.ignore_posbatch = kwargs.get("ignore_posbatch",False)
        self.joint_factor = kwargs.get("joint_factor",1.0)
        self.hidden_units = kwargs.get("hidden_units", "200,200")
        self.hidden_units = [int(x) for x in self.hidden_units.split(",")]
        self.model_info = self._construct_model_info(
            input_size = train_dataset.img_feat_size,
            output_size = self.one_hot.vocab_size,
            hidden_units=self.hidden_units,
            learning_rate = self.learning_rate,
            wv_arr = self.w
        )
        self.test_one_hot = None
        self.test_w = None
        super(NegSamplingFixedModel, self).__init__()
    
    def _get_ids(self, tag_ids, numSamps=[5, 10], uniform_sampling=False):
        """
        Takes a batch worth of text tags and returns positive/negative ids
        """
        pos_word_ids = np.ones((len(tag_ids), numSamps[0]), dtype=np.int32)
        pos_word_ids.fill(-1)
        for ind, tags in enumerate(tag_ids):
            if len(tags) > 0:
                pos_word_ids[ind] = np.random.choice(tags, size=numSamps[0])
        
        neg_word_ids = None
        if uniform_sampling:
            neg_word_ids = np.random.randint(0, 
                                             self.one_hot.vocab_size, 
                                             size=(len(tag_ids), numSamps[1]))
        else:
            neg_word_ids = np.ones((len(tag_ids), numSamps[1]), dtype=np.int32)
            neg_word_ids.fill(-1)
            for ind in range(pos_word_ids.shape[0]):
                if self.ignore_posbatch:
                    # NOTE: This function call should definitely be pos_word_ids[ind]                                               
                    #          but that results in significantly worse performance                                                  
                    #          I wish I understood why.                                                                             
                    #          I think this means we won't sample any tags that appear in the batch    
                    neg_word_ids[ind] = self.negsampler.negsamp_ind(pos_word_ids, numSamps[1])         
                else:
                    neg_word_ids[ind] = self.negsampler.negsamp_ind(pos_word_ids[ind], numSamps[1])
        
        return pos_word_ids, neg_word_ids

    def prep_fit(self, data):
        img_feats, text_feats_list = data

        text_feat_ids = []
        for tags in text_feats_list:
            text_feat_ids.append([self.one_hot.get_index(tag) for tag in tags if tag in self.one_hot])

        pos_ids, neg_ids = self._get_ids(text_feat_ids)
        self.pos_ids = pos_ids
        self.neg_ids = neg_ids

        fetches = [self.model_info["optimizer"], self.model_info["loss"]]
        feed_dict = {
            self.model_info["input"]: img_feats,
            self.model_info["pos_ids"]: pos_ids,
            self.model_info["neg_ids"]: neg_ids
        }

        return fetches, feed_dict

    def prep_predict(self, dataset, cross_eval=False):
        if cross_eval:
            self.test_one_hot = OneHot([dataset], valid_vocab=self.wv_model.vocab)
            self.test_w = construct_W(self.wv_model, self.test_one_hot.get_key_ordering()).T
        else:
            self.test_one_hot = self.one_hot
            self.test_w = self.w

        x = []
        y = []
        for idx in dataset:
            image_feats, text_feats = dataset.get_index(idx)
            text_feats = self.one_hot.get_multiple(text_feats)
            x.append(image_feats)
            y.append(text_feats)
        x = np.asarray(x)
        y = np.asarray(y)

        fetches = [self.model_info["prediction"], ]
        feed_dict = {
            self.model_info["input"]: x
        }
        truth = y
        return fetches, feed_dict, truth

    def post_predict(self, predict_fetches, cross_eval=False):
        predictions = predict_fetches[0]
        if cross_eval and self.test_w is None:
            raise Exception("test_w is not set. Did you call prep_predict?")
        predictions = np.dot(predictions, self.test_w.T)
        return predictions

    def get_training_loss(self, fit_fetches):
        return fit_fetches[1]




