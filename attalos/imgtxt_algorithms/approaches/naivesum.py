import numpy as np
import tensorflow as tf
from attalos.util.transformers.newnaivew2v import NaiveW2V

import attalos.util.log.log as l
from attalos.imgtxt_algorithms.approaches.base import AttalosModel
from attalos.util.transformers.onehot import OneHot
from attalos.imgtxt_algorithms.correlation.correlation import construct_W

# Setup global objects
logger = l.getLogger(__name__)

class NaiveSumModel(AttalosModel):
    """
    This model performs linear regression via NN using the naive sum of word vectors as targets.
    """
    def _construct_model_info(self, input_size, output_size, learning_rate,
                              hidden_units=[200]):
        logger.info("Input size: %s" % input_size)
        logger.info("Output size: %s" % output_size)
        
        model_info = {}
        model_info["input"] = tf.placeholder(shape=(None, input_size), dtype=tf.float32)
        model_info["y_truth"] = tf.placeholder(shape=(None, output_size), dtype=tf.float32)
        
        #hidden_layer = tf.contrib.layers.relu(model_info["input"], 1124)
        #hidden_layer1 = tf.contrib.layers.relu(model_info["input"], 1686)
        #hidden_layer2 = tf.contrib.layers.relu(hidden_layer1, 1124)
        #hidden_layer3 = tf.contrib.layers.relu(hidden_layer2, 562)

        layers = []
        layer = model_info["input"]
        for i, hidden_size in enumerate(hidden_units):
            layer = tf.contrib.layers.relu(layer, hidden_size)
            layers.append(layer)
        
        model_info["predictions"] = tf.contrib.layers.fully_connected(layer,
                                                                      output_size,
                                                                      activation_fn=None)
        model_info["loss"] = tf.reduce_sum(tf.square(model_info["predictions"] - model_info["y_truth"]))
        model_info["optimizer"] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model_info["loss"])
        return model_info
    
    def __init__(self, wv_model, datasets, **kwargs):
        self.wv_model = wv_model
        #self.cross_eval = kwargs.get("cross_eval", False)
        #self.one_hot = OneHot([train_dataset] if self.cross_eval else [train_dataset, test_dataset],
        #                      valid_vocab=wv_model.vocab)
        self.one_hot = OneHot(datasets, valid_vocab=wv_model.vocab)
        self.wv_transformer = NaiveW2V.create_from_vocab(wv_model, self.one_hot, vocab=self.one_hot.get_key_ordering())
        train_dataset = datasets[0] # train_dataset should always be first in datasets iterable
        self.learning_rate = kwargs.get("learning_rate", 0.0001)
        self.model_info = self._construct_model_info(
                input_size = train_dataset.img_feat_size,
                output_size = self.wv_model.get_word_vector_shape()[0], 
                learning_rate = self.learning_rate
        )
        self.test_one_hot = None
        super(NaiveSumModel, self).__init__()

    def prep_fit(self, data):
        img_feats_list, text_feats_list = data

        img_feats = np.array(img_feats_list)
        text_feats = [self.one_hot.get_multiple(text_feats) for text_feats in text_feats_list]
        text_feats = np.array(text_feats)
        text_feats = self.wv_transformer.transform(text_feats)

        fetches = [self.model_info["optimizer"], self.model_info["loss"]]
        feed_dict = {
            self.model_info["input"]: img_feats,
            self.model_info["y_truth"]: text_feats
        }
        return fetches, feed_dict

    def prep_predict(self, dataset, cross_eval=False):
        if self.test_one_hot is None or self.test_dataset is dataset:
            self.test_dataset = dataset
            self.test_one_hot = OneHot([dataset], valid_vocab=self.wv_model.vocab)
            self.test_w = construct_W(self.wv_model, self.test_one_hot.get_key_ordering()).T
            
        x = []
        y = []
        for idx in dataset:
            image_feats, text_feats = dataset.get_index(idx)
            text_feats = self.test_one_hot.get_multiple(text_feats)
            x.append(image_feats)
            y.append(text_feats)
        x = np.asarray(x)
        truth = np.asarray(y)

        fetches = [self.model_info["predictions"], ]
        feed_dict = {
            self.model_info["input"]: x
        }
        return fetches, feed_dict, truth

    def post_predict(self, predict_fetches, cross_eval=False):
        if self.test_one_hot is None:
            raise Exception("test_one_hot is not set. Did you call prep_predict to initialize it?")
        predictions = predict_fetches[0]
        predictions = np.dot(predictions, construct_W(self.wv_model, self.test_one_hot.get_key_ordering()))
        return predictions

    def get_training_loss(self, fit_fetches):
        _, loss = fit_fetches
        return loss
