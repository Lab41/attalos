import numpy as np
import tensorflow as tf
from attalos.util.transformers.onehot import OneHot

import attalos.util.log.log as l
from attalos.imgtxt_algorithms.approaches.base import AttalosModel
from attalos.imgtxt_algorithms.correlation.correlation import scale3
from attalos.util.transformers.newwdv import WDV

# Setup global objects
logger = l.getLogger(__name__)

class WDVModel(AttalosModel):
    """
    This model performs linear regression via NN using the naive sum of word vectors as targets.
    """
    def _construct_model_info(self, input_size, output_size, learning_rate):
        logger.info("Input size: %s" % input_size)
        logger.info("Output size: %s" % output_size)
        
        model_info = {}
        model_info["input"] = tf.placeholder(shape=(None, input_size), dtype=tf.float32)
        model_info["y_truth"] = tf.placeholder(shape=(None, output_size), dtype=tf.float32)
        
        input_output_sum = (input_size+output_size)
        hidden_layer_size = int(input_output_sum*0.5)
        #hidden_layer_size = int(input_output_sum*0.75)
        logger.info("Hidden layer size: %s" % hidden_layer_size)
        hidden_layer = tf.contrib.layers.relu(model_info["input"], hidden_layer_size)
        #hidden_layer_size = int(input_output_sum*0.5)
        logger.info("Hidden layer size: %s" % hidden_layer_size)
        hidden_layer = tf.contrib.layers.relu(hidden_layer, hidden_layer_size)
        #hidden_layer_size = int(input_output_sum*0.25)
        logger.info("Hidden layer size: %s" % hidden_layer_size)
        hidden_layer = tf.contrib.layers.relu(hidden_layer, hidden_layer_size)
        
        model_info["dropout_keep_prob"] = tf.placeholder(tf.float32)
        hidden_layer = tf.nn.dropout(hidden_layer, model_info["dropout_keep_prob"])
        #hidden_layer = tf.contrib.layers.relu(hidden_layer, hidden_layer_size)
        
        model_info["predictions"] = tf.contrib.layers.fully_connected(hidden_layer,
                                                                      output_size,
                                                                      activation_fn=None)
        model_info["loss"] = tf.reduce_sum(tf.square(model_info["predictions"] - model_info["y_truth"]))
        model_info["optimizer"] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model_info["loss"])
        return model_info
    
    def __init__(self, wv_model, datasets, **kwargs): #train_dataset, test_dataset, **kwargs):
        self.wv_model = wv_model
        #self.cross_eval = kwargs.get("cross_eval", False)
        #self.one_hot = OneHot([train_dataset] if self.cross_eval else [train_dataset, test_dataset],
        #                      valid_vocab=wv_model.vocab)
        self.one_hot = OneHot([datasets], valid_vocab=self.wv_model.vocab)
        train_dataset = datasets[0]  # train_dataset should always be first in datasets iterable
        self.wv_transformer = WDV.create_from_vocab(wv_model, vocab1=self.one_hot.get_key_ordering(), preprocess_fn=WDV.preprocess)
        self.learning_rate = kwargs.get("learning_rate", 0.0001)
        self.model_info = self._construct_model_info(
                input_size = train_dataset.img_feat_size,
                output_size = self.one_hot.vocab_size, #self.wv_model.get_word_vector_shape()[0], 
                learning_rate = self.learning_rate
        )
        self.transform_cache = {}
        self.test_one_hot = None
        self.test_wv_transformer = None
        super(WDVModel, self).__init__()
        
    def generate_key(self, word_list):
        return " ".join(sorted(word_list))
        
    def add_cache(self, word_list, val):
        key = self.generate_key(word_list)
        self.transform_cache[key] = val
        
    def get_cache(self, word_list):
        key = self.generate_key(word_list)
        if key in self.transform_cache:
            return self.transform_cache[key]
        else:
            return None

    def prep_fit(self, data):
        img_feats_list, text_feats_list = data

        img_feats = np.array(img_feats_list)
        # normalize img_feats
        # new_img_feats = (new_img_feats.T / np.linalg.norm(new_img_feats, axis=1)).T

        new_text_feats_list = []
        for text_feats in text_feats_list:
            new_text_feats = self.get_cache(text_feats)
            if new_text_feats is None:
                new_text_feats = [self.one_hot.get_multiple(text_feats)]
                new_text_feats = np.array(new_text_feats)
                new_text_feats = self.wv_transformer.transform(new_text_feats, postprocess_fn=WDV.postprocess)
                new_text_feats = new_text_feats[0]  # new_text_feats is a list; get first element
                self.add_cache(text_feats, new_text_feats)
            new_text_feats_list.append(new_text_feats)
        text_feats = np.array(new_text_feats_list)

        fetches = [self.model_info["optimizer"], self.model_info["loss"]]
        feed_dict = {
            self.model_info["input"]: img_feats,
            self.model_info["y_truth"]: text_feats
        }
        return fetches, feed_dict

    def prep_predict(self, dataset, cross_eval=False):
        x = []
        y = []
        if cross_eval:
            self.test_one_hot = OneHot([dataset], valid_vocab=self.wv_model.vocab)
            self.test_wv_transformer = WDV.create_from_vocab(self.wv_model,
                                                             vocab1=self.one_hot.get_key_ordering(),
                                                             vocab2=self.test_one_hot.get_key_ordering(),
                                                             preprocess_fn=scale3)
        else:
            self.test_one_hot = self.one_hot
            self.test_wv_transformer = None

        for idx in dataset:
            image_feats, text_feats = dataset.get_index(idx)
            text_feats = self.test_one_hot.get_multiple(text_feats)
            x.append(image_feats)
            y.append(text_feats)
        x = np.asarray(x)
        y = np.asarray(y)

        fetches = [self.model_info["predictions"], ]
        feed_dict = {
            self.model_info["input"]: x
        }
        truth = y
        return fetches, feed_dict, truth

    def post_predict(self, predict_fetches, cross_eval=False):
        predictions = predict_fetches[0]
        if cross_eval:
            if self.test_wv_transformer is None:
                raise Exception("test_wv_transformers is not set. Did you call prep_predict?")
            predictions = np.dot(predictions, self.test_wv_transformer.wdv_arr)
        return predictions

    def get_training_loss(self, fit_fetches):
        def get_training_loss(self, fit_fetches):
            _, loss = fit_fetches
            return loss


    """

    # is a generator
    # TODO rename get_training_batches
    def iter_batches(self, dataset, batch_size):
        # TODO batch_size = -1 should yield the entire dataset
        num_batches = int(dataset.num_images / batch_size)
        cache_hits = 0
        cache_misses = 0
        
        for batch in xrange(num_batches):
            img_feats_list, text_feats_list = dataset.get_next_batch(batch_size)

            new_img_feats = np.array(img_feats_list)
            # normalize img_feats
            #new_img_feats = (new_img_feats.T / np.linalg.norm(new_img_feats, axis=1)).T

            new_text_feats_list = []
            for text_feats in text_feats_list:
                new_text_feats = self.get_cache(text_feats)
                if new_text_feats is None:
                    #logger.info("Cache miss! %s" % text_feats)
                    new_text_feats = [self.one_hot.get_multiple(text_feats)]
                    new_text_feats = np.array(new_text_feats)
                    new_text_feats = self.wv_transformer.transform(new_text_feats, postprocess_fn=WDV.postprocess)
                    new_text_feats = new_text_feats[0] # new_text_feats is a list; get first element
                    #logger.info("new_text_feats shape: %s" % str(new_text_feats.shape))
                    self.add_cache(text_feats, new_text_feats)
                    cache_misses += 1
                else:
                    #logger.info("Cache hit! %s" % text_feats)
                    cache_hits += 1
                new_text_feats_list.append(new_text_feats)
            new_text_feats_list = np.array(new_text_feats_list) #np.concatenate(new_text_feats_list, axis=0)
            #logger.info("New Text Feats overall List Shape: %s" % str(new_text_feats_list.shape))
            
            yield new_img_feats, new_text_feats_list
        #logger.info("Hits: %s, Misses: %s" % (cache_hits, cache_misses))
    
    # TODO rename get_test_arrs
    def get_eval_data(self, dataset):
        x = []
        y = []
        if self.cross_eval:
            test_one_hot = OneHot([dataset], valid_vocab=self.wv_model.vocab)
            self.test_wv_transformer = WDV.create_from_vocab(self.wv_model,
                                                             vocab1=self.one_hot.get_key_ordering(),
                                                             vocab2=test_one_hot.get_key_ordering(),
                                                             preprocess_fn=scale3)
        for idx in dataset:
            image_feats, text_feats = dataset.get_index(idx)
            if self.cross_eval:
                text_feats = test_one_hot.get_multiple(text_feats)
            else: # self.cross_eval == False
                text_feats = self.one_hot.get_multiple(text_feats)
            x.append(image_feats)
            y.append(text_feats)
        return np.asarray(x), np.asarray(y)
        
    def fit(self, sess, x, y, **kwargs):
        dropout_keep_prob = kwargs.get("dropout_keep_prob", 0.5)
        _, loss = sess.run([self.model_info["optimizer"], self.model_info["loss"]],
                           feed_dict={
                               self.model_info["input"]: x,
                               self.model_info["y_truth"]: y,
                               self.model_info["dropout_keep_prob"]: dropout_keep_prob,
                           })
        return loss
        
    def predict(self, sess, x, y=None):
        fetches = [self.model_info["predictions"]]
        feed_dict = {self.model_info["input"]: x,
                     self.model_info["dropout_keep_prob"]: 1.0}
        #if y is not None:
        #    logger.info("Ignoring y. Cannot evaluate test loss with naivesum model.")
        #    fetches.append(self.model_info["loss"])
        #    feed_dict[self.model_info["y_truth"]] = y
        predictions = sess.run(fetches, feed_dict=feed_dict)
        #if y is not None:
        #    predictions = fetches[0]
        #    loss = fetches[1]
        #else: 
        #    predictions = fetches
        predictions = predictions[0] # predictions is a list
        if self.cross_eval:
            predictions = np.dot(predictions, self.test_wv_transformer.wdv_arr)
        #predictions = NaiveW2V.to_multihot(self.wv_model, self.one_hot, predictions, k=5)    
        return predictions

    """
