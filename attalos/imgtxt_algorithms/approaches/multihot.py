import numpy as np
import tensorflow as tf

from attalos.imgtxt_algorithms.approaches.base import AttalosModel
from attalos.util.transformers.onehot import OneHot


class MultihotModel(AttalosModel):
    """
    This model performs logistic regression via NN using multihot vectors as targets.
    """

    def _construct_model_info(self, input_size, output_size, learning_rate):
        model_info = {}
        model_info["input"] = tf.placeholder(shape=(None, input_size), dtype=tf.float32)
        model_info["y_truth"] = tf.placeholder(shape=(None, output_size), dtype=tf.float32)
        model_info["predictions"] = tf.contrib.layers.fully_connected(model_info["input"],
                                                                      output_size,
                                                                      activation_fn=tf.sigmoid)
        model_info["loss"] = tf.reduce_sum(tf.square(model_info["predictions"] - model_info["y_truth"]))
        model_info["optimizer"] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(model_info["loss"])
        return model_info
    
    def __init__(self, wv_model, datasets, **kwargs): #train_dataset, test_dataset, **kwargs):
        #self.cross_eval = kwargs.get("cross_eval", False)
        self.wv_model = wv_model
        #self.one_hot = OneHot([train_dataset] if self.cross_eval else [train_dataset, test_dataset],
        #                      valid_vocab=wv_model.vocab)
        self.one_hot = OneHot(datasets, valid_vocab=wv_model.vocab)
        train_dataset = datasets[0]  # train_dataset should always be first in datasets iterable
        self.learning_rate = kwargs.get("learning_rate", 0.0001)
        self.model_info = self._construct_model_info(
                input_size = train_dataset.img_feat_size,
                output_size = self.one_hot.vocab_size, 
                learning_rate = self.learning_rate
        )
        self.test_one_hot = None
        super(MultihotModel, self).__init__()

    def prep_fit(self, data):
        img_feats_list, text_feats_list = data

        img_feats = np.array(img_feats_list)
        text_feats = [self.one_hot.get_multiple(text_feats) for text_feats in text_feats_list]
        text_feats = np.array(text_feats)

        fetches = [self.model_info["optimizer"], self.model_info["loss"]]
        feed_dict = {
            self.model_info["input"]: img_feats,
            self.model_info["y_truth"]: text_feats
        }
        return fetches, feed_dict

    def prep_predict(self, dataset, cross_eval=False):
        if cross_eval:
            raise Exception("cross_eval is True. Multihot model does not support cross evaluation.")
        x = []
        y = []
        for idx in dataset:
            image_feats, text_feats = dataset.get_index(idx)
            text_feats = self.test_one_hot.get_multiple(text_feats)
            x.append(image_feats)
            y.append(text_feats)
        x = np.asarray(x)
        y = np.asarray(y)

        fetches = [self.model_info["predictions"],]
        feed_dict = {
            self.model_info["input"]: x
        }
        truth = y
        return fetches, feed_dict, truth

    def post_predict(self, predict_fetches, cross_eval=False):
        if cross_eval:
            raise Exception("cross_eval is True. Multihot model does not support cross evaluation.")
        predictions = predict_fetches[0]
        return predictions

    def get_training_loss(self, fit_fetches):
        _, loss = fit_fetches
        return loss


    """

    # is a generator
    def iter_batches(self, dataset, batch_size):
        # TODO batch_size = -1 should yield the entire dataset
        num_batches = int(dataset.num_images / batch_size)
        for batch in xrange(num_batches):
            img_feats_list, text_feats_list = dataset.get_next_batch(batch_size)

            new_img_feats = np.array(img_feats_list)
            # normalize img_feats
            #new_img_feats = (new_img_feats.T / np.linalg.norm(new_img_feats, axis=1)).T

            new_text_feats = [self.one_hot.get_multiple(text_feats) for text_feats in text_feats_list]
            new_text_feats = np.array(new_text_feats)
            # normalize text feats
            # new_text_feats = (new_text_feats.T / np.linalg.norm(new_text_feats, axis=1)).T

            yield new_img_feats, new_text_feats
    
    def get_eval_data(self, dataset):
        x = []
        y = []
        for idx in dataset:
            image_feats, text_feats = dataset.get_index(idx)
            text_feats = self.one_hot.get_multiple(text_feats)
            x.append(image_feats)
            y.append(text_feats)
        return np.asarray(x), np.asarray(y)
        
    def fit(self, sess, x, y, **kwargs):
        _, loss = sess.run([self.model_info["optimizer"], self.model_info["loss"]],
                           feed_dict={
                               self.model_info["input"]: x,
                               self.model_info["y_truth"]: y
                           })
        return loss
        
    def predict(self, sess, x, y=None):
        fetches = self.model_info["predictions"]
        feed_dict = {self.model_info["input"]: x}
        if y is not None:
            fetches.append(self.model_info["loss"])
            feed_dict[self.model_info["y_truth"]] = y
        fetches = sess.run(fetches, feed_dict=feed_dict)
        if y is not None:
            predictions = fetches[0]
            loss = fetches[1]
        else: 
            predictions = fetches
        return (predictions, loss) if y else predictions

    """

