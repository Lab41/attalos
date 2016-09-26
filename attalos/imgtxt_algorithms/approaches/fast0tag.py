import numpy as np
import tensorflow as tf

from attalos.imgtxt_algorithms.approaches.base import AttalosModel
from attalos.util.transformers.onehot import OneHot
from attalos.imgtxt_algorithms.correlation.correlation import construct_W
from attalos.imgtxt_algorithms.util.negsamp import NegativeSampler

import attalos.util.log.log as l
logger = l.getLogger(__name__)

class FastZeroTagModel(AttalosModel):
    """
    Create a tensorflow graph that finds the principal direction of the target word embeddings 
    (with negative sampling), using the loss function from "Fast Zero-Shot Image Tagging".
    """
    def __init__(self, wv_model, datasets, **kwargs):
        self.wv_model = wv_model
        self.one_hot = OneHot(datasets, valid_vocab=wv_model.vocab)
        word_counts = NegativeSampler.get_wordcount_from_datasets(datasets, self.one_hot)
        self.fast_sample = kwargs.get("fast_sample", False)
        self.negsampler = NegativeSampler(word_counts, self.fast_sample)
        self.w = construct_W(wv_model, self.one_hot.get_key_ordering()).T
        self.learning_rate = kwargs.get("learning_rate", 0.0001)
        self.optim_words = kwargs.get("optim_words", True)
        self.hidden_units = kwargs.get("hidden_units", "200")
        self.use_batch_norm = kwargs.get("use_batch_norm",False)
        self.opt_type = kwargs.get("opt_type","adam")
        if self.hidden_units=='0':                                                                                                  
            self.hidden_units=[]
        else:
            self.hidden_units = [int(x) for x in self.hidden_units.split(',')]
        self.model_info = dict()
         # Placeholders for data
        self.model_info['input'] = tf.placeholder(shape=(None, datasets[0].img_feat_size), dtype=tf.float32)
        self.model_info['pos_ids'] = tf.placeholder(dtype=tf.int32)
        self.model_info['neg_ids'] = tf.placeholder(dtype=tf.int32)
        
        self.model_info['w2v'] = tf.Variable(self.w, dtype=tf.float32)
        self.model_info['pos_vecs'] = tf.transpose(tf.nn.embedding_lookup(self.model_info['w2v'],
                                                                         self.model_info['pos_ids']), 
                                                  perm=[1,0,2])
        self.model_info['neg_vecs'] = tf.transpose(tf.nn.embedding_lookup(self.model_info['w2v'],
                                                                       self.model_info['neg_ids']), 
                                                perm=[1,0,2])

        # Construct fully connected layers
        layer = self.model_info['input']
        layers = []
        for i, hidden_size in enumerate(self.hidden_units):
            layer = tf.contrib.layers.relu(layer, hidden_size)
            layers.append(layer)
            if self.use_batch_norm:
                layer = tf.contrib.layers.batch_norm(layer)
                layers.append(layer)
                logger.info("Using batch normalization")


        # Output layer should always be linear
        layer = tf.contrib.layers.linear(layer, self.w.shape[1])
        layers.append(layer)

        self.model_info['layers'] = layers
        self.model_info['prediction'] = layer

        
        def fztloss( f, pVecs, nVecs ):
            """
            Tensorized cost function from Fast Zero-Shot Learning paper

            Args:
                f: The output from the network, a tensor of shape (# images, word embedding size)
                pVecs: The vector embeddings of the ground truth tags, a tensor
                    of shape (# images, # positive tags, word embedding size)
                nVecs: The vector embeddings of negatively sampled tags, a tensor
                    of shape (# images, # negative samples, word embedding size)

            Returns:
                Scalar tensor representing the batch cost
            """
            posmul = tf.mul(pVecs, f)
            negmul = tf.mul(nVecs, f)

            tfpos = tf.reduce_sum(posmul, reduction_indices=2)
            tfneg = tf.reduce_sum(negmul, reduction_indices=2)

            tfpos = tf.transpose(tfpos, [1,0])
            tfneg = tf.transpose(tfneg, [1,0])

            negexpan = tf.tile( tf.expand_dims(tfneg, -1), [1, 1, tf.shape(tfpos)[1]] )
            posexpan = tf.tile( tf.transpose(tf.expand_dims(tfpos, -1), [0,2,1]), [1, tf.shape(tfneg)[1], 1])
            differences = tf.sub(negexpan, posexpan)  

            return tf.reduce_sum(tf.reduce_sum(tf.log(1 + tf.exp(differences)), reduction_indices=[1,2]))

        loss = fztloss(self.model_info['prediction'], self.model_info['pos_vecs'], self.model_info['neg_vecs'])
        
        self.model_info['loss'] = loss
        if self.opt_type=='sgd':
            optimizer=tf.train.GradientDescent
        else:
            optimizer=tf.train.AdamOptimizer
        self.model_info['optimizer'] = optimizer(learning_rate=self.learning_rate).minimize(loss)     
        super(FastZeroTagModel, self).__init__()   


    def predict_feats(self, sess, x):
        return sess.run(self.model_info['prediction'], feed_dict={self.model_info['input']: x})

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
                # TODO: Check to see if this benefits from the same bug as negsampling code
                neg_word_ids[ind] = self.negsampler.negsamp_ind(pos_word_ids[ind], 
                                                                numSamps[1])
        
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
