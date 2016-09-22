import numpy as np
import tensorflow as tf
from scipy.special import expit as sigmoid

from attalos.imgtxt_algorithms.approaches.base import AttalosModel
from attalos.util.transformers.onehot import OneHot
from attalos.imgtxt_algorithms.correlation.correlation import construct_W
from attalos.imgtxt_algorithms.util.negsamp import NegativeSampler

import attalos.util.log.log as l
logger = l.getLogger(__name__)

class NegSamplingModel(AttalosModel):
    """
    This model performs negative sampling.
    """

    def _construct_model_info(self, input_size, output_size, learning_rate, wv_arr,
                              hidden_units=[200],
                              optim_words=True,
                              opt_type='adam',
                              use_batch_norm=True,
                              weight_decay=0.0):
        model_info = {}
        model_info["input"] = tf.placeholder(shape=(None, input_size), dtype=tf.float32)

        if optim_words:
            model_info["pos_vecs"] = tf.placeholder(dtype=tf.float32)
            model_info["neg_vecs"] = tf.placeholder(dtype=tf.float32)
            logger.info("Optimization on GPU, word vectors are stored separately.")
        else:
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
        for i, hidden_size in enumerate(hidden_units):
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

        # Decide whether or not to use SGD or Adam Optimizers
        if self.opt_type == 'sgd':
            logger.info("Optimization uses SGD with non-variable rate")
            optimizer = tf.train.GradientDescentOptimizer
        else:
            logger.info("Optimization uses Adam")
            optimizer = tf.train.AdamOptimizer
            
        # Are we manually decaying the words? Create a TF variable in that case. 
        if weight_decay:
            logger.info("Learning rate is manually adaptive, dropping every ten (hard coded) epoch")
            model_info['learning_rate'] = tf.placeholder(tf.float32, shape=[])
        else:
            model_info['learning_rate'] = learning_rate
            
        model_info["optimizer"] = optimizer(learning_rate=model_info['learning_rate']).minimize(model_info["loss"])

        #model_info["init_op"] = tf.initialize_all_variables()
        #model_info["saver"] = tf.train.Saver()

        return model_info

    def __init__(self, wv_model, datasets, **kwargs):
        self.wv_model = wv_model
        self.one_hot = OneHot(datasets, valid_vocab=wv_model.vocab)
        word_counts = NegativeSampler.get_wordcount_from_datasets(datasets, self.one_hot)
        self.negsampler = NegativeSampler(word_counts)
        train_dataset = datasets[0] # train_dataset should always be first in datasets
        self.w = construct_W(wv_model, self.one_hot.get_key_ordering()).T

        # Optimization parameters
        # Starting learning rate, currently default to 0.001. This will change iteratively if decay is on.
        self.learning_rate = kwargs.get("learning_rate", 0.0001)
        self.weight_decay = kwargs.get("weight_decay", 0.0)
        self.optim_words = kwargs.get("optim_words", True)
        self.epoch_num = 0
        
        # Sampling methods
        self.ignore_posbatch = kwargs.get("ignore_posbatch",False)
        self.joint_factor = kwargs.get("joint_factor",1.0)
        self.hidden_units = kwargs.get("hidden_units", "200")
        if self.hidden_units=='0':
            self.hidden_units=[]
        else:
            self.hidden_units = [int(x) for x in self.hidden_units.split(",")]
        self.opt_type = kwargs.get("opt_type", "adam")
        self.use_batch_norm = kwargs.get('use_batch_norm',False)
        self.model_info = self._construct_model_info(
            input_size = train_dataset.img_feat_size,
            output_size = self.one_hot.vocab_size,
            hidden_units=self.hidden_units,
            learning_rate = self.learning_rate,
            optim_words = self.optim_words,
            use_batch_norm = self.use_batch_norm,
            wv_arr = self.w, weight_decay = self.weight_decay
        )
        self.test_one_hot = None
        self.test_w = None
        super(NegSamplingModel, self).__init__()

    def iter_batches(self, dataset, batch_size):
        for x, y in super(NegSamplingModel, self).iter_batches(dataset, batch_size):
            yield x, y
            
        # This will decay the learning rate every ten epochs. Hardcoded ten currently...
        if self.weight_decay:
            if self.epoch_num and self.epoch_num % 10 == 0:
                self.learning_rate *= self.weight_decay
        	logger.info('Learning rate dropped to {}'.format(self.learning_rate))
            self.epoch_num+=1
    
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

        if not self.optim_words:
            fetches = [self.model_info["optimizer"], self.model_info["loss"]]
            feed_dict = {
                self.model_info["input"]: img_feats,
                self.model_info["pos_ids"]: pos_ids,
                self.model_info["neg_ids"]: neg_ids
            }
        else:
            pvecs = np.zeros((pos_ids.shape[0], pos_ids.shape[1], self.w.shape[1]))
            nvecs = np.zeros((neg_ids.shape[0], neg_ids.shape[1], self.w.shape[1]))
            for i, ids in enumerate(pos_ids):
                pvecs[i] = self.w[ids]
            for i, ids in enumerate(neg_ids):
                nvecs[i] = self.w[ids]
            pvecs = pvecs.transpose((1, 0, 2))
            nvecs = nvecs.transpose((1, 0, 2))

            fetches = [self.model_info["optimizer"], self.model_info["loss"], self.model_info["prediction"]]
            feed_dict = {
                self.model_info["input"]: img_feats,
                self.model_info["pos_vecs"]: pvecs,
                self.model_info["neg_vecs"]: nvecs
            }
        
        if self.weight_decay:
            feed_dict[self.model_info['learning_rate']] = self.learning_rate

        return fetches, feed_dict

    def _updatewords(self, vpindex, vnindex, vin):
        for i, (vpi, vni) in enumerate(zip(vpindex, vnindex)):
            self.w[vpi] += self.joint_factor*self.learning_rate * np.outer(1 - sigmoid(self.w[vpi].dot(vin[i])), vin[i])
            self.w[vni] -= self.joint_factor*self.learning_rate * np.outer(sigmoid(self.w[vni].dot(vin[i])), vin[i])

    def fit(self, sess, fetches, feed_dict):
        fit_fetches = super(NegSamplingModel, self).fit(sess, fetches, feed_dict)
        if self.optim_words:
            if self.pos_ids is None or self.neg_ids is None:
                raise Exception("pos_ids or neg_ids is not set; cannot update word vectors. Did you run prep_fit()?")
            _, _, prediction = fit_fetches
            self._updatewords(self.pos_ids, self.neg_ids, prediction)
        return fit_fetches

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




