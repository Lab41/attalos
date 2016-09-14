import numpy as np
import tensorflow as tf
from scipy.special import expit as sigmoid


class NegSamplingModel(object):
    """
    Create a tensorflow graph that does regression to a target using a negative sampling loss function
    """
    def __init__(self, input_size,
                    w2v,
                    learning_rate=1.001,
                    hidden_units=[200,200],
                    optim_words=False,
                    use_batch_norm=True):
        self.model_info = dict()

         # Placeholders for data
        self.model_info['input'] = tf.placeholder(shape=(None, input_size), dtype=tf.float32)

        # Inputs to cost function
        self.learning_rate = learning_rate
        self.optim_words = optim_words
        if optim_words:
            self.w2v=w2v
            self.model_info['pos_vecs'] = tf.placeholder(dtype=tf.float32)
            self.model_info['neg_vecs'] = tf.placeholder(dtype=tf.float32)
            print "Optimization on GPU, word vectors stored separately"
        else:
            self.model_info['w2v'] = tf.Variable(w2v)
            w2vgraph=self.model_info['w2v'] 
            self.model_info['pos_ids'] = tf.placeholder(dtype=tf.int32)
            self.model_info['neg_ids'] = tf.placeholder(dtype=tf.int32)  
            self.model_info['pos_vecs'] = tf.transpose(tf.nn.embedding_lookup(w2vgraph,
                                                                              self.model_info['pos_ids']),
                                                                              perm=[1,0,2])
            self.model_info['neg_vecs'] = tf.transpose(tf.nn.embedding_lookup(w2vgraph,
                                                                              self.model_info['neg_ids']),
                                                                              perm=[1,0,2])

        # Construct fully connected layers
        layers = []
        layer = self.model_info['input']
        for i, hidden_size in enumerate(hidden_units[:-1]):
            layer = tf.contrib.layers.relu(layer, hidden_size)
            layers.append(layer)
            if use_batch_norm:
                layer = tf.contrib.layers.batch_norm(layer)
                layers.append(layer)

        # Output layer should always be linear
        layer = tf.contrib.layers.linear(layer, w2v.shape[1])
        layers.append(layer)

        self.model_info['layers'] = layers
        self.model_info['prediction'] = layer

        def meanlogsig(pred, truth):
            reduction_indicies = 2
            return tf.reduce_mean( tf.log( tf.sigmoid( tf.reduce_sum(pred*truth, reduction_indices=reduction_indicies))))
        
        pos_loss = meanlogsig(self.model_info['prediction'], self.model_info['pos_vecs'])
        neg_loss = meanlogsig(-self.model_info['prediction'], self.model_info['neg_vecs'])
        loss = -(pos_loss + neg_loss)
        self.model_info['loss'] = loss 

        # Initialization operations: check to see which ones actually get initialized (make sure "W2V")
        self.model_info['optimizer'] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        self.model_info['init_op'] = tf.initialize_all_variables()
        self.model_info['saver'] = tf.train.Saver()

    def initialize_model(self, sess):
        sess.run(self.model_info['init_op'])

    def predict(self, sess, x):
        return sess.run(self.model_info['prediction'], feed_dict={self.model_info['input']: x})

    def updatewords(self, vpindex, vnindex, vin):
        for i, (vpi, vni) in enumerate(zip(vpindex, vnindex)):
            self.w2v[vpi]+=self.learning_rate*np.outer(1 - sigmoid(self.w2v[vpi].dot(vin[i])),vin[i])
            self.w2v[vni]-=self.learning_rate*np.outer(sigmoid(self.w2v[vni].dot(vin[i])),vin[i])                                                            
    def fit(self, sess, x, y, **kwargs):

        # If you're not optimizing for the words
        if not self.optim_words:
            _, loss = sess.run([self.model_info['optimizer'], self.model_info['loss']],
                               feed_dict={ self.model_info['input']: x,
                                           self.model_info['pos_ids']: y,
                                           self.model_info['neg_ids']: kwargs['neg_word_ids']
                                         })

        # If you are, then you'll need to get the vectors offline
        else:
            neg_ids = kwargs['neg_word_ids']
            pvecs = np.zeros((y.shape[0], y.shape[1], self.w2v.shape[1]))
            nvecs = np.zeros((neg_ids.shape[0], neg_ids.shape[1], self.w2v.shape[1]))
            for i, ids in enumerate(y):
                pvecs[i] = self.w2v[ids]
            for i, ids in enumerate(neg_ids):
                nvecs[i] = self.w2v[ids]
            pvecs = pvecs.transpose((1,0,2))
            nvecs = nvecs.transpose((1,0,2))
            _, loss, preds = sess.run([self.model_info['optimizer'], self.model_info['loss'], self.model_info['prediction']],
                                       feed_dict={ self.model_info['input']: x,
                                                   self.model_info['pos_vecs']: pvecs,
                                                   self.model_info['neg_vecs']: nvecs
                                                 })
            self.updatewords(y, kwargs['neg_word_ids'], preds)

        return loss

    def save(self, sess, model_output_path):
        self.model_info['saver'].save(sess, model_output_path)

    def load(self, sess, model_input_path):
        self.model_info['saver'].restore(sess, model_input_path)
