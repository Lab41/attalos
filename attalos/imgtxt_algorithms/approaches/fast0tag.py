import tensorflow as tf

class FastZeroTagModel(object):
    """
    Create a tensorflow graph that finds the principal direction of the target word embeddings 
    (with negative sampling), using the loss function from "Fast Zero-Shot Image Tagging".
    """
    def __init__(self, input_size,
                    w2v,
                    learning_rate=1e-5,
                    hidden_units=[4096, 2048],
                    use_batch_norm=True):
        self.model_info = dict()
         # Placeholders for data
        self.model_info['input'] = tf.placeholder(shape=(None, input_size), dtype=tf.float32)
        self.model_info['pos_ids'] = tf.placeholder(dtype=tf.int32)
        self.model_info['neg_ids'] = tf.placeholder(dtype=tf.int32)
        
        self.model_info['y_truth'] = tf.transpose(tf.nn.embedding_lookup(w2v,self.model_info['pos_ids']), perm=[1,0,2])
        self.model_info['y_neg'] = tf.transpose(tf.nn.embedding_lookup(w2v,self.model_info['neg_ids']), perm=[1,0,2])

        # Construct fully connected layers
        layers = []
        for i, hidden_size in enumerate(hidden_units[:-1]):
            if i == 0:
                layer = tf.contrib.layers.relu(self.model_info['input'], hidden_size)
            else:
                layer = tf.contrib.layers.relu(layer, hidden_size)

            layers.append(layer)
            if use_batch_norm:
                layer = tf.contrib.layers.batch_norm(layer)
                layers.append(layer)

        # Output layer should always be linear
        layer = tf.contrib.layers.linear(layer, hidden_units[-1])
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

        loss = fztloss(self.model_info['prediction'], self.model_info['y_truth'], self.model_info['y_neg'])
        
        self.model_info['loss'] = loss
        self.model_info['optimizer'] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        self.model_info['init_op'] = tf.initialize_all_variables()
        self.model_info['saver'] = tf.train.Saver()

    def initialize_model(self, sess):
        sess.run(self.model_info['init_op'])

    def predict(self, sess, x):
        return sess.run(self.model_info['prediction'], feed_dict={self.model_info['input']: x})

    def fit(self, sess, x, y, **kwargs):
        _, loss = sess.run([self.model_info['optimizer'], self.model_info['loss']],
                           feed_dict={
                               self.model_info['input']: x,
                               self.model_info['pos_ids']: y,
                               self.model_info['neg_ids']: kwargs['neg_word_ids']
                           })
        return loss

    def save(self, sess, model_output_path):
        self.model_info['saver'].save(sess, model_output_path)

    def load(self, sess, model_input_path):
        self.model_info['saver'].restore(sess, model_input_path)