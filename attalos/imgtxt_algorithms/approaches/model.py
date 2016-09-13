import tensorflow as tf

class MSEModel(object):
    """
    Create a tensorflow graph that does regression to a target using a mean square error loss function
    """
    def __init__(self, input_size,
                    w2v,
                    learning_rate=1.001,
                    hidden_units=[200,200],
                    use_batch_norm=True):
        self.model_info = dict()
         # Placeholders for data
        self.model_info['input'] = tf.placeholder(shape=(None, input_size), dtype=tf.float32)
        self.model_info['pos_ids'] = tf.placeholder(dtype=tf.int32)

        self.model_info['w2v'] = tf.Variable(w2v)
        self.model_info['y_truth']=tf.reduce_sum(tf.nn.embedding_lookup(self.model_info['w2v'],
                                                                        self.model_info['pos_ids']),1)

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

        loss = tf.reduce_sum(tf.square(self.model_info['prediction']-self.model_info['y_truth']))
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
                               self.model_info['pos_ids']: y
                           })
        return loss

    def save(self, sess, model_output_path):
        self.model_info['saver'].save(sess, model_output_path)

    def load(self, sess, model_input_path):
        self.model_info['saver'].restore(sess, model_input_path)
