import time
import h5py
import numpy as np
import argparse
import tensorflow as tf


def tags_2_vec(tags, w2v_model=None):
    if len(tags) == 0:
        raise ValueError('No Tags')
    else:
        word_vectors = []
        good_tags = [tag.lower() for tag in tags if tag.lower() in w2v_model]
        if len(good_tags) == 0:
            raise KeyError('Tags not found: {}'.format(tags))
            return np.zeros(300)

        for tag in good_tags:
            word_vector = w2v_model[tag]
            word_vectors.append(word_vector)
        output = np.sum(word_vectors, axis=0)
        return output / np.linalg.norm(output)


def construct_model(input_size,
                    output_size,
                    learning_rate=0.001,
                    hidden_units=[200,200],
                    use_batch_norm=True):
    model_info = dict()

    # Placeholders for data
    model_info['input'] = tf.placeholder(shape=(None, input_size), dtype=tf.float32)
    model_info['y_pos'] = tf.placeholder(shape=(None, output_size), dtype=tf.float32)

    layers = []
    for i, hidden_size in enumerate(hidden_units):
        if i == 0:
            layer = tf.contrib.layers.relu(model_info['input'], hidden_size)
        else:
            layer = tf.contrib.layers.relu(layer, hidden_size)
        layers.append(layer)
        if use_batch_norm:
            layer = tf.contrib.layers.batch_norm(layer)
            layers.append(layer)

    model_info['layers'] = layers
    model_info['prediction'] = layer

    # Negative sampling
    # Generate negative_sample_count placeholders
    neg_example = tf.placeholder(shape=(None, output_size), dtype=tf.float32)
    model_info['y_neg'] = neg_example

    def tf_dot(x, y):
            return tf.reduce_sum(tf.mul(x, y), 1)

    # Negative Sampling
    sig_pos = tf.sigmoid(tf.scalar_mul(1.0, tf_dot(model_info['y_pos'], model_info['prediction'])))
    sig_neg = tf.sigmoid(tf.scalar_mul(-1.0, tf_dot(neg_example, model_info['prediction'])))
    loss = - tf.reduce_mean(tf.log(tf.clip_by_value(sig_pos, 1e-10, 1.0))) \
                - tf.reduce_mean(tf.log(tf.clip_by_value(sig_neg, 1e-10, 1.0)))

    model_info['loss'] = loss
    model_info['optimizer'] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return model_info


def train_model(curated_file,
                epochs=10,
                batch_size=128,
                hidden_units=[200,200],
                learning_rate=.001,
                model_input_path = None,
                model_output_path = None):

    num_items = curated_file['img_feats'].shape[0]

    image_feat_size = curated_file['img_feats'].shape[1]
    word_feat_size = curated_file['word_feats'].shape[1]

    losses = []  # track losses
    with tf.Graph().as_default():
        # Build Model
        model = construct_model(learning_rate=learning_rate,
                                input_size=image_feat_size,
                                output_size=word_feat_size,
                                hidden_units=hidden_units)
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()

        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        with tf.Session(config=config) as sess:
            sess.run(init)
            if model_input_path:
                saver.restore(sess, model_input_path)
            print('Starting Training')
            total_time = time.time()
            for epoch in range(epochs):
                num_batches_per_epoch = int(num_items/batch_size)
                for batch in range(num_batches_per_epoch):
                    feed_dict = {}

                    image_feats = curated_file['img_feats'][batch*batch_size:(batch+1)*batch_size, :]
                    pos_word_feats = curated_file['word_feats'][batch*batch_size:(batch+1)*batch_size, :]
                    neg_word_feats = curated_file['word_feats'][batch*batch_size+1+epoch:(batch+1)*batch_size+1+epoch, :]

                    feed_dict[model['input']] = image_feats
                    feed_dict[model['y_pos']] = pos_word_feats
                    feed_dict[model['y_neg']] = neg_word_feats

                    start_time = time.time()
                    sess.run(model['optimizer'], feed_dict=feed_dict)
                    if batch%100 == 0:
                        if model_output_path:
                            saver.save(sess, model_output_path)
                        loss, =  sess.run([model['loss']], feed_dict=feed_dict)
                        losses.append(loss)
                        print('Completed batch {} of {} with {} images (loss: {}): {} {}'.format(batch,
                                                                   (int(num_items/batch_size)),
                                                                            image_feats.shape[0],
                                                                    loss,
                                                                  time.time() - total_time,
                                                                  time.time()-start_time))
                        total_time = time.time()
    return losses


def main():
    parser = argparse.ArgumentParser(description='Two layer linear regression')
    parser.add_argument("curated_file",
                        type=str,
                        help="Feature file (hdf5 with img_feats and word_feats objects")


    # Optional Args
    parser.add_argument("--model_input_path",
                        type=str,
                        default=None,
                        help="Model input path (to continue training)")
    parser.add_argument("--model_output_path",
                        type=str,
                        default=None,
                        help="Model output path (to save training)")
    parser.add_argument("--network",
                        type=str,
                        default='4096,400,200',
                        help="Define number of units in each layer of your fully connected network")
    parser.add_argument("--learning_rate",
                        type=float,
                        default=.001,
                        help="Learning Rate")
    parser.add_argument("--epochs",
                        type=int,
                        default=10,
                        help="Number of epochs to run for")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128*8,
                        help="Batch size to use for training")

    args = parser.parse_args()
    print('Loading HDF5')
    # Validate that input hdf5 file has expected keys
    curated_file = h5py.File(args.curated_file)
    if 'img_feats' not in curated_file.keys() or 'word_feats'not in curated_file.keys():
        raise KeyError('Expected img_feats and word_feats keys but found {}'.format(curated_file.keys()))

    # Validate that network definition starts at image size and ends at word vector size
    hidden_units = map(int, args.network.split(','))
    image_feat_size = curated_file['img_feats'].shape[1]
    word_feat_size = curated_file['word_feats'].shape[1]
    if hidden_units[0] != image_feat_size:
        raise ValueError('Expected first layer of network to match image feature size')

    if hidden_units[-1] != word_feat_size:
        raise ValueError('Expected las layer of network to match word vector size')

    # Train network
    print('Calling Train')
    train_model(curated_file,
                epochs=args.epochs,
                batch_size=args.batch_size,
                hidden_units=hidden_units,
                learning_rate=args.learning_rate,
                model_input_path=args.model_input_path,
                model_output_path=args.model_output_path)

if __name__ == '__main__':
    main()
