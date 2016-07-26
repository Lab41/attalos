import argparse
import gzip
import numpy as np
import tensorflow as tf

from attalos.dataset.dataset import Dataset


def tags_2_vec(tags, w2v_model=None):
    if len(tags) == 0:
        return np.zeros(300)
    else:
        return w2v_model[tags[0]]  # TODO: Only taking first tag right now...


def train_model(train_dataset, test_dataset, w2v_model, batch_size=128):
    # Get a single batch to allow us to get feature vector sizes
    image_feats, text_tags = train_dataset.get_next_batch(5)
    word_feats = [tags_2_vec(tags, w2v_model) for tags in text_tags]
    img_feat_size = image_feats.shape[1]
    w2v_feat_size = word_feats[0].shape[0]

    num_items = train_dataset.num_images  # TODO: Don't direct access member variables

    # Allocate GPU memory as needed (vs. allocating all the memory)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Graph().as_default():
        # Placeholders for data
        X = tf.placeholder(shape=(None, img_feat_size), dtype=tf.float32)
        Y = tf.placeholder(shape=(None, w2v_feat_size), dtype=tf.float32)

        # Two layer network
        fc1 = tf.contrib.layers.fully_connected(X, 300, tf.nn.relu)
        fc2 = tf.contrib.layers.fully_connected(fc1, 300, tf.sigmoid)

        # Mean Square error
        loss = tf.reduce_sum(tf.square(Y-fc2))

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)

        init = tf.initialize_all_variables()
        with tf.Session(config=config) as sess:
            sess.run(init)
            for epoch in range(200):
                for batch in range(int(num_items/batch_size)):
                    image_feats, text_tags = train_dataset.get_next_batch(batch_size)
                    word_feats = [tags_2_vec(tags, w2v_model) for tags in text_tags]
                    sess.run(optimizer, feed_dict={X: image_feats, Y: word_feats})
                cost = sess.run(loss, feed_dict={X: image_feats, Y: word_feats})
                print('Epoch:', epoch, 'Cost:', cost)


def main():
    parser = argparse.ArgumentParser(description='Two layer linear regression')
    parser.add_argument('--image_feature_file',
                      dest='image_feature_file',
                      type=str,
                      help='Image Feature file (train)')
    parser.add_argument('--text_feature_file',
                      dest='text_feature_file',
                      type=str,
                      help='Text Feature file')

    args = parser.parse_args()
    train_dataset = Dataset(args.image_feature_file, args.text_feature_file)

    dataset_tags = set()
    for tags in train_dataset.text_feats.values():
        dataset_tags.update(tags)

    # Read w2vec
    w2v_lookup = {}
    for i, line in enumerate(gzip.open('/local_data/yonas/glove.42B.300d.txt.gz')):
        first_word = line[:line.find(' ')]
        if first_word in dataset_tags:
            line = line.strip().split(' ')
            w2v_lookup[line[0]] = np.array([ float(j) for j in line[1:]])

    train_model(train_dataset, None, w2v_lookup)  # TODO: Also read testing data

if __name__ == '__main__':
    main()

