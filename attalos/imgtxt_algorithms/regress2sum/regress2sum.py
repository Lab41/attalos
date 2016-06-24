import argparse
import gzip
import numpy as np
import tensorflow as tf

from attalos.dataset.dataset import Dataset
from MinHeap import get_top_k


def tags_2_vec(tags, w2v_model=None):
    if len(tags) == 0:
        return np.zeros(300)
    else:
        # return w2v_model[tags[0]]  # Only take 1st tag
        # return w2v_model[tags[np.random.randint(0, len(tags), 1)]]  # Random tag
        return np.sum([w2v_model[tag] for tag in tags], axis=0)/len(tags)  # Sum of tags


def train_model(train_dataset,
                test_dataset,
                w2v_model,
                learning_rate=.001,
                batch_size=128,
                num_epochs=200):
    # Get a single batch to allow us to get feature vector sizes
    image_feats, text_tags = train_dataset.get_next_batch(5)
    word_feats = [tags_2_vec(tags, w2v_model) for tags in text_tags]
    img_feat_size = image_feats.shape[1]
    w2v_feat_size = word_feats[0].shape[0]
    num_items = train_dataset.num_images

    # Get validation data
    val_image_feats, val_text_tags = test_dataset.get_next_batch(batch_size*10)
    val_word_feats = [tags_2_vec(tags, w2v_model) for tags in val_text_tags]
    val_batch_size = val_image_feats.shape[0]

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
        fc3 = tf.contrib.layers.fully_connected(fc2, 300, tf.sigmoid)

        # Mean Square error
        loss = tf.reduce_sum(tf.square(Y-fc2))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        init = tf.initialize_all_variables()
        with tf.Session(config=config) as sess:
            sess.run(init)
            for epoch in range(num_epochs):
                for batch in range(int(num_items/batch_size)):
                    image_feats, text_tags = train_dataset.get_next_batch(batch_size)
                    word_feats = [tags_2_vec(tags, w2v_model) for tags in text_tags]
                    sess.run(optimizer, feed_dict={X: image_feats, Y: word_feats})

                cost, output_values = sess.run([loss, fc3], feed_dict={X: val_image_feats, Y: val_word_feats})
                # Get most likely word and check to see if it is in set of tags for training data
                # TODO: Move to a more reasonable metric
                num_correct = 0
                for i in range(output_values.shape[0]):
                    topk = get_top_k(output_values[i, :], w2v_model, 5)
                    if topk[0] in val_text_tags[i]:
                        num_correct += 1
                print('Epoch:', epoch, 'Cost:', cost, 'Percent Correct:', 100.0*num_correct/val_batch_size)


def main():
    import os
    parser = argparse.ArgumentParser(description='Two layer linear regression')
    parser.add_argument("image_feature_file_train",
                        type=str,
                        help="Image Feature file for the training set")
    parser.add_argument("text_feature_file_train",
                        type=str,
                        help="Text Feature file for the training set")
    parser.add_argument("image_feature_file_test",
                        type=str,
                        help="Image Feature file for the test set")
    parser.add_argument("text_feature_file_test",
                        type=str,
                        help="Text Feature file for the test set")
    parser.add_argument("word_vector_file",
                        type=str,
                        help="Text file containing the word vectors")

    # Optional Args
    parser.add_argument("--learning_rate",
                        type=float,
                        default=.05,
                        help="Learning Rate")
    parser.add_argument("--epochs",
                        type=int,
                        default=200,
                        help="Number of epochs to run for")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch size to use for training")

    args = parser.parse_args()
    train_dataset = Dataset(args.image_feature_file_train, args.text_feature_file_train)
    test_dataset = Dataset(args.image_feature_file_test, args.text_feature_file_test)

    # Get the full vocab so we can extract only the word vectors we care about
    dataset_tags = set()
    for dataset in [train_dataset, test_dataset]:
        for tags in dataset.text_feats.values():
            dataset_tags.update(tags)

    # Read w2vec
    w2v_lookup = {}
    if os.path.exists(args.word_vector_file):
        if args.word_vector_file.endswith('.gz'):
            input_file = gzip.open(args.word_vector_file)
        else:
            input_file = open(args.word_vector_file)
    for i, line in enumerate(input_file):
        first_word = line[:line.find(' ')]
        if first_word in dataset_tags:
            line = line.strip().split(' ')
            w2v_lookup[line[0]] = np.array([float(j) for j in line[1:]])

    train_model(train_dataset,
                test_dataset,
                w2v_lookup,
                learning_rate=args.learning_rate,
                batch_size=args.batch_size,
                num_epochs=args.epochs)

if __name__ == '__main__':
    main()

