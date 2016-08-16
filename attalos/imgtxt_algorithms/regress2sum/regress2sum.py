import argparse
import gzip
import numpy as np
import tensorflow as tf

from attalos.dataset.dataset import Dataset
from attalos.evaluation.evaluation import Evaluation


def tags_2_vec(tags, w2v_model=None):
    """
    Takes a list of text tags and returns the normalized sum of the word vectors

    Args:
        tags: A iterable of text tags
        w2v_model: a dictionary like object where the keys are words and the values are word vectors

    Returns:
        Normalized sum of the word vectors
    """
    if len(tags) == 0 or len([tag for tag in tags if tag in w2v_model]) == 0:
        return np.zeros(200)
    else:
        output = np.sum([w2v_model[tag] for tag in tags if tag in w2v_model], axis=0)
        return output / np.linalg.norm(output)


def construct_model(input_size,
                    output_size,
                    learning_rate=1.001,
                    hidden_units=[200,200],
                    use_batch_norm=True):
    model_info = dict()

    # Placeholders for data
    model_info['input'] = tf.placeholder(shape=(None, input_size), dtype=tf.float32)
    model_info['y_truth'] = tf.placeholder(shape=(None, output_size), dtype=tf.float32)

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

    loss = tf.reduce_sum(tf.square(model_info['prediction']-model_info['y_truth']))
    model_info['loss'] = loss
    model_info['optimizer'] = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    return model_info


def evaluate_regressor(sess, model_info, val_image_feats, val_text_tags, w2v_model, k=5, verbose=False):
    """
    Takes a regressor and returns the precision/recall on the test data
    Args:
        sess: A tensorflow session
        model_info: A dictionary containing tensorflow layers (specifically input and prediction)
        val_image_feats: Image features to test performance on
        val_text_tags: Text Tags to test performance on
        w2v_model: a dictionary like object where the keys are words and the values are word vectors
        k: Top number of items to retrieve to test precision/recall on
        verbose: Verbose output or not

    Returns:
        evaluator: A attalos.evaluation.evaluation.Evaluation object
    """
    val_pred = sess.run(model_info['prediction'], feed_dict={model_info['input']:val_image_feats})

    w2ind = {}
    reverse_w2v_model = {}
    wordmatrix = np.zeros((len(w2v_model), len(w2v_model[w2v_model.keys()[0]])))
    for i, word in enumerate(w2v_model):
        w2ind[word] = i
        wordmatrix[i, :] = w2v_model[word]
        reverse_w2v_model[i] = word

    ground_truth_one_hot = np.zeros((len(val_text_tags), len(w2v_model)))
    num_skipped = 0
    total = 0
    skipped = set()
    for i, tags in enumerate(val_text_tags):
        for tag in tags:
            try:
                total += 1
                ground_truth_one_hot[i, w2ind[tag]] = 1
            except KeyError:
                skipped.add(tag)
                num_skipped +=1

    if verbose:
        print('Skipped {} of {} total'.format(num_skipped, total))

    predictions_one_hot = np.zeros((len(val_text_tags), len(w2v_model)))
    for i in range(val_pred.shape[0]):
        normalized_val = val_pred[i, :]/np.linalg.norm(val_pred[i, :])
        # np.dot(wordmatrix, normalized_val) gets the similarity between the two vectors
        # argpartition gets the topk (where k=5)
        indices = np.argpartition(np.dot(wordmatrix,normalized_val), -1*k)[-1*k:]
        for index in indices:
            predictions_one_hot[i, index] = 1

    evaluator = Evaluation(ground_truth_one_hot, predictions_one_hot)

    return evaluator


def train_model(train_dataset,
                test_dataset,
                w2v_model,
                batch_size=128,
                num_epochs=200,
                learning_rate=1.001,
                network_size=[200,200],
                model_input_path = None,
                model_output_path = None,
                verbose=True):
    """
    Train a regression model to map image features into the word vector space
    Args:
        train_dataset: Training attalos.dataset.dataset object
        test_dataset: Test attalos.dataset.dataset object
        w2v_model: A dictionary like object where the keys are words and the values are word vectors
        batch_size: Batch size to use for training
        num_epochs: Number of epochs to train for
        learning_rate: The learning rate for the network
        network_size: A list defining the size of each layer of the neural network
        model_input_path: Path to a file containing initial weights
        model_output_path: Path to save final weights
        verbose: Amounto fdebug information to output
    Returns:
    """
    num_items = train_dataset.num_images

    # Get validation data
    #  Extract features from first image
    image_feats, tags = test_dataset.get_index(0)
    # Get shape and initialize numpy matrix
    image_feat_size = image_feats.shape[0]
    text_feat_size = w2v_model[tags[0]].shape[0]
    val_image_feats = np.zeros((test_dataset.num_images, image_feat_size))
    val_text_tags = []
    # Extract features and place in numpy matrix
    for i in test_dataset:
        image_feats, tags = test_dataset[i]
        val_image_feats[i, :] = image_feats/np.linalg.norm(image_feats)
        val_text_tags.append(tags)

    with tf.Graph().as_default():


        # Build regressor
        model_info = construct_model(image_feat_size,
                                        text_feat_size,
                                        learning_rate=learning_rate,
                                        hidden_units=network_size,
                                        use_batch_norm=True)
        init = tf.initialize_all_variables()
        saver = tf.train.Saver()
        # Allocate GPU memory as needed (vs. allocating all the memory)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            sess.run(init)
            if model_input_path:
                saver.restore(sess, model_input_path)

            for epoch in range(num_epochs):
                for batch in range(int(num_items/batch_size)):
                    image_feats, text_tags = train_dataset.get_next_batch(batch_size)
                    for i in range(batch_size):
                        image_feats[i, :] = image_feats[i, :]/ np.linalg.norm(image_feats[i, :])
                    word_feats = np.array([tags_2_vec(tags, w2v_model) for tags in text_tags])

                    feed_dict = {}
                    feed_dict[model_info['input']] = image_feats
                    feed_dict[model_info['y_truth']] = word_feats
                    sess.run(model_info['optimizer'], feed_dict=feed_dict)

                if verbose:
                    evaluator = evaluate_regressor(sess, model_info, val_image_feats, val_text_tags, w2v_model, verbose=verbose)
                    # Evaluate accuracy
                    print('Epoch: {}'.format(epoch))
                    evaluator.evaluate()

            if model_output_path:
                saver.save(sess, model_output_path)

            return


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
                        default=1.001,
                        help="Learning Rate")
    parser.add_argument("--epochs",
                        type=int,
                        default=200,
                        help="Number of epochs to run for")
    parser.add_argument("--batch_size",
                        type=int,
                        default=128,
                        help="Batch size to use for training")
    parser.add_argument("--network",
                        type=str,
                        default="200,200",
                        help="Define a neural network as comma separated layer sizes")
    parser.add_argument("--model_input_path",
                        type=str,
                        default=None,
                        help="Model input path (to continue training)")
    parser.add_argument("--model_output_path",
                        type=str,
                        default=None,
                        help="Model output path (to save training)")

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
            w2v_vector = np.array([float(j) for j in line[1:]])
            # Normalize vector before storing
            w2v_lookup[line[0]] = w2v_vector / np.linalg.norm(w2v_vector)

    train_model(train_dataset,
                test_dataset,
                w2v_lookup,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                network_size=map(int, args.network.split(',')),
                num_epochs=args.epochs,
                model_input_path=args.model_input_path,
                model_output_path=args.model_output_path)

if __name__ == '__main__':
    main()

