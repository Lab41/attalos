import os
from enum import Enum
import gzip
import time

import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
import tensorflow as tf

# Attalos Imports
import attalos.util.log.log as l
from attalos.dataset.dataset import Dataset
from attalos.evaluation.evaluation import Evaluation

# Local models
from mse import MSEModel
from negsampling import NegSamplingModel


# Setup global objects
logger = l.getLogger(__name__)

class ModelTypes(Enum):
    mse = 1
    negsampling = 2


def evaluate_regressor(sess, model, val_image_feats, val_one_hot, wordmatrix, k=5, verbose=False):
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
    val_pred = model.predict(sess, val_image_feats)

    predictions_one_hot = np.zeros(val_one_hot.shape)
    for i in range(val_pred.shape[0]):
        normalized_val = val_pred[i, :]/np.linalg.norm(val_pred[i, :])
        # np.dot(wordmatrix, normalized_val) gets the similarity between the two vectors
        # argpartition gets the topk (where k=5)
        indices = np.argpartition(np.dot(wordmatrix, normalized_val), -1*k)[-1*k:]
        for index in indices:
            predictions_one_hot[i, index] = 1

    evaluator = Evaluation(val_one_hot, predictions_one_hot)

    return evaluator


def create_wordmatrix(w2v_model, dataset=None):
    """
    Take a w2v dictionary and return matrix/index lookup
    Args:
        w2vmodel: Dictionary where keys are words and values are word vectors
        dataset: If specified limits tags in matrix to tags in dataset

    Returns:
        w2ind: Mapping of word to index
        wordmatrix: Numpy matrix of word vectors
    """
    dataset_tags = None
    if dataset:
        dataset_tags = set()
        for tags in dataset.text_feats.values():
            dataset_tags.update(tags)
        num_tags_in_output = len(dataset_tags.intersection(w2v_model.keys()))
    else:
        num_tags_in_output = len(w2v_model)

    # Create word vector matrix to allow for embedding lookup
    w2ind = {}
    wordmatrix = np.zeros((num_tags_in_output, len(w2v_model[w2v_model.keys()[0]])), dtype=np.float32)
    i =0
    for word in w2v_model:
        if dataset_tags is None or word in dataset_tags:
            w2ind[word] = i
            wordmatrix[i, :] = w2v_model[word]
            i += 1
    return w2ind, wordmatrix


def dataset_to_onehot(dataset, w2ind):
    """
    Take a dataset and prepare it for convient evaluation
    Args:
        dataset: attalos.dataset.dataset object
        w2ind: a dictionary like object mapping words to their index

    Returns:
        img_feats: A matrix of image feautres
        one_hot: A sparse matrix of one hot tags

    """
    image_feat, tags = dataset.get_index(0)

    image_feats = np.zeros((dataset.num_images, image_feat.shape[0]))
    one_hot = dok_matrix((dataset.num_images, len(w2ind)), dtype=np.int32)
    # Extract features and place in numpy matrix
    for i in dataset:
        image_feat, tags = dataset[i]
        image_feats[i, :] = image_feat
        for tag in tags:
            if tag in w2ind:
                one_hot[i, w2ind[tag]] = 1

    return image_feats, csr_matrix(one_hot)


def train_model(train_dataset,
                test_dataset,
                w2v_model,
                batch_size=128,
                num_epochs=200,
                learning_rate=1.001,
                network_size=[200,200],
                model_input_path = None,
                model_output_path = None,
                verbose=True,
                model_type=ModelTypes.negsampling):
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
    # Get validation data
    #  Extract features from first image
    image_feats, tags = test_dataset.get_index(0)
    # Get shape and initialize numpy matrix
    image_feat_size = image_feats.shape[0]


    # Turn w2v dictionary into a matrix
    w2ind, word_matrix = create_wordmatrix(w2v_model)
    val_w2ind, val_word_matrix = create_wordmatrix(w2v_model, test_dataset)

    # Precompute onehot representation for evaluation
    val_image_feats, val_one_hot = dataset_to_onehot(test_dataset, val_w2ind)


    # Setup data structures for negative sampling
    if model_type == ModelTypes.negsampling:
        word_counts = np.zeros(word_matrix.shape[0])
        for item_id in train_dataset:
            _, tags = train_dataset[item_id]
            for tag in tags:
                if tag in w2ind:
                    word_counts[w2ind[tag]] += 1
        labelpdf = word_counts / word_counts.sum()
        vocabsize = word_matrix.shape[0]
        def negsamp(ignored_inds, num2samp):
            # Negative sampler that takes in indicies

            # Create new probability vector excluding positive samples
            nlabelpdf = np.copy(labelpdf)
            nlabelpdf[ignored_inds] = 0
            nlabelpdf /= nlabelpdf.sum()

            return np.random.choice(vocabsize, size=num2samp, p=nlabelpdf)

    # Time to start building our graph
    with tf.Graph().as_default():
        # Build regressor
        if model_type == ModelTypes.mse:
            logger.info('Building regressor with mean square error loss')
            model = MSEModel(image_feat_size,
                                         word_matrix,
                                        learning_rate=learning_rate,
                                        hidden_units=network_size,
                                        use_batch_norm=True)
        elif model_type == ModelTypes.negsampling:
            logger.info('Building regressor with negative sampling loss')
            model = NegSamplingModel(image_feat_size,
                                        word_matrix,
                                        learning_rate=learning_rate,
                                        hidden_units=network_size,
                                        use_batch_norm=True)

        # Allocate GPU memory as needed (vs. allocating all the memory)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            # Initialize model
            model.initialize_model(sess)

            # Optionally restore saved model
            if model_input_path:
                model.load(sess, model_input_path)

            NUM_POSTIVE_EXAMPLES = 5
            NUM_NEGATIVE_EXAMPLES = 10
            # Reuse space for each iteration
            pos_word_ids = np.ones((batch_size, NUM_POSTIVE_EXAMPLES), dtype=np.int32)
            neg_word_ids = np.ones((batch_size, NUM_NEGATIVE_EXAMPLES), dtype=np.int32)
            performance = []
            for epoch in range(num_epochs):
                batch_time_total = 0
                run_time_total = 0

                loss = None
                for batch in range(int(train_dataset.num_images/batch_size)):
                    batch_time = time.time()
                    # Get raw data
                    image_feats, text_tags = train_dataset.get_next_batch(batch_size)

                    # Generate positive examples
                    pos_word_ids.fill(-1)
                    for i, tags in enumerate(text_tags):
                        j = 0
                        for tag in tags:
                            if tag in w2ind and j < NUM_POSTIVE_EXAMPLES:
                                pos_word_ids[i, j] = w2ind[tag]
                                j += 1

                    if model_type == ModelTypes.negsampling:
                        neg_word_ids.fill(-1)
                        for i in range(neg_word_ids.shape[0]):
                            neg_word_ids[i] = negsamp(pos_word_ids, NUM_NEGATIVE_EXAMPLES)

                    batch_time = time.time() - batch_time
                    batch_time_total += batch_time

                    run_time = time.time()
                    if model_type == ModelTypes.mse:
                        loss = model.fit(sess, image_feats, pos_word_ids)
                    elif model_type == ModelTypes.negsampling:
                        loss = model.fit(sess, image_feats,pos_word_ids, neg_word_ids=neg_word_ids)
                    run_time = time.time() - run_time
                    run_time_total += run_time

                if verbose:
                    eval_time = time.time()
                    evaluator = evaluate_regressor(sess, model, val_image_feats, val_one_hot, val_word_matrix, verbose=False)
                    performance.append(evaluator.evaluate())
                    eval_time = time.time() - eval_time
                    # Evaluate accuracy
                    #print('Epoch {}: Loss: {} Timing: {} {} {}'.format(epoch, loss, batch_time_total, run_time_total, eval_time))
                    logger.debug('Epoch {}: Loss: {} Perf: {} {} {}'.format(epoch, loss, *performance[-1]))

            if model_output_path:
                model.save(sess, model_output_path)

            return performance


args = None
def convert_args_and_call_model():
    global args
    train_dataset = Dataset(args.image_feature_file_train, args.text_feature_file_train, load_image_feats_in_mem=args.in_memory)
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
    else:
        raise IOError('No word vector file specified')

    for i, line in enumerate(input_file):
        first_word = line[:line.find(' ')]
        if first_word in dataset_tags:
            line = line.strip().split(' ')
            w2v_vector = np.array([float(j) for j in line[1:]])
            # Normalize vector before storing
            w2v_lookup[line[0]] = w2v_vector

    if args.model_type == 'mse':
        model_type = ModelTypes.mse
    elif args.model_type == 'negsampling':
        model_type = ModelTypes.negsampling

    return train_model(train_dataset,
                test_dataset,
                w2v_lookup,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                network_size=map(int, args.network.split(',')),
                num_epochs=args.epochs,
                model_input_path=args.model_input_path,
                model_output_path=args.model_output_path,
                model_type=model_type)


def main():
    import argparse

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
                        default=.001,
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
    parser.add_argument("--model_type",
                        type=str,
                        default="mse",
                        choices=['mse', 'negsampling'],
                        help="Loss function to use for training")
    parser.add_argument("--in_memory",
                        action='store_true',
                        default="store_false",
                        help="Load training image features into memory for faster training")
    parser.add_argument("--model_input_path",
                        type=str,
                        default=None,
                        help="Model input path (to continue training)")
    parser.add_argument("--model_output_path",
                        type=str,
                        default=None,
                        help="Model output path (to save training)")

    global args
    args = parser.parse_args()

    try:
        # Sacred Imports
        from sacred import Experiment
        from sacred.observers import MongoObserver

        from sacred.initialize import Scaffold

        # Monkey patch to avoid having to declare all our variables
        def noop(item):
            pass
        Scaffold._warn_about_suspicious_changes = noop

        ex = Experiment('Regress2sum')
        ex.observers.append(MongoObserver.create(url=os.environ['MONGO_DB_URI'],
                                             db_name='attalos_experiment'))
        ex.main(lambda: convert_args_and_call_model())
        ex.run(config_updates=args.__dict__)
    except ImportError:
        # We don't have sacred, just run the script
        convert_args_and_call_model()


if __name__ == '__main__':
    main()

