import argparse
from enum import Enum
import gzip
import numpy as np
from scipy.sparse import dok_matrix, csr_matrix
import tensorflow as tf
import time

from attalos.dataset.dataset import Dataset
from attalos.evaluation.evaluation import Evaluation

# Local models
from mse import MSEModel
from negsampling import NegSamplingModel

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
        val_image_feats[i, :] = image_feats
        val_text_tags.append(tags)

    # Create word vector matrix to allow for embedding lookup
    w2ind = {}
    wordmatrix = np.zeros((len(w2v_model), len(w2v_model[w2v_model.keys()[0]])), dtype=np.float32)
    for i, word in enumerate(w2v_model):
        w2ind[word] = i
        wordmatrix[i, :] = w2v_model[word]

    # Precompute onehot vectors to speed up evalutation
    val_one_hot = dok_matrix((len(val_text_tags), len(w2v_model)), dtype=np.int32)
    num_skipped = 0
    skipped = set()
    for i, tags in enumerate(val_text_tags):
        for tag in tags:
            try:
                val_one_hot[i, w2ind[tag]] = 1
            except KeyError:
                skipped.add(tag)
                num_skipped +=1
    # Convert to more efficient structure
    val_one_hot = csr_matrix(val_one_hot)


    # Setup data structures for negative sampling
    if model_type == ModelTypes.negsampling:
        word_counts = np.zeros(wordmatrix.shape[0])
        for item_id in train_dataset:
            _, tags = train_dataset[item_id]
            for tag in tags:
                if tag in w2ind:
                    word_counts[w2ind[tag]] += 1
        labelpdf = word_counts / word_counts.sum()
        vocabsize = wordmatrix.shape[0]
        def negsamp(ignored_inds, num2samp):
            # Negative sampler that takes in indicies

            # Create new probability vector excluding positive samples
            nlabelpdf = np.copy(labelpdf)
            nlabelpdf[ignored_inds] = 0
            nlabelpdf /= nlabelpdf.sum()

            return np.random.choice(vocabsize, size=num2samp, p=nlabelpdf)

    with tf.Graph().as_default():

        # Build regressor
        if model_type == ModelTypes.mse:
            print('Building regressor with mean square error loss')
            model = MSEModel(image_feat_size,
                                         wordmatrix,
                                        learning_rate=learning_rate,
                                        hidden_units=network_size,
                                        use_batch_norm=True)

        elif model_type == ModelTypes.negsampling:
            print('Building regressor with negative sampling loss')
            model = NegSamplingModel(image_feat_size,
                                        wordmatrix,
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
            pos_word_ids = np.ones((batch_size, NUM_POSTIVE_EXAMPLES), dtype=np.int32)
            neg_word_ids = np.ones((batch_size, NUM_NEGATIVE_EXAMPLES), dtype=np.int32)
            evaluators = []
            for epoch in range(num_epochs):
                batch_time_total = 0
                run_time_total = 0

                loss = None
                for batch in range(int(num_items/batch_size)):
                    batch_time = time.time()
                    image_feats, text_tags = train_dataset.get_next_batch(batch_size)

                    # Generate positive examples
                    NUM_POSTIVE_EXAMPLES = 5
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
                    evaluator = evaluate_regressor(sess, model, val_image_feats, val_one_hot, wordmatrix, verbose=False)
                    evaluators.append(evaluator.evaluate())
                    eval_time = time.time() - eval_time
                    # Evaluate accuracy
                    #print('Epoch {}: Loss: {} Timing: {} {} {}'.format(epoch, loss, batch_time_total, run_time_total, eval_time))
                    print('Epoch {}: Loss: {} Perf: {} {} {}'.format(epoch, loss, *evaluators[-1]))

            if model_output_path:
                saver.save(sess, model_output_path)

            return evaluators


def main():
    import os
    import pickle
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

    args = parser.parse_args()
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

    result = train_model(train_dataset,
                test_dataset,
                w2v_lookup,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                network_size=map(int, args.network.split(',')),
                num_epochs=args.epochs,
                model_input_path=args.model_input_path,
                model_output_path=args.model_output_path,
                model_type=model_type)


if __name__ == '__main__':
    main()

