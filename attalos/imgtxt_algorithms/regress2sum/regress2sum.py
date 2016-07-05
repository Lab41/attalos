import argparse
import gzip
import numpy as np
import tensorflow as tf

from attalos.dataset.dataset import Dataset
from MinHeap import get_top_k
from LocalEval import get_pr


def tags_2_vec(tags, w2v_model=None):
    """
    Takes a list of text tags and returns the normalized sum of the word vectors

    Args:
        tags: A iterable of text tags
        w2v_model: a dictionary like object where the keys are words and the values are word vectors

    Returns:
        Normalized sum of the word vectors
    """
    if len(tags) == 0:
        return np.zeros(300)
    else:
        output = np.sum([w2v_model[tag] for tag in tags], axis=0)
        return output / np.linalg.norm(output)


def evaluate_regressor(regressor, val_image_feats, val_text_tags, w2v_model, k=5):
    """
    Takes a regressor and returns the precision/recall on the test data
    Args:
        regressor: a tensorflow.contrib.learn regression estimator
        val_image_feats: Image features to test performance on
        val_text_tags: Text Tags to test performance on
        w2v_model: a dictionary like object where the keys are words and the values are word vectors
        k: Top number of items to retrieve to test precision/recall on

    Returns:
        (precision, recall, f1): A tuple of precision, recall, and F1 score
    """
    val_pred = regressor.predict(val_image_feats)
    topks = []
    for i in range(val_pred.shape[0]):
        topk = get_top_k(val_pred[i, :]/np.linalg.norm(val_pred[i, :]), w2v_model, k)
        topks.append(topk)

    return get_pr(val_text_tags, topks)


def train_model(train_dataset,
                test_dataset,
                w2v_model,
                batch_size=128,
                num_epochs=200,
                save_path=None,
                verbose=True):
    """
    Train a regression model to map image features into the word vector space
    Args:
        train_dataset: Training attalos.dataset.dataset object
        test_dataset: Test attalos.dataset.dataset object
        w2v_model: A dictionary like object where the keys are words and the values are word vectors
        batch_size: Batch size to use for training
        num_epochs: Number of epochs to train for
        save_path: Path to save model to allow restart

    Returns:
        regressor: The trained regression estimator
    """
    num_items = train_dataset.num_images

    # Get validation data
    val_image_feats, val_text_tags = test_dataset.get_next_batch(batch_size*10)
    for i in range(batch_size):
        val_image_feats[i, :] = val_image_feats[i, :]/np.linalg.norm(val_image_feats[i, :])

    # Allocate GPU memory as needed (vs. allocating all the memory)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Build regressor
    regressor = tf.contrib.learn.TensorFlowDNNRegressor(hidden_units=[200,200],
                                                        steps=10,
                                                        continue_training=True,
                                                        verbose=0)

    for epoch in range(num_epochs):
        for batch in range(int(num_items/batch_size)):
            image_feats, text_tags = train_dataset.get_next_batch(batch_size)
            for i in range(batch_size):
                image_feats[i, :] = image_feats[i, :]/ np.linalg.norm(image_feats[i, :])
            word_feats = [tags_2_vec(tags, w2v_model) for tags in text_tags]
            regressor.fit(image_feats, word_feats)

        if verbose:
            precision, recall, f1 = evaluate_regressor(regressor, val_image_feats, val_text_tags, w2v_model)
            # Evaluate accuracy
            print('Epoch:', epoch, 'Precision:', precision, 'Recall:', recall, 'F1:', f1)

    if save_path:
        regressor.save(save_path)

    return regressor


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
            w2v_vector = np.array([float(j) for j in line[1:]])
            # Normalize vector before storing
            w2v_lookup[line[0]] = w2v_vector / np.linalg.norm(w2v_vector)

    train_model(train_dataset,
                test_dataset,
                w2v_lookup,
                batch_size=args.batch_size,
                num_epochs=args.epochs)

if __name__ == '__main__':
    main()

