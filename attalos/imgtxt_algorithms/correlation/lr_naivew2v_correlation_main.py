import argparse
import numpy as np
from sklearn import linear_model

import attalos.util.log.log as l
import logging

from attalos.dataset.dataset import Dataset

from attalos.dataset.wordvectors.w2v import W2VWrapper
from attalos.dataset.wordvectors.glove import GloveWrapper

from attalos.dataset.transformers.onehot import OneHot
from attalos.dataset.transformers.naivew2v import NaiveW2V

from attalos.evaluation.evaluation import Evaluation


logger = l.getLogger(__name__)


def get_xy(dataset, tag_transformer=None):
    x = []
    y = []
    for idx in dataset:
        image_feats, text_feats = dataset.get_index(idx)
        if tag_transformer:
            text_feats = tag_transformer.get_multiple(text_feats)
        if idx % 1000 == 0:
            logger.debug("%s of %s" % (idx, dataset.num_images))
        x.append(image_feats)
        y.append(text_feats)
    return np.asarray(x), np.asarray(y)


def train(train_data, test_data, interpreter=None, n_jobs=-1, k=5):
    train_x, train_y = train_data
    model = linear_model.LinearRegression(n_jobs=n_jobs)

    logger.info("Fitting model.")
    model.fit(train_x, train_y)
    logger.info("Finished fitting.")

    if test_data:
        test_x, test_y = test_data

        logger.info("Predicting from test data.")
        predictions = model.predict(test_x)
        logger.info("Finished predicting.")

        if interpreter:
            logger.info("Converting predictions to multihot.")
            predictions = interpreter.to_multihot(predictions)
            logger.info("Finished converting.")

        logger.info("Evaluating performance.")
        evaluator = Evaluation(test_y, predictions, k=k)
        evaluator.evaluate()
        logger.info("Finished evaluating.")

    return model


def main():
    parser = argparse.ArgumentParser(description='Linear regression')

    # Required args
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
    parser.add_argument("--word_vector_type",
                        choices=("word2vec", "glove"),
                        default="word2vec",
                        help="Word vector type")
    parser.add_argument("--logging_level",
                        choices=("debug", "info", "warning", "error"),
                        default="warning",
                        help="Python logging level")

    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.logging_level.upper()))

    logger.info("Parsing train and test datasets.")
    train_dataset = Dataset(args.image_feature_file_train, args.text_feature_file_train)
    test_dataset = Dataset(args.image_feature_file_test, args.text_feature_file_test)

    logger.info("Reading word vectors from file.")
    if args.word_vector_type == "glove":
        from glove import Glove
        glove_model = Glove.load_stanford(args.word_vector_file)
        w2v_model = GloveWrapper(glove_model)
    else:  # args.word_vector_type == "word2vec" (default)
        import word2vec
        w2v_model = W2VWrapper(word2vec.load(args.word_vector_file))

    logger.info("Creating one hot tag mapper.")
    one_hot = OneHot([train_dataset, test_dataset], valid_vocab=w2v_model.vocab)

    logger.info("Creating w2v transformer.")
    w2v_transformer = NaiveW2V(one_hot, w2v_model, vocab=one_hot.keys())

    logger.info("Preparing train data from train datasets.")
    train_x, train_y = get_xy(train_dataset, tag_transformer=one_hot)

    logger.info("Transforming y using w2v transformer.")
    transformed_y = w2v_transformer.transform(train_y)
    train_data = (train_x, transformed_y)

    logger.info("Preparing test data from test dataset.")
    test_data = get_xy(test_dataset, tag_transformer=one_hot)

    logger.info("Training model.")
    model = train(train_data, test_data, interpreter=w2v_transformer)
    logger.info("Done.")


if __name__ == "__main__":
    main()
