import argparse
import logging
import numpy as np
from sklearn import linear_model

from attalos.dataset.dataset import Dataset
from attalos.dataset.transformers.onehot import OneHot
from attalos.evaluation.evaluation import Evaluation

import attalos.util.log.log as l


logger = l.getLogger(__name__)


def get_xy(dataset, tag_transformer=None):
    x = []
    y = []
    for idx in dataset:
        image_feats, text_feats = dataset.get_index(idx)
        if tag_transformer:
            text_feats = tag_transformer.get_multiple(text_feats)
        x.append(image_feats)
        y.append(text_feats)
    return np.asarray(x), np.asarray(y)


def train(train_dataset, test_dataset=None, tag_transformer=None, n_jobs=-1):
    x, targets = get_xy(train_dataset, tag_transformer=tag_transformer)
    model = linear_model.LinearRegression(n_jobs=n_jobs)
    logger.info("Training model.")
    model.fit(x, targets)
    if test_dataset:
        x_test, truth = get_xy(test_dataset, tag_transformer=tag_transformer)
        logger.info("Evaluating model.")
        predictions = model.predict(x_test)
        evaluator = Evaluation(truth, predictions, k=5)
        evaluator.evaluate()
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
    parser.add_argument("--logging_level",
                        choices=("debug", "info", "warning", "error"),
                        default="warning",
                        help="Python logging level")

    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.logging_level.upper()))

    logger.info("Parsing train and test datasets.")
    train_dataset = Dataset(args.image_feature_file_train, args.text_feature_file_train)
    test_dataset = Dataset(args.image_feature_file_test, args.text_feature_file_test)

    logger.info("Creating one hot tag mapper.")
    one_hot = OneHot([train_dataset, test_dataset])

    train(train_dataset,
          test_dataset,
          one_hot)


if __name__ == "__main__":
    main()
