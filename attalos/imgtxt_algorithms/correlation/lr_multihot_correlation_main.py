import argparse
import logging
import numpy as np
from sklearn import linear_model

from attalos.dataset.dataset import Dataset
from attalos.dataset.transformers.onehot import OneHot
from attalos.evaluation.evaluation import Evaluation

import attalos.util.log.log as l
import logging

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

logger.info("Done.")


def train(train_data, test_data, **kwargs):
    interpreter = kwargs.get("interpreter", None)
    n_jobs = kwargs.get("n_jobs", -1)
    
    x_train, y_train = train_data
    model = linear_model.LinearRegression(n_jobs=n_jobs)
    logger.info("Fitting model.")
    model.fit(x_train, y_train)
    logger.info("Finished fitting.")
    if test_data:
        x_test, y_test = test_data
        logger.info("Predicting from test data.")
        predictions = model.predict(x_test)
        logger.info("Finished predicting.")
        logger.info("Evaluating performance.")
        evaluator = Evaluation(y_test, predictions, k=5)
        evaluator.evaluate()
        logger.info("Finished evaluating.")
    return model

logger.info("Done.")


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

    logger.info("Creating one hot transformer.")
    one_hot = OneHot([train_dataset, test_dataset])

    logger.info("Preparing train data from train datasets.")
    train_data = get_xy(train_dataset, tag_transformer=one_hot)

    logger.info("Preparing test data from test dataset.")
    test_data = get_xy(test_dataset, tag_transformer=one_hot)

    logger.info("Training model.")
    model = train(train_data, test_data)
    logger.info("Done.")


if __name__ == "__main__":
    main()
