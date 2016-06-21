from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import pickle

import numpy as np
import scipy as sp

from sklearn import metrics
from sklearn.feature_extraction.text import TfidfTransformer

from attalos.dataset.onehot import OneHot

TAG_INDEX = 2


class Eval(object):
    """
    Assumes:
        predicted: numpy matrix of category prediction confidence [trial, tag]
        eval_data: numpy matrix of ground truth labeling [trial, tag]

        precision and roc_auc cannot be evaluated if a label has no true samples
    """
    def __init__(self, truth, predictions):            
        self.predictions_raw = predictions
        temp = predictions
        temp[np.abs(self.predictions_raw) < 0.5] = 0
        temp[np.abs(self.predictions_raw) >= 0.5] = 1
        self.predictions = temp.astype(int)

        self.ground_truth = truth
        self.ntrials = predictions.shape[0]
        self.ntags = predictions.shape[1]
        
        self.metrics = [self.w_precision, self.w_recall, self.coverage_error, 
                            self.ranking_precision, self.ranking_loss, self.roc_auc]

    def print_evaluation(self):
        print('---Evaluation---')
        for metric in self.metrics:
            print(metric())

    def m_precision(self):
        try:
            self.m_precision = metrics.precision_score(self.ground_truth, self.predictions, average='macro')
        except UndefinedMetricWarning:
            pass
        return 'Precision (macro): ' + str(self.m_precision)

    def w_precision(self):
        try:
            self.w_precision = metrics.precision_score(self.ground_truth, self.predictions, average='weighted')
        except UndefinedMetricWarning:
            pass
        return 'Precision (weighted): ' + str(self.w_precision)

    def m_recall(self):
        try:
            self.m_recall = metrics.recall_score(self.ground_truth, self.predictions, average='macro')
        except UndefinedMetricWarning:
            pass
        return 'Recall (macro): ' + str(self.m_recall)

    def w_recall(self):
        try:
            self.w_recall = metrics.recall_score(self.ground_truth, self.predictions, average='weighted')
        except UndefinedMetricWarning:
            pass
        return 'Recall (weighted): ' + str(self.w_recall)

    def roc_auc(self):
        """
        Assumes:
            each column has at least two values (i.e. each example tag appears more than once)
        """
        #scores = np.empty((N_TAGS, 1))
        #for image_n in range(0, N_TAGS):
        #    score = metrics.roc_auc_score(self.ground_truth[:,image_n], self.predictions_raw[:,image_n])
        #    scores[image_n] = score
        #self.roc_auc = np.average(scores)
        try:
            self.roc_auc = metrics.roc_auc_score(self.ground_truth, self.predictions_raw)
            return 'Area Under Curve [0, 1, where 0.5 = chance]: ' + str(self.roc_auc)
        except ValueError:
            return 'Area Under Curve could not be computed ...'

    def coverage_error(self):
        self.coverage_error = metrics.coverage_error(self.ground_truth, self.predictions_raw)
        avg_true_labels = np.count_nonzero(self.ground_truth) / self.ntrials
        return 'Coverage Error [' + str(avg_true_labels) + ', ~): ' + str(self.coverage_error)

    def ranking_precision(self):
        self.ranking_precision = metrics.label_ranking_average_precision_score(self.ground_truth, self.predictions_raw)
        return 'Ranking Precision (0, 1]: ' + str(self.ranking_precision)

    def ranking_loss(self):
        self.ranking_loss = metrics.label_ranking_loss(self.ground_truth, self.predictions_raw)
        return 'Ranking Loss: ' + str(self.ranking_loss)

    def msq_error(self):
        scores = np.empty((self.ntrials, 1))
        for image_n in range(0, self.ntrials):
            score = metrics.mean_squared_error(self.ground_truth[image_n], self.predictions_raw[image_n])
            scores[image_n] = score
        self.msq_error = np.average(scores)
        return 'Mean Squared Error: ' + str(self.msq_error)

    def spearman(self):
        scores = np.empty((self.ntags, 1))
        for tag_n in range(0, self.ntags):
            [spearman_value, p_value] = sp.stats.spearmanr(self.ground_truth[:,tag_n], self.predictions_raw[:,tag_n])
            if (math.isnan(spearman_value)):
                spearman_value = 0.0
            scores[tag_n] = spearman_value

        self.spearman = np.average(scores)
        return 'Average Spearman\'s coefficient is: ' + str(self.spearman)

    """
    GRAVEYARD

    def tf_idf(self):
        transformer = TfidfTransformer().fit(self.ground_truth)
        tf_idfs = transformer.transform(self.ground_truth)
        print(tf_idfs)
    """

    def kendall_tau(self):
        scores = np.empty((self.ntrials, 1)) 
        for image_n in range(0, self.ntrials):
            [kt_value, p_value] = sp.stats.kendalltau(self.ground_truth[image_n], self.predictions_raw[image_n])
            scores[image_n] = kt_value
            print("kt value = " + str(kt_value))
        self.kendall_tau = np.average(scores)
        return 'Kendall\'s tau [-1, 1]: ' + str(self.kendall_tau)

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Test Evaluation')
    parser.add_argument('evaluation_dataset_filename',
                        dest='evaluation_dataset_filename',
                        type=str,
                        help='Evaluation Dataset Filename')
    parser.add_argument('prediction_matrix_filename',
                        dest='prediction_matrix_filename',
                        type=str,
                        help='Prediction Matrix Filename')
    
    args = parser.parse_args()

    #Structures should have been saved to file via pickle.dump()
    """evaluation_dataset_file = open(args.evaluation_dataset_filename, "rb")
    prediction_matrix_file = open(args.prediction_matrix_filename, "rb")

    evaluation_dataset = pickle.load(evaluation_dataset_file)
    prediction_matrix = pickle.load(prediction_matrix_file)

    evaluation_dataset_file.close()
    prediction_matrix_file.close()

    evaluated = TestingEval(evaluation_dataset, prediction_matrix)

    evaluated.print_evaluation()"""

if __name__ == '__main__':
    main()