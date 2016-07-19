from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import scipy as sp

from sklearn import metrics

from oct2py import octave

class Eval(object):
    """
    Assumes:
        predicted: numpy matrix of category prediction confidence [trial, tag]
        eval_data: numpy matrix of ground truth labeling [trial, tag]

        precision and roc_auc cannot be evaluated if a label has no true samples
    """
    def __init__(self, truth, predictions):            
        self.predictions_raw = predictions

        self.k = 0.5
        self.predictions = self.confidence_threshold(0.5).astype(int)

        self.ground_truth = truth
        self.ntrials = predictions.shape[0]
        self.ntags = predictions.shape[1]

        self.metrics = [self.m_precision, self.m_recall, self.coverage_error, 
                        self.ranking_precision, self.ranking_loss, self.roc_auc]

    def top_k(self, k):
        if k <= 0:
            return
        elif k < 1 and k > 0:
            self.predictions = self.confidence_threshold(k).astype(int)
            return
        elif k > self.predictions_raw.shape[1]:
            return

        predictions = np.zeros(self.predictions_raw.shape)

        for raw_row, prediction_row in zip(self.predictions_raw, predictions):
            top_indices = np.argsort(raw_row)[-int(k):]
            prediction_row[top_indices] = 1

        self.predictions = predictions.astype(int)
        self.k = k

    def confidence_threshold(self, threshold):
        temp = np.copy(self.predictions_raw)
        temp[np.abs(self.predictions_raw) <= threshold] = 0
        temp[np.abs(self.predictions_raw) > threshold] = 1
        self.k = threshold
        return temp

    def print_evaluation(self):
        print('---Evaluation---')
        if self.k >= 1:
            print('---(where k = ' + str(self.k) + ')---')
        else:
            print('---where confidence > ' + str(self.k) + ' is classified as positive---')
        for metric in self.metrics:
            print(metric())

    def m_precision(self):
        """
        Unweighted precision score
        """
        try:
            self.m_precision = metrics.precision_score(self.ground_truth, self.predictions, average='macro')
        except UndefinedMetricWarning:
            pass
        return 'Precision: ' + str(self.m_precision)

    def w_precision(self):
        """
        Weighted precision score, requires appearance of each label in the ground truth set
        """
        try:
            self.w_precision = metrics.precision_score(self.ground_truth, self.predictions, average='weighted')
        except UndefinedMetricWarning:
            pass
        return 'Precision (weighted): ' + str(self.w_precision)

    def m_recall(self):
        """
        Unweighted recall score
        """
        try:
            self.m_recall = metrics.recall_score(self.ground_truth, self.predictions, average='macro')
        except UndefinedMetricWarning:
            pass
        return 'Recall: ' + str(self.m_recall)

    def w_recall(self):
        """
        Weighted recall score
        """
        try:
            self.w_recall = metrics.recall_score(self.ground_truth, self.predictions, average='weighted')
        except UndefinedMetricWarning:
            pass
        return 'Recall (weighted): ' + str(self.w_recall)

    def f1(self):
        self.f1 = metrics.f1_score(self.ground_truth, self.predictions, average='micro')
        return 'F1: ' + str(self.f1)

    def roc_auc(self):
        """
        Assumes:
            each column has at least two values (i.e. each example tag appears more than once)
        """
        try:
            self.roc_auc = metrics.roc_auc_score(self.ground_truth, self.predictions_raw, average='macro')
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

    def kendall_tau(self):
        scores = np.empty((self.ntrials, 1)) 
        for image_n in range(0, self.ntrials):
            [kt_value, p_value] = sp.stats.kendalltau(self.ground_truth[image_n], self.predictions_raw[image_n])
            scores[image_n] = kt_value
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

    evaluation_dataset = np.loadtxt(evaluation_dataset_filename)
    prediction_matrix = np.loadtxt(prediction_matrix_filename)

    evaluated = Eval(evaluation_dataset, prediction_matrix)

    evaluated.print_evaluation()

if __name__ == '__main__':
    main()