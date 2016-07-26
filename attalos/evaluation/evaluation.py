from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np
import scipy as sp

from sklearn import metrics

class Evaluation(object):
    """
    Assumes:
        predicted: matrix of label prediction confidence [trial, tag]
        eval_data: matrix of ground truth classifications [trial, tag]
    """
    def __init__(self, truth, predictions, k=0.5):            
        self.predictions_raw = predictions

        self.set_k(k)

        self.ground_truth = truth
        self.ntrials = predictions.shape[0]
        self.ntags = predictions.shape[1]

        # Defaults are precision, recall, and F1.
        # May be extended by appending other metric functions to list
        self.metrics = [self.precision, self.recall, self.f1]

    def set_k(self, k):
        """
        Performs two possible functions:
            If k is greater than 1, computes predictions for top-k
                from the raw predictions.
            If k is between 0 and 1, sets k as a threshold for
                positive predictions.
        """
        if k <= 0:
            return
        elif k < 1 and k > 0:
            self.predictions = self.confidence_threshold(k).astype(int)
            return
        elif k > self.predictions_raw.shape[1]:
            return

        predictions = np.zeros(self.predictions_raw.shape)

        for raw_r, prediction_r in zip(self.predictions_raw, predictions):
            top_indices = np.argsort(raw_r)[-int(k):]
            prediction_r[top_indices] = 1

        self.predictions = predictions.astype(int)
        self.k = k

    def confidence_threshold(self, threshold):
        """
        Sets the given float as a threshold for positive prediction,
        producing binary predictions from the raw confidence scores.
        """
        temp = np.copy(self.predictions_raw)
        temp[np.abs(self.predictions_raw) <= threshold] = 0
        temp[np.abs(self.predictions_raw) > threshold] = 1
        self.k = threshold
        return temp

    # Beginning of metrics

    def precision(self):
        """
        Unweighted mean precision score across predicted tags
        """
        try:
            self.m_precision = metrics.precision_score(
                self.ground_truth, self.predictions, 
                average='macro')
        except UndefinedMetricWarning:
            pass
        return 'Precision: ' + str(self.m_precision)

    def recall(self):
        """
        Unweighted mean recall score across predicted tags
        """
        try:
            self.m_recall = metrics.recall_score(
                self.ground_truth, self.predictions, 
                average='macro')
        except UndefinedMetricWarning:
            pass
        return 'Recall: ' + str(self.m_recall)

    def f1(self):
        """
        Computes recall score for top-k or above-threshold predictions
        """
        self.f1 = metrics.f1_score(
            self.ground_truth, self.predictions, 
            average='macro')
        return 'F1: ' + str(self.f1)

    def roc_auc(self):
        """
        Computes the area under the ROC curve, given the raw
            confidence scores.
        Assumes:
            each column has at least two values 
            (i.e. each example tag appears more than once)
        """
        try:
            self.roc_auc = metrics.roc_auc_score(
                self.ground_truth, self.predictions_raw, 
                average='macro')
            return 'AUC: ' + str(self.roc_auc)
        except ValueError:
            return 'Area Under Curve could not be computed ...'

    def coverage_error(self):
        """
        The coverage_error function computes the average number of labels 
        that have to be included in the final prediction such that all true 
        labels are predicted. This is useful if you want to know how many 
        top-scored-labels you have to predict in average without missing any 
        true one. The best value of this metrics is thus the average number 
        of true labels.
        """
        self.coverage_error = metrics.coverage_error(
            self.ground_truth, self.predictions_raw)
        avg_true_labels = np.count_nonzero(self.ground_truth) / self.ntrials
        ce_message = 'Coverage Error [' + str(avg_true_labels) + ', ~): '
        return ce_message + str(self.coverage_error)

    def ranking_precision(self):
        """
        Label ranking average precision (LRAP) is the average over each 
        ground truth label assigned to each sample, of the ratio of 
        true vs. total labels with lower score. This metric will yield 
        better scores if you are able to give better rank to the labels 
        associated with each sample. The obtained score is always strictly 
        greater than 0, and the best value is 1.
        """
        self.ranking_precision = metrics.label_ranking_average_precision_score(
            self.ground_truth, self.predictions_raw)
        rp_message = 'Ranking Precision (0, 1]: '
        return rp_message + str(self.ranking_precision)

    def ranking_loss(self):
        """
        Computes the ranking loss, which averages the number of
        incorrectly-ordered labels (i.e. true labels have a lower 
        score than false labels, weighted by the the inverse number 
        of false and true labels) based on raw precision scores.
        """
        self.ranking_loss = metrics.label_ranking_loss(
            self.ground_truth, self.predictions_raw)
        rl_message = 'Ranking Loss: '
        return rl_message + str(self.ranking_loss)

    def spearman(self):
        """
        Computes the average Spearman's correlation coefficient over
        the raw predictions and ground truths.
        """
        scores = np.empty((self.ntags, 1))
        for tag_n in range(0, self.ntags):
            [spearman_value, p_value] = sp.stats.spearmanr(
                self.ground_truth[:,tag_n], self.predictions_raw[:,tag_n])
            if (math.isnan(spearman_value)):
                spearman_value = 0.0
            scores[tag_n] = spearman_value

        self.spearman = np.average(scores)
        s_message = 'Average Spearman\'s coefficient is: '
        return s_message + str(self.spearman)

    # End of metrics

    def evaluate(self):
        """
        Prints the results of evaluation metrics.
        """
        print('---Evaluation---')
        if self.k >= 1:
            print('---(where k = ' 
                  + str(self.k) 
                  + ')---')
        else:
            print('---where confidence > ' 
                  + str(self.k) 
                  + ' is classified as positive---')
        for metric in self.metrics:
            print(metric())


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

    evaluated = Evaluation(evaluation_dataset, prediction_matrix)

    evaluated.evaluate()

if __name__ == '__main__':
    main()