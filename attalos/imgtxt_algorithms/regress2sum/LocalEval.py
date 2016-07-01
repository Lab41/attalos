# TODO: This should be replaced by official evaluation code
import numpy as np


def calculate_precision(ground_truth, predicted):
    ground_truth_set = set(ground_truth)
    predicted_set = set(predicted)
    return float(len(ground_truth_set.intersection(predicted_set)))/len(predicted_set)


def calculate_recall(ground_truth, predicted):
    ground_truth_set = set(ground_truth)
    predicted_set = set(predicted)
    return float(len(ground_truth_set.intersection(predicted_set)))/len(ground_truth_set)


def get_pr(ground_truths, predictions):
    precisions = []
    recalls = []
    f1s = []
    for i in range(len(ground_truths)):
        precision = calculate_precision(ground_truths[i], predictions[i])
        recall = calculate_recall(ground_truths[i], predictions[i])

        if precision == 0 and recall == 0:
            f1 = 0
        else:
            f1 = 2*precision*recall/(precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return np.mean(precisions), np.mean(recalls),  np.mean(f1s)
