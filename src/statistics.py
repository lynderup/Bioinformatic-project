import numpy as np

import compare_tm_pred as compare_char
import compare_prediction as compare_tmh


def to_dictionary(predictions):
    true = {}
    pred = {}

    for name, xs, zs, prediction in predictions:
        true[name] = "%s # %s" % (xs, zs)
        pred[name] = "%s # %s" % (xs, prediction)

    return true, pred


def char_statistic(predictions):
    acs = []
    sns = []
    sps = []

    for prediction in predictions:
        true, pred = to_dictionary(prediction)

        ac, sn, sp = compare_char.do_compare(true, pred, False)

        acs.append(ac)
        sns.append(sn)
        sps.append(sp)

    ac_mean = np.mean(acs)
    ac_variance = np.var(acs)

    sn_mean = np.mean(sns)
    sn_variance = np.var(sns)

    sp_mean = np.mean(sps)
    sp_variance = np.var(sps)

    return ac_mean, ac_variance, sn_mean, sn_variance, sp_mean, sp_variance


def tmh_50ed5_statistic(predictions):
    precisions = []
    recalls = []

    for prediction in predictions:
        precision, recall = \
            compare_tmh.compare_predictions(prediction, compare_tmh.endpoints_diff_below_5_overlap_over_50_percent)

        precisions.append(precision)
        recalls.append(recall)

    precision_mean = np.mean(precisions)
    precision_variance = np.var(precisions)

    recall_mean = np.mean(recalls)
    recall_variance = np.var(recalls)

    return precision_mean, precision_variance, recall_mean, recall_variance


def tmh_25_statistic(predictions):
    precisions = []
    recalls = []

    for prediction in predictions:
        precision, recall = compare_tmh.compare_predictions(prediction, compare_tmh.overlap_over_25_percent)

        precisions.append(precision)
        recalls.append(recall)

    precision_mean = np.mean(precisions)
    precision_variance = np.var(precisions)

    recall_mean = np.mean(recalls)
    recall_variance = np.var(recalls)

    return precision_mean, precision_variance, recall_mean, recall_variance


class Statistic:

    def __init__(self, name):
        self.predictions = []
        self.name = name

    def add_prediction(self, prediction):
        self.predictions.append(prediction)

    def print_statistics(self):
        ac_mean, ac_variance, sn_mean, sn_variance, sp_mean, sp_variance = char_statistic(self.predictions)
        precision_mean_50ed5, precision_variance_50ed5, recall_mean_50ed5, recall_variance_50ed5 = \
            tmh_50ed5_statistic(self.predictions)
        precision_mean_25, precision_variance_25, recall_mean_25, recall_variance_25 = \
            tmh_25_statistic(self.predictions)

        print(self.name)
        print("Char measure")
        print("Approximate correlation:")
        print("Mean: %.4f   Variance: %.4f" % (ac_mean, ac_variance))
        print("Sensitivity:")  # Recall
        print("Mean: %.4f   Variance: %.4f" % (sn_mean, sn_variance))
        print("Specificity:")  # Precision
        print("Mean: %.4f   Variance: %.4f" % (sp_mean, sp_variance))

        print("50% overlap, endpoint diff =< 5 measure")
        print("Precision:")
        print("Mean: %.4f   Variance: %.4f" % (precision_mean_50ed5, precision_variance_50ed5))
        print("Recall:")
        print("Mean: %.4f   Variance: %.4f" % (recall_mean_50ed5, recall_variance_50ed5))

        print("25% overlap measure")
        print("Precision:")
        print("Mean: %.4f   Variance: %.4f" % (precision_mean_25, precision_variance_25))
        print("Recall:")
        print("Mean: %.4f   Variance: %.4f" % (recall_mean_25, recall_variance_25))


