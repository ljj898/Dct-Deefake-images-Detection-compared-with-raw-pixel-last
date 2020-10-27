import os
import numpy as np
from sklearn import metrics


def get_roc_auc(y_true, y_pos_score):
    return metrics.roc_auc_score(y_true, y_pos_score)


def get_f1(y_true, y_pred):
    return metrics.f1_score(y_true, y_pred)


def get_acc(y_true, y_pred):
    return metrics.accuracy_score(y_true, y_pred)


def evaluate(y_true, y_pred, y_pos_score):
    acc = get_acc(y_true, y_pred)
    f1 = get_f1(y_true, y_pred)
    roc_auc = get_roc_auc(y_true, y_pos_score)
    return acc, f1, roc_auc


def detail(y_true, all_scores, paths):
    '''
    错分样本分析
    '''
    assert y_true.shape[0] == all_scores.shape[0]
    if all_scores.shape[1] == 1:
        threshold = 0.5
        all_scores = np.squeeze(all_scores)
        y_pred = np.array(all_scores > threshold).astype(int)
        score = all_scores
    else:
        y_pred = np.argmax(all_scores, axis=1)
        score = all_scores[:, 1]
    error = y_true != y_pred
    indexes = np.where(error)[0]
    results = []
    for i in indexes:
        result = "{} label:{} pred:{} score:{}".format(paths[i], y_true[i], y_pred[i], score[i])
        results.append(result)

    with open(os.path.join("./output/hardExample", 'FPFN.txt'), 'w') as f:
        for item in results:
            f.write(item + ' \n')


def confusion_matrix(y_true, y_pred):
    return metrics.confusion_matrix(y_true, y_pred, labels=[0, 1])
