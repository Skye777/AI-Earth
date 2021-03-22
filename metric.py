"""
@author: Skye Cui
@file: metric.py
@time: 2021/3/21 15:23
@description: 
"""
import numpy as np
from sklearn.metrics import mean_squared_error


def rmse(y_true, y_preds):
    return np.sqrt(mean_squared_error(y_pred=y_preds, y_true=y_true))


def nino_seq(ssta):
    # inputs: [sample, time, h, w]
    # outputs: [sample, time]
    nino = []
    for sample in range(len(ssta)):
        n_index = [np.mean(ssta[sample, i, 10:13, 38:49]) for i in range(len(ssta[sample]))]
        nino.append(n_index)
    return nino


def score(y_true, y_preds):
    # inputs: [sample, time, h, w, predictor]
    accskill_score = 0
    rmse_score = 0
    a = [1.5] * 4 + [2] * 7 + [3] * 7 + [4] * 6
    y_true_mean = np.mean(y_true, axis=0)
    y_pred_mean = np.mean(y_true, axis=0)

    for i in range(24):
        fenzi = np.sum((y_true[:, i] - y_true_mean[i]) * (y_preds[:, i] - y_pred_mean[i]))
        fenmu = np.sqrt(
            np.sum((y_true[:, i] - y_true_mean[i]) ** 2) * np.sum((y_preds[:, i] - y_pred_mean[i]) ** 2))
        cor_i = fenzi / fenmu

        accskill_score += a[i] * np.log(i + 1) * cor_i

        rmse_score += rmse(y_true[:, i], y_preds[:, i])
    return 2 / 3.0 * accskill_score - rmse_score
