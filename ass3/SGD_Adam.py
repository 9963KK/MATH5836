import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim import Adam

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.cm as cm


kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def simple_nn_train(X, y, model=MLPClassifier(warm_start=True, early_stopping=True, solver='adam'),iter=30):
    '''
    :param model: The model used to train the data
    :param X: Value
    :param y: Label
    :param iter: iteration times
    :return: ovo, ovr: The f1_score and roc_auc of ovo and ovr
    '''
    ovo = []
    ovr = []
    for stra in ['ovo', 'ovr']:
        f1_socre_list = []
        roc_auc_score_list = []
        for i in range(1, iter + 1):
            f1_res = []
            roc_auc_res = []
            for train_index, test_index in kf.split(X, y):
                X_train, X_test, y_train, y_test = X.iloc[train_index], X.iloc[
                    test_index], y.iloc[train_index], y.iloc[test_index]
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_prob = model.predict_proba(X_test)
                f1 = f1_score(y_test, y_pred, average='micro')
                roc_auc = roc_auc_score(y_test, y_prob, multi_class=stra)
                f1_res.append(f1)
                roc_auc_res.append(roc_auc)
            f1_res_mean = np.mean(f1_res)
            roc_auc_res_mean = np.mean(roc_auc_res)
            if i % (iter / 10) == 0:
                print(
                    f'Iteration {i}/ {iter}: Using Stragety {stra} f1_score={f1_res_mean:.3f}; roc_auc={roc_auc_res_mean:.3f}')
            f1_socre_list.append(f1_res_mean)
            roc_auc_score_list.append(roc_auc_res_mean)
        print(
            f'Final Summary: Iterations {iter}, Stragety {stra}, f1_score={f1_socre_list[-1]:.3f}, roc_auc={roc_auc_score_list[-1]:.3f}')
        if stra == 'ovo':
            ovo.append(f1_socre_list)
            ovo.append(roc_auc_score_list)
        else:
            ovr.append(f1_socre_list)
            ovr.append(roc_auc_score_list)
    return ovo, ovr


def multi_class(x):
    if x <= 7:
        return 0
    elif x <= 10:
        return 1
    elif x <= 15:
        return 2
    else:
        return 3
if __name__ == '__main__':
    df = pd.read_csv('../abalone/abalone.data', names=['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight', 'viscera_weight', 'shell_weight', 'rings'])
    # class_dict = {0: '0~7', 1: '8~10', 2: '11~15', 3: '16~'}
    # Use different number to map the value of ring-age
    y = df.iloc[:, -1].map(multi_class)
    X = df.iloc[:, :-1]
    X['sex'] = X['sex'].map({'M': 0, 'F': 1, 'I': 3})
    ovo, ovr = simple_nn_train(X, y, model=MLPClassifier(warm_start=True, early_stopping=True, solver='adam'),iter=100)