import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc
from module.data_loader.mongodb_loader import ax_to_json


def plot_confusion_matrix(matrix):
    df_matrix = pd.DataFrame(matrix, index=[i for i in "ABC"],
                         columns=[i for i in "ABC"])
    ax = sn.heatmap(df_matrix, annot=True)
    return ax_to_json(ax)


def plot_roc_per_class(y_true, y_pred_proba):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    lb = LabelBinarizer()
    lb.fit(y_true)
    _y_true = lb.transform(y_true)

    n_classes = _y_true.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], thresholds = roc_curve(_y_true[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fig, ax = plt.subplots()
    for i in range(n_classes):
        ax.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Some extension of Receiver operating characteristic to multi-class')
    ax.legend(loc="lower right")
    return ax_to_json(ax)
