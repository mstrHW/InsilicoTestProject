import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc


def plot_confusion_matrix(matrix):
    df_matrix = pd.DataFrame(matrix, index=[i for i in "ABC"],
                         columns=[i for i in "ABC"])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_matrix, annot=True)
    plt.show()


def plot_roc_per_class(y_true, y_pred_proba):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    lb = LabelBinarizer()
    lb.fit(y_true)
    _y_true = lb.transform(y_true)
    # y_pred = lb.transform(y_pred)

    n_classes = _y_true.shape[1]

    for i in range(n_classes):
        fpr[i], tpr[i], thresholds = roc_curve(_y_true[:, i], y_pred_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label='ROC curve of class {0} (area = {1:0.2f})'
                                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.show()

    plt.show()
