from sklearn.metrics import precision_recall_fscore_support, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


def get_f1_per_class(y_test, y_pred):
    return classification_report(y_test, y_pred, output_dict=True)


def multiclass_roc_auc_score(y_test, y_pred_proba, average=None):
    lb = LabelBinarizer()
    lb.fit(y_test)
    _y_test = lb.transform(y_test)

    return roc_auc_score(_y_test, y_pred_proba, average=average)
