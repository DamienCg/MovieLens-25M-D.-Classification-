from matplotlib import pyplot as plt
from pretty_confusion_matrix import pp_matrix_from_data
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, ConfusionMatrixDisplay, \
    classification_report
import numpy as np


def printScore(y_test,y_pred,best_params):
    f1sc = f1_score(y_test, y_pred, average="macro", labels=np.unique(y_pred))
    Prec = precision_score(y_test, y_pred, average="macro", labels=np.unique(y_pred))
    Reca = recall_score(y_test, y_pred, average="macro", labels=np.unique(y_pred))
    Acc = accuracy_score(y_test, y_pred)
    print(classification_report(y_test, y_pred, zero_division=1))

    data = {'F1': f1sc, 'Precision': Prec, 'Accuracy': Acc,
            'Recall': Reca}
    courses = list(data.keys())
    values = list(data.values())

    # creating the bar plot
    plt.bar(courses, values, width=0.4)
    plt.xlabel("Score")
    plt.ylabel("%")
    plt.title("Best Param: " + str(best_params))
    plt.show()


def Print_Conf_Matrix(Model,X_test,y_test,y_pred,class_names):

    # Plot confusion matrix
    pp_matrix_from_data(y_test, y_pred, cmap=plt.cm.RdBu)
    titles_options = [
        ("Confusion matrix, without normalization", None),
        ("Normalized confusion matrix", "true"),
    ]
    for title, normalize in titles_options:
        disp = ConfusionMatrixDisplay.from_estimator(
            Model,
            X_test,
            y_test,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize=normalize,
        )
        disp.ax_.set_title(title)

    plt.show()