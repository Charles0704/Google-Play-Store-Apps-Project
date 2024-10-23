import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from itertools import cycle
from sklearn import metrics
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.metrics import confusion_matrix, classification_report, f1_score, precision_score, recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import auc, roc_curve

def cm(test_X,test_Y,model,str):         # 指标以及混淆矩阵
    y_pred = model.predict(test_X)
    cm = confusion_matrix(test_Y, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    accuracy = accuracy_score(test_Y, y_pred)
    precision, recall, f1_score, _ = precision_recall_fscore_support(test_Y, y_pred, average='weighted')

    print(f'Accuracy: {accuracy}')
    print(f'Precision: {precision}')
    print(f'Recall: {recall}')
    print(f'F1 Score: {f1_score}')

    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(cm_normalized, annot=True, ax=ax, cmap='Blues', fmt='.2f')

    # Labels, title and ticks
    label_classes = np.unique(test_Y)
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Normalized Confusion Matrix of '+str)
    ax.xaxis.set_ticklabels(['0', '1', '2'])
    ax.yaxis.set_ticklabels(['0', '1', '2'])
    plt.show()


    from itertools import cycle

def ROC(train_Y,test_Y,test_X,model,str):
    y_score=model.predict_proba(test_X)
    label_binarizer = LabelBinarizer().fit(train_Y)
    y_onehot_test = label_binarizer.transform(test_Y)

    fpr, tpr, roc_auc = dict(), dict(), dict()
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    n_classes=y_onehot_test.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    fpr_grid = np.linspace(0.0, 1.0, 1000)

    # Interpolate all ROC curves at these points
    mean_tpr = np.zeros_like(fpr_grid)

    for i in range(n_classes):
        mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

    # Average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = fpr_grid
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    
    fig, ax = plt.subplots(figsize=(7, 7))
    plt.style.use('ggplot')

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    for class_id in range(n_classes):
        RocCurveDisplay.from_predictions(
            y_onehot_test[:, class_id],
            y_score[:, class_id],
            name=f"ROC curve for quality {1+class_id}",
            ax=ax,
        )

    plt.plot(np.linspace(0,1,100),np.linspace(0,1,100),'--',color='grey',label='chance')
    plt.axis("square")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve of "+str)
    plt.legend()
    plt.savefig("ROC.pdf")
    plt.show()

    