"""Anomaly metrics."""
import numpy as np
import pandas as pd
import os
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt





def save_validation_losses_density(val_losses, val_labels, reports_save_path, dataset_type):

    """ Saves loss density """

    df = pd.DataFrame(val_losses, columns=['loss'])
    classes = ['NoFod', 'Fod']
    df['labels'] = [classes[label] for label in val_labels]
    sns.displot(df,x="loss" ,hue="labels")
    save_path = f"{reports_save_path}/Density_{dataset_type}.png"
    plt.savefig(save_path)

def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels, test_threshold=None
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """


    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
        )
    F1_scores = np.divide(
            2 * precision * recall,
            precision + recall,
            out=np.zeros_like(precision),
            where=(precision + recall) != 0,
    )

    if test_threshold is None:
        optimal_threshold = thresholds[np.argmax(F1_scores)]
    else:
        optimal_threshold = test_threshold*0.9
        

    
    f1_score = F1_scores[np.argmax(F1_scores)]
    predictions  = (anomaly_prediction_weights >= optimal_threshold).astype(int)
    # decaled_pred = np.delete(predictions, -1)
    # decaled_pred = np.insert(decaled_pred,0,0)
    # new_pred = np.array([int(i or j) for (i,j) in zip(predictions,decaled_pred)])
    accuracy = 100*((predictions == anomaly_ground_truth_labels).sum())/len(anomaly_ground_truth_labels)


    #########################

    return {"auroc": auroc, "threshold": thresholds, "f1_score": f1_score, 
                "accuracy": accuracy,"optimal_threshold":optimal_threshold, "predictions":predictions}


def save_confusion_matrix(labels, predictions,reports_save_path):

    """ Saves confusion matrix """

    classes = [0,1]
    matrix = metrics.confusion_matrix(labels, predictions, labels=classes)
    annot = [["00","01"],["10","11"]]
    for i in range(2):
        for j in range(2):
            percentage = matrix[i][j] / sum(matrix[i]) * 100
            annot[i][j] = f"{matrix[i][j]} ({percentage:.1f}%)"

    # Plot the confusion matrix with Seaborn
    fig = plt.figure(figsize=(5,5))
    ax = plt.subplot()

    # annot=True to annotate cells, ftm='g' to disable scientific notation
    sns.heatmap(matrix, annot=annot, square=True, fmt="s", ax=ax)
    # labels, title and ticks
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    ax.set_title('Confusion Matrix')
    classes = ['NoFod','Fod']
    ax.xaxis.set_ticklabels([classes[0], classes[1]])
    ax.yaxis.set_ticklabels([classes[0], classes[1]])

    plt.savefig(f"{reports_save_path}/matrix.png")