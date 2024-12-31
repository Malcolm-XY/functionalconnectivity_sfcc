# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 21:09:50 2024

@author: usouu
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def svm_evaluation_single(data, labels, partitioning='sequential', rate=0.7):
    """
    Train and evaluate an SVM classifier with given data and labels.

    Parameters:
        data (array-like): Feature data for training and testing.
        labels (array-like): Corresponding labels for the data.
        partitioning (str): Partitioning method, 'sequential' or 'randomized'.
        rate (float): Proportion of data used for training (0 < rate < 1).

    Returns:
        float: Accuracy of the model on the test set.
    """
    # Validate input parameters
    if partitioning not in ['sequential', 'randomized']:
        raise ValueError("Partitioning must be 'sequential' or 'randomized'.")
    if not (0 < rate < 1):
        raise ValueError("Rate must be a float between 0 and 1.")

    # Convert data and labels to numpy arrays for easier manipulation
    data = np.array(data)
    labels = np.array(labels)

    # Split data based on the chosen partitioning method
    if partitioning == 'sequential':
        split_index = int(rate * len(data))
        data_train, data_test = data[:split_index], data[split_index:]
        labels_train, labels_test = labels[:split_index], labels[split_index:]
    elif partitioning == 'randomized':
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        split_index = int(rate * len(data))
        train_indices, test_indices = indices[:split_index], indices[split_index:]
        data_train, data_test = data[train_indices], data[test_indices]
        labels_train, labels_test = labels[train_indices], labels[test_indices]
    
    # Construct and train SVM
    svm_classifier = SVC(kernel='rbf', C=1, gamma='scale', decision_function_shape='ovr')
    svm_classifier.fit(data_train, labels_train)
    
    # Test the classifier
    labels_pred = svm_classifier.predict(data_test)
    accuracy = accuracy_score(labels_test, labels_pred)    
    report = classification_report(labels_test, labels_pred, output_dict=True)
    
    # Store results
    result_entry = {
        "Accuracy": accuracy,
        "Class_F1_Scores": {f"Class_{key}": value['f1-score'] for key, value in report.items() if key.isdigit()},
        "Detailed_Report": report
    }
    
    # Output the evaluation metrics
    print("Classification Report:")
    print(classification_report(labels_test, labels_pred))
    print(f"Accuracy: {accuracy:.2f}\n")
    
    return result_entry

# 假设 data 和 labels 已经定义
# labels: shape (samples,)
# data: shape (samples, features)

# import utils
# get label and cmdata
# labels = utils.get_label()
# cmdata = utils.load_cmdata('PCC', 'joint', 'sub1ex1')

# accuracy_sequential = svm_evaluation_single(cmdata, labels)
# accuracy_randomized = svm_evaluation_single(cmdata, labels, partitioning='randomized')