# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 21:09:50 2024

@author: usouu
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score

def svm_5fold_cross_validation(X, y, kernel='rbf', C=1.0, gamma='scale'):
    """
    Perform 5-fold cross-validation using SVM.
    
    Parameters:
        X (array-like): Feature matrix (samples x features).
        y (array-like): Target labels.
        kernel (str): Kernel type for SVM (default: 'rbf').
        C (float): Regularization parameter (default: 1.0).
        gamma (str or float): Kernel coefficient (default: 'scale').
    
    Returns:
        dict: A dictionary containing accuracy, class-wise F1 scores, and detailed classification report.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []
    reports = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = SVC(kernel=kernel, C=C, gamma=gamma)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=None)  # Per-class F1 scores
        report = classification_report(y_test, y_pred, output_dict=True)
        
        accuracies.append(acc)
        f1_scores.append(f1)
        reports.append(report)
    
    # Compute mean accuracy and F1-scores
    mean_accuracy = np.mean(accuracies)
    mean_f1_scores = np.mean(np.array(f1_scores), axis=0)
    
    return {
        'Accuracy': mean_accuracy,
        'Class_F1_Scores': mean_f1_scores,
        'Detailed_Report': reports
    }

def svm_evaluation_single(data, labels, partitioning='sequential', rate=0.7):
    """
    Train and evaluate an SVM classifier with given data and labels.

    Parameters:
        data (array-like): Feature data for training and testing. shape (samples, features)
        labels (array-like): Corresponding labels for the data. shape (samples,)
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

def cms_array(matrices):
    channel_0 = matrices[:, 0, :, :]
    channel_1 = matrices[:, 1, :, :]
    channel_2 = matrices[:, 2, :, :]
    
    channel_0_array = matrices_lower_triangles(channel_0)
    channel_1_array = matrices_lower_triangles(channel_1)
    channel_2_array = matrices_lower_triangles(channel_2)
    
    channel_0_array = np.array(channel_0_array)
    channel_1_array = np.array(channel_1_array)
    channel_2_array = np.array(channel_2_array)
    
    channel_full_array = np.hstack((channel_0_array, channel_1_array, channel_2_array))
    
    return channel_0_array, channel_1_array, channel_2_array, channel_full_array

def matrix_lower_triangle(matrix):
    tril_indices = np.tril_indices(matrix.shape[0], k=-1)
    lower_triangle_values = matrix[tril_indices]
    return lower_triangle_values

def matrices_lower_triangles(matrices):
    samples = []
    for matrix in matrices:
        tril_indices = np.tril_indices(matrix.shape[0], k=-1)
        lower_triangle_values = matrix[tril_indices]
        samples.append(lower_triangle_values)
    
    return samples

if __name__ == '__main__':
    import utils_common
    # %% Example usage
    # sample_experiment, sample_feature = 'sub1ex1', 'PCC'
    # cms = utils_common.load_cms_seed(sample_experiment, sample_feature)
    
    # alpha, beta, gamma, data = cms_array(cms)
    
    # labels = utils_common.read_labels_seed()
    
    # # single validation
    # result_entry_sequential = svm_evaluation_single(data, labels)
    # result_entry_randomized = svm_evaluation_single(data, labels, 'randomized')
    
    # # cross validation
    # results = svm_5fold_cross_validation(data, labels)
    
    # %% Circle
    subject_range, experiment_range, selected_feature, dataset = range(1, 2), range(1, 4), 'PCC', 'SEED'
    result_entries = []
    labels = utils_common.read_labels(dataset)
    
    for sub in subject_range:
        for ex in experiment_range:
            identifier = f'sub{sub}ex{ex}'
            cms = utils_common.load_cms(dataset, experiment=identifier, feature=selected_feature)
            
            alpha, beta, gamma, data = cms_array(cms)
            
            result_entry = svm_5fold_cross_validation(data, labels)
            result_entries.append(result_entry)