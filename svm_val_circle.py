# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:15:08 2024

@author: usouu
"""

import os
import pandas as pd

import utils_common
import svm_validation

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, classification_report

def n_fold_cross_validation(data, labels, n_splits=5, stratified=True):
    """
    Perform n-fold cross-validation using an SVM classifier.

    Parameters:
        data (array-like): Feature data for training and testing. shape (samples, features)
        labels (array-like): Corresponding labels for the data. shape (samples,)
        n_splits (int): Number of folds for cross-validation.
        stratified (bool): Whether to use stratified KFold (ensuring class balance in each fold).

    Returns:
        dict: Cross-validation results containing accuracy scores and F1-scores per class.
    """
    # Convert data and labels to numpy arrays
    data = np.array(data)
    labels = np.array(labels)
    
    # Select the cross-validation strategy
    if stratified:
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Store results
    fold_results = []
    all_accuracies = []
    all_f1_scores = {}

    # Perform cross-validation
    for fold_idx, (train_index, test_index) in enumerate(kf.split(data, labels)):
        print(f"Fold {fold_idx + 1}/{n_splits}")

        # Split data
        data_train, data_test = data[train_index], data[test_index]
        labels_train, labels_test = labels[train_index], labels[test_index]

        # Train and evaluate SVM
        svm_classifier = SVC(kernel='rbf', C=1, gamma='scale', decision_function_shape='ovr')
        svm_classifier.fit(data_train, labels_train)

        # Predict and evaluate
        labels_pred = svm_classifier.predict(data_test)
        accuracy = accuracy_score(labels_test, labels_pred)
        report = classification_report(labels_test, labels_pred, output_dict=True)

        # Store accuracy
        all_accuracies.append(accuracy)

        # Store F1 scores per class
        class_f1_scores = {f"Class_{key}": value['f1-score'] for key, value in report.items() if key.isdigit()}
        for class_label, f1_score in class_f1_scores.items():
            if class_label not in all_f1_scores:
                all_f1_scores[class_label] = []
            all_f1_scores[class_label].append(f1_score)

        # Store results per fold
        fold_results.append({
            "Fold": fold_idx + 1,
            "Accuracy": accuracy,
            "Class_F1_Scores": class_f1_scores
        })

        # Print report for each fold
        print(f"Accuracy for Fold {fold_idx + 1}: {accuracy:.4f}")
        print(classification_report(labels_test, labels_pred))

    # Compute overall statistics
    mean_accuracy = np.mean(all_accuracies)
    std_accuracy = np.std(all_accuracies)
    mean_f1_scores = {class_label: np.mean(scores) for class_label, scores in all_f1_scores.items()}

    # Return the results
    return {
        "Fold_Results": fold_results,
        "Mean_Accuracy": mean_accuracy,
        "Std_Accuracy": std_accuracy,
        "Mean_F1_Scores": mean_f1_scores
    }

# 示例测试
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    X = np.random.rand(100, 10)  # 100 样本, 10 特征
    y = np.random.randint(0, 3, 100)  # 3 类标签 (0, 1, 2)



    # 进行5折交叉验证
    results = n_fold_cross_validation(X, y, n_splits=5, stratified=True)

    print("\nOverall Cross-Validation Results:")
    print(f"Mean Accuracy: {results['Mean_Accuracy']:.4f} ± {results['Std_Accuracy']:.4f}")
    print("Mean F1-Scores per Class:")
    for class_label, f1_score in results["Mean_F1_Scores"].items():
        print(f"{class_label}: {f1_score:.4f}")

