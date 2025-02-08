# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 21:09:50 2024

@author: usouu
"""
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score

def save_results(results, filename):
    """
    Save results dictionary as a CSV file in the results folder.
    If the file exists, append new results; otherwise, create a new file.
    """
    os.makedirs("results", exist_ok=True)
    file_path = os.path.join("results", filename)
    df = pd.DataFrame(results)
    
    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    
    df.to_csv(file_path, index=False)
    print(f"Results saved to {file_path}")

def svm_5fold_cross_validation(X, y, kernel='rbf', C=1.0, gamma='scale', identifier="experiment"):
    """
    Perform 5-fold cross-validation using SVM and save the results.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    f1_scores = []
    reports = []
    
    for train_index, test_index in skf.split(X, y):
        print(f'Processing {identifier}')
        
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = SVC(kernel=kernel, C=C, gamma=gamma)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average=None)  # Per-class F1 scores
        report = classification_report(y_test, y_pred, output_dict=True)
        
        accuracies.append(acc)
        f1_scores.append(f1.tolist())  # Convert to list for DataFrame compatibility
        reports.append(report)
    
    # Compute mean accuracy and F1-scores
    mean_accuracy = np.mean(accuracies)
    mean_f1_scores = np.mean(np.array(f1_scores), axis=0).tolist()
    
    results = {
        'Identifier': [identifier] * len(accuracies),
        'Fold': list(range(1, len(accuracies) + 1)),
        'Accuracy': accuracies,
        'Mean_Accuracy': [mean_accuracy] * len(accuracies),
        'Mean_Class_F1_Scores': [mean_f1_scores] * len(accuracies)
    }
    
    return results

def cms_array(matrices):
    """
    Process matrices into feature arrays by extracting lower triangular elements.
    """
    channel_0 = matrices[:, 0, :, :]
    channel_1 = matrices[:, 1, :, :]
    channel_2 = matrices[:, 2, :, :]
    
    channel_0_array = matrices_lower_triangles(channel_0)
    channel_1_array = matrices_lower_triangles(channel_1)
    channel_2_array = matrices_lower_triangles(channel_2)
    
    return np.array(channel_0_array), np.array(channel_1_array), np.array(channel_2_array), np.hstack((channel_0_array, channel_1_array, channel_2_array))

def matrices_lower_triangles(matrices):
    """
    Extract lower triangular elements from each matrix in the dataset.
    """
    return [matrix[np.tril_indices(matrix.shape[0], k=-1)] for matrix in matrices]

if __name__ == '__main__':
    import utils_common
    
    subject_range, experiment_range, selected_feature, dataset = range(1, 16), range(1, 4), 'PLV', 'SEED'
    labels = utils_common.read_labels(dataset)
    
    for sub in subject_range:
        for ex in experiment_range:
            identifier = f'sub{sub}ex{ex}'
            cms = utils_common.load_cms(dataset, experiment=identifier, feature=selected_feature)
            
            alpha, beta, gamma, data = cms_array(cms)
            
            # Perform 5-fold cross-validation and save results
            results = svm_5fold_cross_validation(data, labels, identifier=identifier)
            
            save_results(results, f"SVM_{selected_feature}_results.csv")
