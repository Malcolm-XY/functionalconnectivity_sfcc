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

def cms_array(matrices_dict):
    """
    Process matrices into feature arrays by extracting lower triangular elements.
    """
    alpha = matrices_dict['alpha']
    beta = matrices_dict['beta']
    gamma = matrices_dict['gamma']
    
    alpha_array = matrices_lower_triangles(alpha)
    beta_array = matrices_lower_triangles(beta)
    gamma_array = matrices_lower_triangles(gamma)
    
    return np.array(alpha_array), np.array(beta_array), np.array(gamma_array), np.hstack((alpha_array, beta_array, gamma_array))

def matrices_lower_triangles(matrices):
    """
    Extract lower triangular elements from each matrix in the dataset.
    """
    return [matrix[np.tril_indices(matrix.shape[0], k=-1)] for matrix in matrices]

# %% Example Usage
import utils_feature_loading
fcs_h5 = utils_feature_loading.read_fcs('seed', 'sub1ex1', 'pcc')
alpha, beta, gamma, data = cms_array(fcs_h5)

fcs_mat = utils_feature_loading.read_fcs_mat('seed', 'sub1ex1', 'pcc', 'joint')
alpha, beta, gamma, data = cms_array(fcs_mat)

# import utils_basic_reading
# path_grandparent = os.path.abspath(os.path.join(os.getcwd(), "../.."))
# dataset = 'SEED'
# feature = 'pcc'
# identifier = 'sub1ex1'
# path_file = os.path.join(path_grandparent, 'Research_Data', dataset, 'functional connectivity', f'{feature}_mat', f'{identifier}.mat')
# fcs_mat = utils_basic_reading.read_mat(path_file)

# %% Usage
if __name__ == '__main__':
    import utils_feature_loading
    
    subject_range, experiment_range, selected_feature, dataset = range(1, 2), range(1, 4), 'PCC', 'SEED'
    labels = np.reshape(utils_feature_loading.read_labels(dataset), -1)
    
    for sub in subject_range:
        for ex in experiment_range:
            identifier = f'sub{sub}ex{ex}'
            
            fcs = utils_feature_loading.read_fcs_mat(dataset, selected_feature, identifier)
            
            alpha, beta, gamma, data = cms_array(fcs)
            
            # Perform 5-fold cross-validation and save results
            results = svm_5fold_cross_validation(data, labels, identifier=identifier)
            
            save_results(results, f"SVM_{selected_feature}_results.csv")
