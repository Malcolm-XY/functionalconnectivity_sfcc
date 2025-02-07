# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 21:05:47 2025

@author: 18307
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import classification_report, accuracy_score

def preprocess_labels(labels):
    """
    Convert labels to {-1, 1} for SVM compatibility.
    """
    unique_classes = np.unique(labels)
    if len(unique_classes) != 2:
        raise ValueError("This implementation only supports binary classification.")
    return np.where(labels == unique_classes[0], -1, 1)

def create_svm_model(input_dim):
    """
    Create a simple linear SVM model in TensorFlow.
    """
    model = keras.Sequential([
        layers.Dense(1, activation='linear', input_shape=(input_dim,))
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.01),
        loss='hinge',  # Hinge loss for SVM
        metrics=['accuracy']
    )
    return model

def train_svm_tf(data, labels, partitioning='sequential', rate=0.7, epochs=50, batch_size=32):
    """
    Train an SVM model using TensorFlow.
    """
    # Validate parameters
    if partitioning not in ['sequential', 'randomized']:
        raise ValueError("Partitioning must be 'sequential' or 'randomized'.")
    if not (0 < rate < 1):
        raise ValueError("Rate must be a float between 0 and 1.")

    data = np.array(data, dtype=np.float32)
    labels = preprocess_labels(np.array(labels))
    
    # Split data
    if partitioning == 'sequential':
        split_index = int(rate * len(data))
        data_train, data_test = data[:split_index], data[split_index:]
        labels_train, labels_test = labels[:split_index], labels[split_index:]
    else:
        indices = np.arange(len(data))
        np.random.shuffle(indices)
        split_index = int(rate * len(data))
        data_train, data_test = data[indices[:split_index]], data[indices[split_index:]]
        labels_train, labels_test = labels[indices[:split_index]], labels[indices[split_index:]]
    
    # Create and train the model
    model = create_svm_model(input_dim=data.shape[1])
    model.fit(data_train, labels_train, epochs=epochs, batch_size=batch_size, verbose=1)
    
    # Evaluate the model
    labels_pred = model.predict(data_test).flatten()
    labels_pred = np.where(labels_pred >= 0, 1, -1)
    
    accuracy = accuracy_score(labels_test, labels_pred)
    report = classification_report(labels_test, labels_pred, output_dict=True)
    
    result_entry = {
        "Accuracy": accuracy,
        "Class_F1_Scores": {f"Class_{key}": value['f1-score'] for key, value in report.items() if key.isdigit()},
        "Detailed_Report": report
    }
    
    # Output results
    print("Classification Report:")
    print(classification_report(labels_test, labels_pred))
    print(f"Accuracy: {accuracy:.2f}\n")
    
    return result_entry

if __name__ == '__main__':
    import utils_common
    
    # %% Example usage
    # sample_experiment, sample_feature = 'sub1ex1', 'PCC'
    # cms = utils_common.load_cms_seed(sample_experiment, sample_feature)
    # data = cms.reshape(cms.shape[0], -1)
    # labels = utils_common.read_labels_seed()
    
    # result_entry_sequential = train_svm_tf(data, labels)
    # result_entry_randomized = train_svm_tf(data, labels, partitioning='randomized')
    
    # %% 
    subject_range, experiment_range, selected_feature, dataset = range(1, 16), range(1, 4), 'PCC', 'SEED'
    
    result_entry_randomized = []
    labels = utils_common.read_labels(dataset)
    for sub in subject_range:
        for ex in experiment_range:
            identifier = f'sub{sub}ex{ex}'
            cms = utils_common.load_cms(dataset, identifier, selected_feature)
            data = cms.reshape(cms.shape[0], -1)
            
            result_entry_randomized_temp = train_svm_tf(data, labels, partitioning='randomized')
            result_entry_randomized.append(result_entry_randomized_temp)