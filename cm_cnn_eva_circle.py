# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 18:41:40 2024

@author: usouu
"""

import os
import pandas as pd
import torch

import utils
import models
import cnn_evaluation

def k_fold_evaluation_circle(model, subject_range, experiment_range, feature, band):
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Device: {device}')
    model = model.to(device)

    # Loop through subjects and exercises
    # Initialize a list to store all results
    results = []

    # Loop through subjects and exercises
    for subject in subject_range:
        for experiment in experiment_range:
            identifier = f"sub{subject}ex{experiment}"
            print(f"Processing: {identifier}")
            
            # Get labels
            labels = utils.get_label()
            # Get cm data
            cmdata = utils.load_cmdata2d(feature, band, identifier)
            
            # Evaluate using the CNN evaluation function
            result_entry = cnn_evaluation.k_fold_evaluation(model, cmdata, labels, k_folds=5, batch_size=128)
            
            # Ensure 'Identifier' is the first column by creating a new ordered dictionary
            result_entry = {"Identifier": identifier, **result_entry}
            
            # Store the result
            results.append(result_entry)

    return results

def save2xlsx(results):
    # Define the Excel file path
    excel_file = "results.xlsx"

    # Check if the Excel file exists and load existing data if it does
    if os.path.exists(excel_file):
        existing_df = pd.read_excel(excel_file)
        new_data_df = pd.DataFrame([{key: value if key != "Class_F1_Scores" else str(value) for key, value in entry.items()} for entry in results])
        combined_df = pd.concat([existing_df, new_data_df], ignore_index=True)
    else:
        combined_df = pd.DataFrame([{key: value if key != "Class_F1_Scores" else str(value) for key, value in entry.items()} for entry in results])

    # Save results to Excel
    combined_df.to_excel(excel_file, index=False)

    print("Processing complete. Results saved to 'results.xlsx'.")

model = models.CNN2DModel_3ch()
results=k_fold_evaluation_circle(model, range(2,3), range(1,2), 'PLV', 'joint')
save2xlsx(results)