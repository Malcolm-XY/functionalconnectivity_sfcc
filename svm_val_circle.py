# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:15:08 2024

@author: usouu
"""

import os
import pandas as pd

import utils_common
import svm_validation


# Loop through subjects and exercises
# Initialize a list to store all results
results = []

# Loop through subjects and exercises
for subject in range(1, 2):
    for experiment in range(1, 4):
        identifier = f"sub{subject}ex{experiment}"
        print(f"Processing: {identifier}")
        
        # get labels
        labels = utils_common.read_labels_seed()
        # get connectivity matrices
        cms = utils_common.load_cms_seed(experiment=identifier, feature='PCC')
        data = cms.reshape(cms.shape[0], -1)
        
        # Evaluate using the SVM evaluation function
        result_entry = svm_validation.svm_evaluation_single(data, labels, 'sequential', 0.7)
        
        # Ensure 'Identifier' is the first column by creating a new ordered dictionary
        result_entry = {"Identifier": identifier, **result_entry}
        
        # Store the result
        results.append(result_entry)

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
