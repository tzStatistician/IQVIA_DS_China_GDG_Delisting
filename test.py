import json
import os
import pandas as pd
import numpy as np

from sklearn.metrics import auc, precision_recall_fscore_support, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE

from utils.model import create_model_output_folders, perform_grid_search_cv
from utils.preprocessing import get_json_path, data_loader, preprocess_data, save_processed_data

# data_outputer, data_preprocesser, data_resampler

# Step 1 - Load Data
config_path = get_json_path('raw_minmax_xgb_test.json')
print("JSON Path successfully loaded.")

# Load the configuration file
with open(config_path, 'r') as json_file:
    config = json.load(json_file)

raw_df = data_loader(config['data']['filename'])
print("Raw data successfully loaded.")
print("The shape of raw data is: ", raw_df.shape)

# Preprocess the data
categorical_var_names = config['data']['categorical_variables']
scaling_method = config['preprocessing']['scaling_method']

print("Categorical variables are", categorical_var_names)
print("Scaling method is", scaling_method)

X, y = preprocess_data(raw_df, 'delist_tab', categorical_var_names, scaling_method)
print("Data successfully preprocessed.")
print("Predictors shape is:", X.shape)
print("Outcome shape is:", y.shape)

# Save the processed data
save_processed_data(pd.concat([X, y], axis=1), 'processed_data_test')
print("Processed data successfully saved.")

# Create model output folders
create_model_output_folders(config)
print("Model output folders successfully created.")

# Get the model name
model_name = config['model']['name']
print("Model name is", model_name)

best_estimator, cv_results = perform_grid_search_cv(X, y, config)
print(best_estimator)
cv_results_df = pd.DataFrame.from_dict(cv_results)

# Set up a loop to iterate over different random states to create different training and testing sets
for random_state in range(2):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=random_state, stratify=y)

    # Perform grid search CV for hyperparameter tuning
    best_estimator = perform_grid_search_cv(X_train, y_train, config_path)

    # Calculate and save scores and SHAP values for each fold during the grid search CV process
    calculate_save_scores_shap(best_estimator, X_train, y_train, config_path)

    # Create and save plots for the best estimator
    create_plots_for_best_estimator(best_estimator, X_train, y_train, config_path)

    # Evaluate the model performance on the testing set
    y_pred = best_estimator.predict(X_test)

    # Calculate precision, recall, and F1-score for the testing set
    precision = precision_score(y_test, y_pred, average='binary')
    recall = recall_score(y_test, y_pred, average='binary')
    f1 = f1_score(y_test, y_pred, average='binary')

    # Save the scores for the testing set in a CSV file
    test_scores_df = pd.DataFrame({
        'Precision': [precision],
        'Recall': [recall],
        'F1_Score': [f1]
    })

    # Create a unique filename for the testing set scores CSV file
    test_scores_filename = f'test_scores_random_state_{random_state}.csv'

    # Define the path to save the testing set scores CSV file
    test_scores_filepath = os.path.join('output', model_name, 'score', test_scores_filename)

    # Save the testing set scores CSV file
    test_scores_df.to_csv(test_scores_filepath, index=False)
