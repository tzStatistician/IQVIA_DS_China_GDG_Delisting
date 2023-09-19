import argparse
import json
import pandas as pd

from utils.model import create_model_output_folders, perform_grid_search_cv, save_cv_results
from utils.preprocessing import data_loader, preprocess_data, save_processed_data

import warnings

warnings.filterwarnings("ignore")
######################################## Step 1 - Load Data ########################################
# Set up the argument parser
parser = argparse.ArgumentParser(description='Perform grid search CV for hyperparameter tuning.')
parser.add_argument('--config', type=str, required=True, help=r'C:\Users\u1158100\Documents\test\IQVIA_DS_China_GDG_Delisting\setting')
args = parser.parse_args()

# Load the configuration file
with open(args.config, 'r') as f:
    config = json.load(f)

# Load data
raw_df = data_loader(config['data']['filename'])
print("Raw data successfully loaded.")
print("The shape of raw data is: ", raw_df.shape)

######################################## Step 2 - Preprocess the data ########################################
# Get the categorical variable names and scaling method
categorical_var_names = config['data']['categorical_variables']
scaling_method = config['preprocessing']['scaling_method']

print("Categorical variables are", categorical_var_names)
print("Scaling method is", scaling_method)

X, y, X_cat_encoded_name = preprocess_data(raw_df, 'delist_tab', categorical_var_names)
print("Data successfully preprocessed.")
print("Predictors shape is:", X.shape)
print("Outcome shape is:", y.shape)

# Save the processed data
save_processed_data(config, pd.concat([X, y], axis=1))
print("Processed data successfully saved.")

####################################### Step 3 - Perform Grid Search CV for Hyperparameter Tuning ########################################
create_model_output_folders(config)
print("Model output folders successfully created.")

# Get the model name
model_name = config['model']['name']
print("Model name is", model_name)

# Perform grid search CV
best_estimator, cv_results = perform_grid_search_cv(X, y, X_cat_encoded_name, config)
print("Best hyperparameters combinations are", best_estimator)

# Save the CV results
cv_results_df = pd.DataFrame.from_dict(cv_results)
save_cv_results(cv_results_df, config)

