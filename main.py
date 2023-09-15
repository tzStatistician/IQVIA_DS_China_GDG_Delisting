import json
import pandas as pd

from utils.model import create_model_output_folders, perform_grid_search_cv, save_cv_results
from utils.preprocessing import get_json_path, data_loader, preprocess_data, save_processed_data

######################################## Step 1 - Load Data ########################################
config_path = get_json_path('raw_minmax_xgb_test.json')
print("JSON Path successfully loaded.")

# Load the configuration file
with open(config_path, 'r') as json_file:
    config = json.load(json_file)

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

X, y = preprocess_data(raw_df, 'delist_tab', categorical_var_names, scaling_method)
print("Data successfully preprocessed.")
print("Predictors shape is:", X.shape)
print("Outcome shape is:", y.shape)

# Save the processed data
save_processed_data(pd.concat([X, y], axis=1), 'processed_data_test')
print("Processed data successfully saved.")

######################################## Step 3 - Perform Grid Search CV for Hyperparameter Tuning ########################################
# Create model output folders
create_model_output_folders(config)
print("Model output folders successfully created.")

# Get the model name
model_name = config['model']['name']
print("Model name is", model_name)

# Perform grid search CV
best_estimator, cv_results = perform_grid_search_cv(X, y, config)
print("Best hyperparameters combinations are", best_estimator)

# Save the CV results
cv_results_df = pd.DataFrame.from_dict(cv_results)
save_cv_results(cv_results_df, config)