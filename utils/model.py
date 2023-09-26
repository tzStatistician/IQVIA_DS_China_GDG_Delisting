# Basic module 
import os
import numpy as np
import joblib
import json

from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# Model selection modules
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Model evaluation modules
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, accuracy_score

############################ Function to test & create model output folders ############################
def create_model_output_folders(config):
    model_name = config['model']['name']
    output_path = os.path.join('output', model_name)
    output_score_path = os.path.join(output_path, 'score')
    output_plot_path = os.path.join(output_path, 'plot')
    output_be_path = os.path.join(output_path, 'best_estimator')
    output_validation_path = os.path.join(output_path, 'validation')

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(output_score_path):
        os.makedirs(output_score_path)
    if not os.path.exists(output_plot_path):
        os.makedirs(output_plot_path)
    if not os.path.exists(output_be_path):
        os.makedirs(output_be_path)
    if not os.path.exists(output_validation_path):
        os.makedirs(output_validation_path)

############################ Define a NULL oversampler ############################
class NullOversampler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X

 
############################ Define grid search function ############################
def perform_grid_search_cv(X, y, X_cat_encoded_name, config):
    # Read config file
    model_name = config['model']['name']
    hyperparameters_grid = config['model']['hyperparameters_grid']
    numerical_vars_names = X.columns.difference(X_cat_encoded_name)
    scaling_method = config['preprocessing']['scaling_method']
    oversampling_method = config['model']['oversampling']
    search_method = config['model']['search_method']
    candidates = config['model']['candidates']

    # Define model
    if model_name == 'XGBoost':
        model = xgb.XGBClassifier(random_state=816, verbosity=0)
    elif model_name == 'RandomForest':
        model = RandomForestClassifier(random_state=816, verbose=0)
    elif model_name == 'LGB':
        model = lgb.LGBMClassifier(random_state=816, verbosity=0)
    else:
        raise ValueError("Please specify a valid model name.")

    # Define oversampler
    if oversampling_method == 'SMOTE':
        oversampler = SMOTE(random_state=816)
    elif oversampling_method == 'ADASYN':
        oversampler = ADASYN(random_state=816)
    elif oversampling_method == 'BorderlineSMOTE':
        oversampler = BorderlineSMOTE(random_state=816)
    elif oversampling_method == 'NullOversampler':
        oversampler = NullOversampler()
    else:
        raise ValueError("Please specify a valid oversampling method.")

    # Define scaler
    if scaling_method == 'MinMaxScaler':
        scaler = MinMaxScaler()
    elif scaling_method == 'StandardScaler':
        scaler = StandardScaler()
    else:
        raise ValueError("Please specify a valid scaling method.")

    # Define a ColumnTransformer object
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numerical_vars_names),
            ('cat', 'passthrough', X_cat_encoded_name)
        ])

    # Define pipeline
    pipe = Pipeline([('Preprocessor', preprocessor),
                     ('Oversampling', oversampler),
                     ('Model', model)])

    # Define scorer
    scorer = {
        'f1_0': make_scorer(f1_score, labels=[0]),
        'precision_0': make_scorer(precision_score, labels=[0]),
        'recall_0': make_scorer(recall_score, labels=[0]),
        'f1_1': make_scorer(f1_score, labels=[1]),
        'precision_1': make_scorer(precision_score, labels=[1]),
        'recall_1': make_scorer(recall_score, labels=[1]),
        'accuracy': make_scorer(accuracy_score),
        'f1_mean': make_scorer(f1_score, average='macro'),
        'precision_mean': make_scorer(precision_score, average='macro'),
        'recall_mean': make_scorer(recall_score, average='macro')
    }

    # Define custom refit callable function
    def custom_refit(cv_results_):
        recall_threshold = 0.8

        # Extracting the recall and F1-score for the positive class (label 1)
        recall_scores = cv_results_['mean_test_recall_1']
        f1_scores = cv_results_['mean_test_f1_1']

        # Identifying the parameter combinations where recall > 0.8
        valid_indices = np.where(recall_scores > recall_threshold)[0]

        # Finding the index with the highest F1-score among the valid indices
        if len(valid_indices) > 0:
            best_index = valid_indices[np.argmax(f1_scores[valid_indices])]
        else:
            best_index = np.argmax(f1_scores)

        return best_index

    # Use GridSearchCV
    if search_method == "GridSearchCV":
        search = GridSearchCV(pipe, hyperparameters_grid, cv=5, scoring=scorer, refit=custom_refit, verbose=3, n_jobs=-1)
    elif search_method == "RandomizedSearchCV":
        search = RandomizedSearchCV(pipe, hyperparameters_grid, n_iter=candidates, cv=5, scoring=scorer, refit=custom_refit, verbose=3, n_jobs=-1)
    else:
        raise ValueError("Please specify a valid search method.")
    
    search.fit(X, y)

    best_estimator = search.best_estimator_
    best_params = search.best_params_
    cv_results = search.cv_results_

    return best_estimator, best_params, cv_results

############################ Define function to save cv_results ############################
def save_cv_results(cv_results_df, config):
    # find correct directory
    df_name = config['data']['filename'][0:3]
    scaling_method = config['preprocessing']['scaling_method']
    oversampling_method = config['model']['oversampling']
    search_method = config['model']['search_method']
    model_name = config['model']['name']
    hyperparameters_section = config['model']['hyperparameters_section']

    # Define filename
    filename = f'{df_name}_{scaling_method}_{oversampling_method}_{search_method}_{model_name}_{hyperparameters_section}.csv'
    
    model_output_path = os.path.join('output', model_name)
    score_output_path = os.path.join(model_output_path, 'score')

    # save cv_results_df
    cv_results_df.to_csv(os.path.join(score_output_path, filename), index=False)

    # Create a smaller output file
    cv_results_df_clean = cv_results_df[["mean_fit_time", "params", "mean_test_f1_1", "std_test_f1_1", "rank_test_f1_1", "mean_test_recall_1", "std_test_recall_1", "rank_test_recall_1", "mean_test_f1_mean", "std_test_f1_mean", "rank_test_f1_mean"]]
    cv_results_df_clean_name = f'{df_name}_{scaling_method}_{oversampling_method}_{model_name}_{hyperparameters_section}_clean.csv'
    cv_results_df_clean.to_csv(os.path.join(score_output_path, cv_results_df_clean_name), index=False)

############################ Define fuction to save best estimator ############################
def save_best_estimator(best_estimator, best_params, config):
    # find correct directory
    df_name = config['data']['filename'][0:3]
    scaling_method = config['preprocessing']['scaling_method']
    oversampling_method = config['model']['oversampling']
    search_method = config['model']['search_method']
    model_name = config['model']['name']
    hyperparameters_section = config['model']['hyperparameters_section']
    # Define filename
    filename = f'{df_name}_{scaling_method}_{oversampling_method}_{search_method}_{model_name}_{hyperparameters_section}.pkl'
    param_name = f'{df_name}_{scaling_method}_{oversampling_method}_{search_method}_{model_name}_{hyperparameters_section}_params.json'

    model_output_be_path = os.path.join('output', model_name, 'best_estimator')

    # save best_estimator
    joblib.dump(best_estimator, os.path.join(model_output_be_path, filename))

    # Save best_params
    with open(os.path.join(model_output_be_path, param_name), 'w') as f:
        f.write(json.dumps(best_params))