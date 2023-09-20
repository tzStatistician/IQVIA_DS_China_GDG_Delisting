# Basic module 
import os

from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin

# model selection module
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, accuracy_score

# Function to test & create model output folders
def create_model_output_folders(config):
    model_name = config['model']['name']
    model_output_path = os.path.join('output', model_name)

    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
        os.makedirs(os.path.join(model_output_path, 'score'))
        os.makedirs(os.path.join(model_output_path, 'plot'))

# Define a NULL oversampler
class NullOversampler(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X

# Function to perform grid search CV for hyperparameter tuning
def perform_grid_search_cv(X, y, X_cat_encoded_name, config):
    # Read config file
    model_name = config['model']['name']
    hyperparameters_grid = config['model']['hyperparameters_grid']
    numerical_vars_names = X.columns.difference(X_cat_encoded_name)
    scaling_method = config['preprocessing']['scaling_method']

    # Define model
    if model_name == 'XGBoost':
        model = xgb.XGBClassifier(random_state=816, verbosity=0)
    elif model_name == 'RandomForest': 
        model = RandomForestClassifier(random_state=816, verbose=0)
    elif model_name == 'LGB':
        model = lgb.LGBMClassifier(random_state=816, verbosity=0)

    oversampling_method = config['model']['oversampling']

    # Define oversampler
    if oversampling_method == 'SMOTE':
        oversampler = SMOTE(random_state=816)
    elif oversampling_method == 'ADASYN':
        oversampler = ADASYN(random_state=816)
    elif oversampling_method == 'BorderlineSMOTE':
        oversampler = BorderlineSMOTE(random_state=816)
    else :
        oversampler = NullOversampler()

    # Define scaler
    if scaling_method == 'MinMaxScaler':
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    # Define a ColumnTransformer object
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', scaler, numerical_vars_names),
            ('cat', 'passthrough', X_cat_encoded_name)
            ])
    # Define pipeline
    pipe = Pipeline([('Processpr', preprocessor),
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

    # Use GridSearchCV
    grid_search = GridSearchCV(pipe, hyperparameters_grid, cv=5, scoring=scorer, refit= 'f1_1', verbose=2, n_jobs=-1)
    grid_search.fit(X, y)

    best_estimator = grid_search.best_estimator_
    cv_results = grid_search.cv_results_

    return best_estimator, cv_results

def save_cv_results(cv_results_df, config):
    # find correct directory
    df_name = config['data']['filename'][0:3]
    scaling_method = config['preprocessing']['scaling_method']
    oversampling_method = config['model']['oversampling']
    model_name = config['model']['name']

    # Define filename
    filename = f'{df_name}_{scaling_method}_{oversampling_method}_{model_name}.csv'
    
    model_output_path = os.path.join('output', model_name)
    score_output_path = os.path.join(model_output_path, 'score')

    # save cv_results_df
    cv_results_df.to_csv(os.path.join(score_output_path, filename), index=False)

    # Create a smaller output file
    cv_results_df_clean = cv_results_df[["mean_fit_time", "params", "mean_test_f1_1", "std_test_f1_1", "rank_test_f1_1", "mean_test_recall_1", "std_test_recall_1", "rank_test_recall_1", "mean_test_f1_mean", "std_test_f1_mean", "rank_test_f1_mean"]]
    cv_results_df_clean_name = f'{df_name}_{scaling_method}_{oversampling_method}_{model_name}_clean.csv'
    cv_results_df_clean.to_csv(os.path.join(score_output_path, cv_results_df_clean_name), index=False)