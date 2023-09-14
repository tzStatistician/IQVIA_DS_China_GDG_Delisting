# Basic module 
import numpy as np
import json
import os
import pandas as pd

# model selection module
from sklearn.model_selection import StratifiedKFold, train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, accuracy_score, roc_curve, auc
import shap

################################################### test v1 date: 09/12/2023 ###################################################

# Function to test & create model output folders
def create_model_output_folders(config):
    model_name = config['model']['name']
    model_output_path = os.path.join('output', model_name)

    if not os.path.exists(model_output_path):
        os.makedirs(model_output_path)
        os.makedirs(os.path.join(model_output_path, 'score'))
        os.makedirs(os.path.join(model_output_path, 'plot'))


# Function to perform grid search CV for hyperparameter tuning
def perform_grid_search_cv(X, y, config):
    model_name = config['model']['name']
    hyperparameters_grid = config['model']['hyperparameters_grid']

    if model_name == 'XGBoost':
        model = xgb.XGBClassifier(random_state=816)
    else: 
        model = RandomForestClassifier(random_state=816)

    # Define pipeline
    pipe = Pipeline([('Oversampling', SMOTE(random_state=816)), 
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
    grid_search = GridSearchCV(pipe, hyperparameters_grid, cv=5, scoring=scorer, refit='f1_1', return_train_score=True, verbose=1, n_jobs=-1)
    grid_search.fit(X, y)

    best_estimator = grid_search.best_estimator
    cv_results = grid_search.cv_results_

    return best_estimator, cv_results


def evaluate_best_estimator(best_estimator, X, y):
    cv = StratifiedKFold(n_splits=5)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    metrics = {'f1_score': [], 'precision': [], 'recall': [], 'accuracy': []}

    for train, test in cv.split(X, y):
        smote = SMOTE(random_state=816)
        X_train_res, y_train_res = smote.fit_resample(X[train], y[train])
        
        y_pred = best_estimator.predict(X[test])
        y_score = best_estimator.predict_proba(X[test])[:, 1]
        
        metrics['f1_score'].append(f1_score(y[test], y_pred))
        metrics['precision'].append(precision_score(y[test], y_pred))
        metrics['recall'].append(recall_score(y[test], y_pred))
        metrics['accuracy'].append(accuracy_score(y[test], y_pred))

        fpr, tpr, thresholds = roc_curve(y[test], y_score)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

    # Plot ROC curve
    plt.figure(figsize=(10, 10))
    for i in range(5):
        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')

    # Calculate mean of metrics
    metrics_mean = {key: np.mean(value) for key, value in metrics.items()}

    return best_estimator, metrics_mean