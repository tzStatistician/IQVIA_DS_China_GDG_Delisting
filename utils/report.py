import os
import joblib
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import StratifiedKFold

def best_estimator_loader(config):
    # find correct directory
    df_name = config['data']['filename'][0:3]
    scaling_method = config['preprocessing']['scaling_method']
    oversampling_method = config['model']['oversampling']
    search_method = config['model']['search_method']
    model_name = config['model']['name']
    hyperparameters_section = config['model']['hyperparameters_section']

    # Define filename
    filename = f'{df_name}_{scaling_method}_{oversampling_method}_{search_method}_{model_name}_{hyperparameters_section}.pkl'

    model_output_be_path = os.path.join('output', model_name, 'best_estimator')

    # load best_estimator
    best_estimator = joblib.load(os.path.join(model_output_be_path, filename))

    return best_estimator

def gene_test(best_estimator, config):
    # load processed data
    project_folder = os.getcwd()
    df_name = config['data']['filename'][0:3]
    filename = f'{df_name}_processed'
    processed_data_path = os.path.join(project_folder, 'processed', filename + '.csv')
    processed_data = pd.read_csv(processed_data_path)
    print("Processed data successfully loaded.")

    # 10 fold cross validation
    X = processed_data.drop('delist_tab', axis=1)
    y = processed_data['delist_tab'].values

    cv = StratifiedKFold(n_splits=10)

    # For ROC curve plotting
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    plt.figure(figsize=(10, 8))

    # For metrics calculation
    metrics_data = {
        'Fold': [],
        'F1_Score_0': [], 'F1_Score_1': [], 'F1_Score_Mean': [],
        'Precision_0': [], 'Precision_1': [], 'Precision_Mean': [],
        'Recall_0': [], 'Recall_1': [], 'Recall_Mean': [],
        'Accuracy': []
    }

    fold_num = 1
    for train, test in cv.split(X, y):
        probas_ = best_estimator.fit(X.iloc[train], y[train]).predict_proba(X.iloc[test])
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])

        # ROC curve data
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'ROC fold {fold_num} (AUC = {roc_auc:0.2f})')

        # Metrics calculation
        y_pred = best_estimator.predict(X.iloc[test])
        metrics_data['Fold'].append(fold_num)
        metrics_data['F1_Score_0'].append(f1_score(y[test], y_pred, pos_label=0))
        metrics_data['F1_Score_1'].append(f1_score(y[test], y_pred, pos_label=1))
        metrics_data['F1_Score_Mean'].append(f1_score(y[test], y_pred, average='weighted'))

        metrics_data['Precision_0'].append(precision_score(y[test], y_pred, pos_label=0))
        metrics_data['Precision_1'].append(precision_score(y[test], y_pred, pos_label=1))
        metrics_data['Precision_Mean'].append(precision_score(y[test], y_pred, average='weighted'))

        metrics_data['Recall_0'].append(recall_score(y[test], y_pred, pos_label=0))
        metrics_data['Recall_1'].append(recall_score(y[test], y_pred, pos_label=1))
        metrics_data['Recall_Mean'].append(recall_score(y[test], y_pred, average='weighted'))

        metrics_data['Accuracy'].append(accuracy_score(y[test], y_pred))

        fold_num += 1

    # Finalizing ROC curve
    ## mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:0.2f})', lw=2, alpha=0.8)
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=0.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curves for each fold')
    plt.legend(loc='lower right')

    # Save the figure
    scaling_method = config['preprocessing']['scaling_method']
    oversampling_method = config['model']['oversampling']
    search_method = config['model']['search_method']
    model_name = config['model']['name']
    hyperparameters_section = config['model']['hyperparameters_section']

    output_path = os.path.join('output', model_name)
    output_plot_path = os.path.join(output_path, 'plot')
    output_validation_path = os.path.join(output_path, 'validation')

    figure_name = f'{df_name}_{scaling_method}_{oversampling_method}_{search_method}_{model_name}_{hyperparameters_section}.png'
    vali_figure_path = os.path.join(output_plot_path, figure_name)
    plt.savefig(vali_figure_path, dpi=300)
    print("Report ROC curve figure successfully saved.")

    # Saving metrics data to CSV
    df_metrics = pd.DataFrame(metrics_data)
    metrics_data_name = f'vali_{df_name}_{scaling_method}_{oversampling_method}_{search_method}_{model_name}_{hyperparameters_section}.csv'
    df_metrics.to_csv(os.path.join(output_validation_path, metrics_data_name), index=False)
    print("Validation metrics data successfully saved.")