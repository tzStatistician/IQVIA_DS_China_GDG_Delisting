# Delisted Products Detection
# Group: IQVIA DS China
# Contributor: Tianxiao Zhang, Zhichong Ni
# Version: 1.0
# Date: 09/06/2023

import json
import pandas as pd
from sklearn.metrics import auc, precision_recall_fscore_support, roc_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from utils.model import evaluate_model, plot_roc, plot_shap
from utils.preprocessing import data_loader, data_outputer, data_preprocesser, data_resampler
from imblearn.over_sampling import SMOTE
############################################## Step 1 - Load Data and Resample ##############################################
# Load data
raw = data_loader('result.csv')
print(raw.shape)
print("Raw data loaded.")

# Resample
resampled = data_resampler(raw, 'delist_tab', random_state=816, resample_number=1000)
print(resampled.shape)
print("Data resampled. Number of non-delisted products: ", resampled.shape[0] - resampled['delist_tab'].sum())

############################################## Step 2 - Preprocess Data ##############################################
# categorical_vars is the name list of categorical variables in the raw dataset (with no feature engineering)
categorical_vars = ['atc1', 'mnflg', 'vbp_flag', 'VBP_Batch', 'VBP_time', 'NRDL', 'NRDL_LIMTT', 'ENTRY_TIME']
processed = data_preprocesser(categorical_vars, resampled, 'delist_tab')
print(processed.shape)
print("Data preprocessed.")

# Save processed data
data_outputer(processed, 'result_processed.csv')
print("Data saved.")

############################################## Step 3 - Model Training and Evaluation ##############################################
df = pd.read_csv("path_to_your_data.csv")
X = df.drop('delist_tab', axis=1)
y = df['delist_tab']
precision_list = []
recall_list = []
f1_list = []
tpr_list = []
fpr_list = []
roc_auc_list = []
y_real = []
y_proba = []
with open('config.json', 'r') as f:
    config = json.load(f)
for random_state in range(10):
    X_train_main, X_test_main, y_train_main, y_test_main = train_test_split(X, y, test_size=0.2, random_state=random_state)
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    for train_index, test_index in kf.split(X_train_main, y_train_main):
        X_train, X_test = X_train_main.iloc[train_index], X_train_main.iloc[test_index]
        y_train, y_test = y_train_main.iloc[train_index], y_train_main.iloc[test_index]
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        for model_name in config["models"]:
            best_model = GridSearchCV(X_train_res, y_train_res, model_name)
            evaluate_model(best_model, X_test, y_test)
            y_pred = best_model.predict(X_test)
            precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            y_score = best_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_score)
            roc_auc = auc(fpr, tpr)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
            roc_auc_list.append(roc_auc)
            y_real.append(y_test)
            y_proba.append(y_score)
plot_roc(fpr_list, tpr_list, roc_auc_list)
plot_shap(best_model, X)
