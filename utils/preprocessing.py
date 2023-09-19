import os
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Read raw data based on file name
def data_loader(file_name):
    project_folder = os.getcwd()
    file_path = os.path.join(project_folder, 'raw_data', file_name)
    return pd.read_csv(file_path)

def preprocess_data(data, target_var_name, categorical_var_names):
    # Separate target variable and input variables
    y = data[[target_var_name]]
    X = data.drop(columns=[target_var_name])

    # Separate numerical and categorical variables
    numerical_vars = X.columns.difference(categorical_var_names)
    
    X_numerical = X[numerical_vars]
    X_categorical = X[categorical_var_names]

    # Handle missing values for numerical variables
    X_numerical.fillna(0, inplace=True)

    # Encode categorical variables
    if not X_categorical.empty:
        for col in categorical_var_names:
            X_categorical[col] = LabelEncoder().fit_transform(X_categorical[col])

        X_categorical = pd.get_dummies(X_categorical, columns=categorical_var_names, drop_first=True, dtype=int)

    X_cat_encoded_name = X_categorical.columns.tolist()
    # Concatenate numerical and categorical variables
    X = pd.concat([X_numerical, X_categorical], axis=1)
    
    return X, y, X_cat_encoded_name

def save_processed_data(config, data):
    df_name = config['data']['filename'][0:3]
    filename = f'{df_name}_processed'
    output_filepath = os.path.join('processed', filename + '.csv')
    data.to_csv(output_filepath, index=False)

# Resample data keeping all delisted products and randomly sample non-delisted products (resample_number = number of non-delisted products)
def data_resampler(raw_data, outcome_name, random_state, resample_number):
    positive_data = raw_data[raw_data[outcome_name] == 1]
    negative_sample = raw_data[raw_data[outcome_name] == 0].sample(n=resample_number, random_state=random_state)
    return pd.concat([positive_data, negative_sample], axis=0).reset_index(drop=True)

# define a function to get json file path by json file name
def get_json_path():
    project_folder = os.getcwd()
    json_path = os.path.join(project_folder, 'setting')
    return json_path