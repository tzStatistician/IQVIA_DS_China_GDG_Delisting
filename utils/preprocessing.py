import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler

# define a function to get json file path by json file name
def get_json_path(json_file_name):
    project_folder = os.getcwd()
    json_path = os.path.join(project_folder, 'setting', json_file_name)
    return json_path


# Read raw data based on file name
def data_loader(file_name):
    project_folder = os.getcwd()
    file_path = os.path.join(project_folder, 'raw_data', file_name)
    return pd.read_csv(file_path)

# Resample data keeping all delisted products and randomly sample non-delisted products (resample_number = number of non-delisted products)
def data_resampler(raw_data, outcome_name, random_state, resample_number):
    positive_data = raw_data[raw_data[outcome_name] == 1]
    negative_sample = raw_data[raw_data[outcome_name] == 0].sample(n=resample_number, random_state=random_state)
    return pd.concat([positive_data, negative_sample], axis=0).reset_index(drop=True)

# Preprocess data: Fill missing values with 0 and normalize (!!!tune for standardscaler!!) numerical variables 
# Preprocess data: OneHot-encode categorical variables
def data_preprocesser(categorical_vars, resampled_data, outcome_name):
    numerical_vars = list(set(resampled_data.columns) - set(categorical_vars) - {outcome_name})
    resampled_data[numerical_vars] = resampled_data[numerical_vars].fillna(0)
    resampled_data[numerical_vars] = MinMaxScaler().fit_transform(resampled_data[numerical_vars])

    for col in categorical_vars:
        resampled_data[col] = LabelEncoder().fit_transform(resampled_data[col])

    resampled_data = pd.get_dummies(resampled_data, columns=categorical_vars, drop_first=True)
    return resampled_data

# Output processed data to processed folder
def data_outputer(dataframe, output_file_name):
    project_folder = os.getcwd()
    output_path = os.path.join(project_folder, 'processed', output_file_name)
    dataframe.to_csv(output_path, index=False)

################################################################### test v1 JSON+ ###################################################################
def preprocess_data(data, target_var_name, categorical_var_names, scaling_method):
    # Separate target variable and input variables
    y = data[[target_var_name]]
    X = data.drop(columns=[target_var_name])

    # Separate numerical and categorical variables
    numerical_vars = X.columns.difference(categorical_var_names)
    
    X_numerical = X[numerical_vars]
    X_categorical = X[categorical_var_names]

    # Handle missing values for numerical variables
    X_numerical.fillna(0, inplace=True)

    # Scale numerical variables
    if scaling_method == 'MinMaxScaler':
        scaler = MinMaxScaler()
        X_numerical = pd.DataFrame(scaler.fit_transform(X_numerical), columns=numerical_vars)

    # Encode categorical variables
    if not X_categorical.empty:
        for col in categorical_var_names:
            X_categorical[col] = LabelEncoder().fit_transform(X_categorical[col])

        X_categorical = pd.get_dummies(X_categorical, columns=categorical_var_names, drop_first=True, dtype=int)

    # Concatenate numerical and categorical variables
    X = pd.concat([X_numerical, X_categorical], axis=1)
    
    return X, y

def save_processed_data(data, filename):
    output_filepath = os.path.join('processed', filename + '.csv')
    data.to_csv(output_filepath, index=False)