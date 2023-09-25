# IQVIA CHPA Delisted Products Detection

**Organization:** IQVIA DS China

## Introduction

This project aims to predict the delisted products in the future China Medication Market. The delisting products prediction could increase the CHPA report quality. Predictions are created using machine learning methods, specifically, XGBoost and LightGBM. 

Data used in this project is from the IQVIA CHPA database. The training data includes sales-in data of various medication from May 2018 to May 2023, and several other variables representing medication and pharmaceutical company characteristics. The outcome in this project is a binary variable indicating whether the medication is delisted in the future.

## Usage

1. **Clone the repository**
```
git clone https://github.com/tzStatistician/IQVIA_DS_China_GDG_Delisting.git
```

2. **Set up virtual environment**

```
pip install virtualenv 
python -m venv venv_name
venv_name\Scripts\activate
deactivate
```

2. **Install all required packages**

```
pip install -r requirements.txt
```

3. **Change path in code**

Line 16 in main.py

```
parser.add_argument('--config', type=str, required=True, help=r'path_to_your_setting_folder')
```

4. **Modify JSON settings**

Add/Modify/Delete .josn files to implement experiments by configurations you set.

5. **Modify .bat file**

Modify command lines in batch file "multi_settings_training.bat" to meet your task.

6. **Execute**

```
cd path_to_your_project_folder
venv_name\Scripts\activate
multi_settings_training.bat
```