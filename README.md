# IQVIA CHPA Delisted Products Detection

**Organization:** IQVIA DS China

## Introduction

This project aims to predict the delisted products in the future. The delisting products prediction could increase the CHPA report quality. Predictions are created using machine learning methods, specifically, XGBoost and LightGBM. 

Data used in this project is from the IQVIA CHPA database. The training data includes sales-in data from May 2018 to May 2023, and several other variables representing medication characteristics. The outcome is a binary variable indicating whether the medication is delisted in the future.

## Usage

1. **Clone the repository**
```
git clone https://github.com/tzStatistician/IQVIA_DS_China_GDG_Delisting.git
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

5. **Modify .bat file**
Modify command lines in batch file should be modified to meet your task.