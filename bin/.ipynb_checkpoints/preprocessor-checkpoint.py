#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import ADASYN   
import argparse


SEED = 2025

def load_data(path):
    
    """
    Load dataset and process it by:
    - removing sku column
    - reordering columns such that:
        [0:15) are numerical features,
        [15:20) are binary features, 
        and [20] is the target

    """

    new_cols = ['national_inv', 'lead_time', 'in_transit_qty', 'forecast_3_month',
       'forecast_6_month', 'forecast_9_month', 'sales_1_month',
       'sales_3_month', 'sales_6_month', 'sales_9_month', 'min_bank',
       'pieces_past_due', 'perf_6_month_avg',
       'perf_12_month_avg', 'local_bo_qty', 'potential_issue', 'deck_risk', 'oe_constraint',
       'ppap_risk', 'stop_auto_buy', 'rev_stop', 'went_on_backorder']
    
    data = pd.read_csv(path)
    if 'sku' in data.columns:
        data = data.drop(columns=['sku'])
    
    data = data.reindex(columns=new_cols)
    data = data.apply(lambda x: x.astype(float) if x.dtype == 'int64' else x)

    return data



def preprocess_data(train_data, test_data):
    """
    Preprocess the dataset by:
    - Removing rows with missing values.
    - Encoding categorical variables.
    - Scaling numerical features.
    """
    # Remove rows with missing values
    train_data = train_data.dropna()
    test_data = test_data.dropna()

    # Get column indices for numerical columns
    numerical_cols = train_data.select_dtypes(include=['float64', 'int64']).columns
    numerical_indices = [train_data.columns.get_loc(col) for col in numerical_cols]
    
    # Normalize numerical features
    scaler = StandardScaler()
    train_data.iloc[:, numerical_indices] = scaler.fit_transform(train_data.iloc[:, numerical_indices])
    test_data.iloc[:, numerical_indices] = scaler.transform(test_data.iloc[:, numerical_indices])  # Use the same scaler

    # Get column indices for object (categorical) columns
    categorical_cols = train_data.select_dtypes(include=['object']).columns
    categorical_indices = [train_data.columns.get_loc(col) for col in categorical_cols]
    
    # Encode categorical variables
    for idx in categorical_indices:
        # Map 'No' to 0 and 'Yes' to 1, convert to int
        train_data.iloc[:, idx] = train_data.iloc[:, idx].map({'No': 0, 'Yes': 1}).astype(int)
        test_data.iloc[:, idx] = test_data.iloc[:, idx].map({'No': 0, 'Yes': 1}).astype(int)
    
    
    return train_data.astype('float').values, test_data.astype('float').values, train_data.columns




# Handle Imbalance: Oversampling with SMOTE
def oversample_smote(X, y):
    """
    Apply Synthetic Minority Oversampling Technique (SMOTE) to balance the dataset.
    """
    smote = SMOTE(random_state=SEED)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Handle Imbalance: Undersampling with RandomUnderSampler
def undersample_random(X, y):
    """
    Apply Random UnderSampling to balance the dataset.
    """
    rus = RandomUnderSampler(random_state=SEED)
    X_resampled, y_resampled = rus.fit_resample(X, y)
    return X_resampled, y_resampled

# Handle Imbalance: Combination (SMOTE + Tomek Links)
def combine_smote_tomek(X, y):
    """
    Apply a combination of SMOTE and Tomek Links to balance the dataset.
    """
    smote_tomek = SMOTETomek(random_state=SEED)
    X_resampled, y_resampled = smote_tomek.fit_resample(X, y)
    return X_resampled, y_resampled

# Handle Imbalance: ADASYN
def combine_adasyn(X, y):
    """
    Apply ADASYN (Adaptive Synthetic Sampling) to balance the dataset.
    """
    adasyn = ADASYN(random_state=SEED)
    X_resampled, y_resampled = adasyn.fit_resample(X, y)
    return X_resampled, y_resampled



def save_dataset(datasets, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for k in datasets.keys():
        X_train, X_test, X_rus, y_train, y_test, y_rus = datasets[k]

        np.savez(f'{save_dir}/{k}.npz', X_train=X_train, X_test=X_test, X_rus=X_rus, y_train=y_train, y_test=y_test, y_rus=y_rus)
    
def save_col_names(col_names, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'{save_dir}/col_names.txt', 'w') as f:
        for col in col_names:
            f.write(f"{col}\n")





if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process train, test, and output file paths.")
    parser.add_argument('--train', required=True, help='Path to the training dataset')
    parser.add_argument('--test', required=True, help='Path to the testing dataset')
    parser.add_argument('--outdir', required=True, help='Path to save the output')

    args = parser.parse_args()

    train_data = load_data(args.train)
    test_data = load_data(args.test)

    train_data, test_data, col_names = preprocess_data(train_data, test_data)

    X_train = train_data[:, :-1]  
    y_train = train_data[:, -1].astype('int')   

    X_test = test_data[:, :-1]   
    y_test = test_data[:, -1].astype('int')    

    # Apply imbalance handling methods
    datasets = {}

    X_rus, y_rus = undersample_random(X_test, y_test)

    # Original dataset (no resampling)
    datasets['original'] = (X_train, X_test, X_rus, y_train, y_test, y_rus)

    # Undersampled dataset (Random UnderSampling)
    X_combined, y_combined = undersample_random(X_train, y_train)
    datasets['rus'] = (X_combined, X_test, X_rus, y_combined, y_test, y_rus)

    # Oversampled dataset (SMOTE)
    X_combined, y_combined = oversample_smote(X_train, y_train)
    datasets['smote'] = (X_combined, X_test, X_rus, y_combined, y_test, y_rus)

    # Combined method (SMOTE + Tomek Links)
    X_combined, y_combined = combine_smote_tomek(X_train, y_train)
    datasets['smote_tomek'] = (X_combined, X_test, X_rus, y_combined, y_test, y_rus)

    # Combined method (ADASYN)
    X_combined, y_combined = combine_adasyn(X_train, y_train)
    datasets['adasyn'] = (X_combined, X_test, X_rus, y_combined, y_test, y_rus)

    # Save the datasets
    save_dataset(datasets, args.outdir)
    # Save column names
    save_col_names(col_names, args.outdir)


## benchmark/preprocessor.py --train data/data1/Kaggle_Training_Dataset_v2.csv.gz --test data/data1/Kaggle_Test_Dataset_v2.csv.gz --outdir data_processed/data1

# benchmark/preprocessor.py --train data/data2/train.csv.gz --test data/data2/test.csv.gz --outdir data_processed/data2