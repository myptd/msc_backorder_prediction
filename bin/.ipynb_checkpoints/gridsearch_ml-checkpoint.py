#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import time
import pickle
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
import xgboost as xgb

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score, 
    precision_recall_fscore_support, classification_report, average_precision_score,
    accuracy_score, precision_score, recall_score, fbeta_score
)

from sklearn.metrics import roc_curve, precision_recall_curve, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt


def load_data(file_path):
    data = np.load(file_path)
    return data['X_train'], data['X_test'], data['X_rus'], data['y_train'], data['y_test'], data['y_rus']



def save_model(model, filename, method='pickle'):
    """
    Save a scikit-learn model to a file.

    Parameters:
    model: scikit-learn model object
        The trained model to save.
    filename: str
        The file path where the model will be saved.
    method: str, optional (default='pickle')
        The serialization method to use ('pickle' or 'joblib').

    Returns:
    None
    """
    if method == 'pickle':
        with open(filename, 'wb') as f:
            pickle.dump(model, f)
    elif method == 'joblib':
        joblib.dump(model, filename)
    else:
        raise ValueError("Invalid method. Choose 'pickle' or 'joblib'.")

def load_model(filename, method='pickle'):
    """
    Load a scikit-learn model from a file.

    Parameters:
    filename: str
        The file path where the model is saved.
    method: str, optional (default='pickle')
        The serialization method used to save the model ('pickle' or 'joblib').

    Returns:
    model: scikit-learn model object
        The loaded model.
    """
    if method == 'pickle':
        with open(filename, 'rb') as f:
            return pickle.load(f)
    elif method == 'joblib':
        return joblib.load(filename)
    else:
        raise ValueError("Invalid method. Choose 'pickle' or 'joblib'.")


def optimize_f_thresholds(y_true, y_prob, beta=1, step=0.01):
    """
    Find optimal thresholds for F-beta score.
    
    Parameters:
    y_true (array-like): True binary labels
    y_prob (array-like): Predicted probabilities
    beta (float): Beta value for F-score (default: 1)
    step (float): Threshold search step size (default: 0.01)
    
    Returns:
    tuple: (max_score, pd.DataFrame)
        max_score: The maximum F-beta score
        pd.DataFrame: DataFrame with thresholds as columns and F-beta score as the only row
    """
    thresholds = np.arange(0.0, 1.01, step)
    results = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        results.append(fbeta_score(y_true, y_pred, beta=beta))

    df = pd.DataFrame([results], columns=thresholds)
    df.index = [f'F{beta}']
    
    return float(np.max(results)), df



def optimize_f_thresholds(y_true, y_prob, beta=1, step=0.01, pdf_path=None):
    """
    Find optimal thresholds for F-beta score with optional PDF output.
    
    Parameters:
    y_true (array-like): True binary labels
    y_prob (array-like): Predicted probabilities
    beta (float): Beta value for F-score (default: 1)
    step (float): Threshold search step size (default: 0.01)
    pdf_path (str): Optional path to save threshold analysis plot as PDF
    
    Returns:
    tuple: (max_score, pd.DataFrame)
        max_score: The maximum F-beta score
        pd.DataFrame: DataFrame with thresholds as columns and F-beta score as the only row
    """
    thresholds = np.arange(0.0, 1.01, step)
    results = []
    
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        results.append(fbeta_score(y_true, y_pred, beta=beta))

    df = pd.DataFrame([results], columns=thresholds)
    df.index = [f'F{beta}']
    
    # Create threshold analysis plot if PDF path is provided
    if pdf_path:
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, results, color='darkgreen', lw=2)
        
        # Mark optimal threshold
        max_idx = np.argmax(results)
        plt.scatter(thresholds[max_idx], results[max_idx], color='red', 
                   zorder=3, label=f'Optimal Threshold ({thresholds[max_idx]:.2f})')
        
        plt.xlabel('Threshold')
        plt.ylabel(f'F{beta} Score')
        plt.title(f'F{beta} Score vs. Threshold')
        plt.grid(True)
        plt.legend()
        plt.savefig(pdf_path, bbox_inches='tight')
        plt.close()
    
    return np.max(results), df


def plot_performance_curves(y_test, y_pred_proba, title='Model Performance Curves', save_path=None):
    """
    Plot ROC and Precision-Recall curves with metrics annotations and optional PDF output.
    
    Parameters:
    y_test (array-like): True binary labels
    y_pred_proba (array-like): Predicted probabilities for positive class
    title (str): Title for the plot (default: 'Model Performance Curves')
    save_path (str): Optional path to save plot as PDF
    """
    # Calculate curve data and metrics
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)

    # Create plot
    fig = plt.figure(figsize=(12, 6))
    plt.suptitle(title, y=1.02)

    # ROC Curve
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(fpr, tpr, color='darkblue', lw=2, label=f'AUC-ROC = {auc_roc:.2f}')
    ax1.plot([0, 1], [0, 1], 'k--', label='Random Baseline')
    ax1.set(xlabel='False Positive Rate', ylabel='True Positive Rate', 
           xlim=[0.0, 1.0], ylim=[0.0, 1.05])
    ax1.set_title('ROC Curve')
    ax1.legend(loc='lower right')
    ax1.grid(True)

    # PR Curve
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(recall, precision, color='darkorange', lw=2, label=f'AUC-PR = {auc_pr:.2f}')
    ax2.set(xlabel='Recall', ylabel='Precision', xlim=[0.0, 1.0], ylim=[0.0, 1.05])
    ax2.set_title('Precision-Recall Curve')
    ax2.legend(loc='best')
    ax2.grid(True)

    plt.tight_layout()
    
    # Save to PDF if requested
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    #plt.show()
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a machine learning model on dataset.")
    parser.add_argument("--data", type=str, required=True, help="Path to the NPZ data file.")
    parser.add_argument("--model", type=str, choices=['SVM', 'LR', 'AdaBoost', 'XGBoost', 'RF'], required=True, help="Model to use.")
    parser.add_argument("--save", type=str, default="./results", help="Directory to save results.")
    parser.add_argument("--seed", type=int, default=2025, help="Seeding.")

    args = parser.parse_args()
    
    ## create output directory if it does not exist
    if not os.path.exists(args.save):
        os.makedirs(args.save)

    
    ## load data
    X_train, X_test, X_rus, y_train, y_test, y_rus = load_data(args.data)

    # Compute class weights from training data
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = dict(zip(np.unique(y_train), class_weights))

    # # Get the model from the dictionary
    # models = {
    #     'SVM': SVC(random_state=args.seed, probability=True, class_weight=class_weight_dict),
    #     'LR': LogisticRegression(random_state=args.seed, class_weight=class_weight_dict),
    #     'AdaBoost': AdaBoostClassifier(random_state=args.seed),  # AdaBoost does not directly support class weights
    #     'XGBoost': xgb.XGBClassifier(random_state=args.seed, scale_pos_weight=class_weights[1] / class_weights[0]),  # Use scale_pos_weight for XGBoost
    #     'RF': RandomForestClassifier(random_state=args.seed, class_weight=class_weight_dict),
    # }


    # clf = models[args.model]
    # model_name = type(clf).__name__
    # flag = os.path.splitext(os.path.basename(args.data))[0]
    
    # ## fit the model
    # start_time = time.time()
    # clf.fit(X_train, y_train)
    # training_time = (time.time() - start_time) / 60  # Convert to minutes


    # Define parameter grids for each model

    param_grids = {
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.01, 0.1],
            'kernel': ['rbf', 'linear']
        },
        'LR': {
            'C': [0.01, 0.1, 1, 10, 100],
            'solver': ['liblinear', 'saga'],
            'max_iter': [100, 200, 500]
        },
        'AdaBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0]
        },
        'XGBoost': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 5, 7]
        },
        'RF': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    }

    # Define base models with class weights
    base_models = {
        'SVM': SVC(random_state=args.seed, probability=True, class_weight=class_weight_dict),
        'LR': LogisticRegression(random_state=args.seed, class_weight=class_weight_dict),
        'AdaBoost': AdaBoostClassifier(random_state=args.seed),
        'XGBoost': xgb.XGBClassifier(random_state=args.seed, scale_pos_weight=class_weights[1] / class_weights[0]),
        'RF': RandomForestClassifier(random_state=args.seed, class_weight=class_weight_dict),
    }

    # Get the base model and parameter grid
    base_model = base_models[args.model]
    param_grid = param_grids[args.model]

    # Create GridSearchCV object
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=5,  
        scoring='average_precision', 
        n_jobs=-1,  # Use all available cores
        verbose=1
    )

    model_name = type(base_model).__name__
    flag = os.path.splitext(os.path.basename(args.data))[0]

    ## fit the model with grid search
    start_time = time.time()
    grid_search.fit(X_train, y_train)
    training_time = (time.time() - start_time) / 60 

    # Get the best model
    clf = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Print best parameters
    print(f"Best parameters for {model_name}: {best_params}")
    print(f"Best cross-validation score: {grid_search.best_score_:.4f}")

    # save model
    save_model(clf, os.path.join(args.save, f"{model_name}_{flag}_model_weight.pkl"))
    save_model(grid_search, os.path.join(args.save, f"{model_name}_{flag}_grid_search.pkl"))

    # Predictions on test set

    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1]

    # Compute relevant metrics
    f1, df_f1 = optimize_f_thresholds(y_test, y_pred_proba, beta=1)
    f05, df_f05 = optimize_f_thresholds(y_test, y_pred_proba, beta=0.5)
    f2, df_f2 = optimize_f_thresholds(y_test, y_pred_proba, beta=2)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)

    ## save the performance of test sets
    results =[flag, f1, f05, f2, auc_roc, auc_pr, training_time]
    df = pd.DataFrame([results], columns=["imbalance_handling", "max_f1", "max_f05", "max_f2", "auc_roc", "auc_pr","training_time_min"])
    df.to_csv(os.path.join(args.save, f"{model_name}_{flag}_performance.csv"), index=False)
    
    ## save the optimized thresholds
    df_f1.to_csv(os.path.join(args.save, f"{model_name}_{flag}_f1_optim.csv"), index=False)
    df_f05.to_csv(os.path.join(args.save, f"{model_name}_{flag}_f05_optim.csv"), index=False)
    df_f2.to_csv(os.path.join(args.save, f"{model_name}_{flag}_f2_optim.csv"), index=False)

    ## plot the performance curves
    plot_performance_curves(y_test, y_pred_proba, title=f"{model_name}_{flag} Performance Curves", save_path=os.path.join(args.save, f"{model_name}_{flag}_performance_curves.pdf"))

    ## save predictions for later use
    np.savez(os.path.join(args.save, f"{model_name}_{flag}_predictions.npz"), y_test=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba)


# benchmark/eval_ml2.py --data data_processed/data2/rus.npz --model AdaBoost --save test_eval_ml2