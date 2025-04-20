#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import numpy as np
import pandas as pd
import torch
from train_dnn import MasterLightningModel, load_data, MLP, ConvModel
from gridsearch_ml import optimize_f_thresholds, plot_performance_curves
from sklearn.metrics import (
    balanced_accuracy_score, f1_score, roc_auc_score, 
    precision_recall_fscore_support, classification_report, average_precision_score,
    accuracy_score, precision_score, recall_score, fbeta_score
)
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, precision_recall_fscore_support


def main():
    parser = argparse.ArgumentParser(description='Evaluate DNN model performance')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to processed data (smote.npz)')
    parser.add_argument('--save', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='mps',
                       choices=['cpu', 'mps', 'cuda'],
                       help='Device for inference')

    args = parser.parse_args()
    
    # Create output directory if not exists
    os.makedirs(args.save, exist_ok=True)

    # Load model
    model = MasterLightningModel.load_from_checkpoint(args.model_path)
    model.eval()

    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.data_path)

    # Run inference
    with torch.no_grad():
        y_test_pred = model(X_test.to(args.device))
    y_pred_proba = y_test_pred.detach().cpu().numpy()
    y_pred = y_pred_proba > 0.5

    # Compute metrics
    f1, df_f1 = optimize_f_thresholds(y_test, y_pred_proba, beta=1)
    f05, df_f05 = optimize_f_thresholds(y_test, y_pred_proba, beta=0.5)
    f2, df_f2 = optimize_f_thresholds(y_test, y_pred_proba, beta=2)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)

    # Prepare results
    model_name = os.path.basename(args.model_path).replace('.ckpt', '').replace('best_model_', '')
    flag = os.path.basename(args.data_path).split('.')[0]
    
    results = [flag, f1, f05, f2, auc_roc, auc_pr, "NA"]
    df = pd.DataFrame([results], columns=[
        "imbalance_handling", "max_f1", "max_f05", "max_f2", 
        "auc_roc", "auc_pr", "training_time_min"
    ])

    # Save outputs
    df.to_csv(os.path.join(args.save, f"{model_name}_{flag}_performance.csv"), index=False)
    df_f1.to_csv(os.path.join(args.save, f"{model_name}_{flag}_f1_optim.csv"), index=False)
    df_f05.to_csv(os.path.join(args.save, f"{model_name}_{flag}_f05_optim.csv"), index=False)
    df_f2.to_csv(os.path.join(args.save, f"{model_name}_{flag}_f2_optim.csv"), index=False)
    
    plot_performance_curves(
        y_test, y_pred_proba,
        title=f"{model_name}_{flag} Performance Curves",
        save_path=os.path.join(args.save, f"{model_name}_{flag}_performance_curves.pdf")
    )
    
    np.savez(
        os.path.join(args.save, f"{model_name}_{flag}_predictions.npz"),
        y_test=y_test, y_pred=y_pred, y_pred_proba=y_pred_proba
    )

if __name__ == "__main__":
    main()


# python bin/eval_dnn.py --model_path output_training/dnn_phase2_dataset1_smote/checkpoints/best_model_mlp_0.001.ckpt --data_path data/processed_data1/smote.npz --save output_training/eval_dnn_phase2_dataset1_smote