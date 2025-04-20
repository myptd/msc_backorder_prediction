#!/bin/bash

data='smote'
learning_rates=(1e-3 5e-4 1e-4 5e-5 1e-5 1e-6 5e-6)  # Array of learning rates to test

for lr in "${learning_rates[@]}"; do
    
    # Train MLP model with current learning rate
    python bin/train_dnn.py --data_path data/processed_data1/${data}.npz \
        --model_type mlp \
        --learning_rate ${lr} \
        --logdir output_training/train_dnn_phase2_dataset1_${data}

    # Train Conv model with current learning rate
    python bin/train_dnn.py --data_path data/processed_data1/${data}.npz \
        --model_type conv \
        --learning_rate ${lr} \
        --logdir output_training/train_dnn_phase2_dataset1_${data}

done

