#!/bin/bash

data='smote'

for ckpt in $(ls output_training/train_dnn_phase2_dataset1_${data}/checkpoints/*.ckpt); do
    # Evaluate MLP model with current learning rate
    python bin/eval_dnn.py --model_path $ckpt \
        --data_path data/processed_data1/${data}.npz \
        --save output_training/eval_dnn_phase2_dataset1_${data}
done


