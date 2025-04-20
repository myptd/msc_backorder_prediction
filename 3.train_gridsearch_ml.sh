#!/bin/bash


data='rus'


model='LR'
python bin/gridsearch_ml.py --data data/processed_data1/${data}.npz --model ${model} --save output_training/phase2_dataset1_rus

model='AdaBoost'
python bin/gridsearch_ml.py --data data/processed_data1/${data}.npz --model ${model} --save output_training/phase2_dataset1_rus


model='RF'
python bin/gridsearch_ml.py --data data/processed_data1/${data}.npz --model ${model} --save output_training/phase2_dataset1_rus


model='XGBoost'
python bin/gridsearch_ml.py --data data/processed_data1/${data}.npz --model ${model} --save output_training/phase2_dataset1_rus


model='SVM'
python bin/gridsearch_ml.py --data data/processed_data1/${data}.npz --model ${model} --save output_training/phase2_dataset1_rus
