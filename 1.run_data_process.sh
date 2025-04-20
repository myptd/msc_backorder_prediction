#!/bin/bash

## process dataset1
python bin/preprocessor.py --train data/data1/Kaggle_Training_Dataset_v2.csv.gz --test data/data1/Kaggle_Test_Dataset_v2.csv.gz --outdir data/processed_data1
