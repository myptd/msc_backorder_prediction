#!/bin/bash

export SCRIPT="python bin/baseline_ml.py --data data/processed_data1/{2}.npz --model {1} --save output_training/phase1_dataset1"

# parallel -j7 echo "Processing: Model={1}, Dataset={2}..." "&&" $SCRIPT "&&" echo "Done: Model={1}, Dataset={2}" ::: AdaBoost RF LR XGBoost SVM ::: rus original adasyn smote smote_tomek

parallel -j7 echo "Processing: Model={1}, Dataset={2}..." "&&" $SCRIPT "&&" echo "Done: Model={1}, Dataset={2}" ::: AdaBoost RF LR XGBoost ::: rus original adasyn smote smote_tomek

echo "All processes completed."
