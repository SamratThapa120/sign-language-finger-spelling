CUDA_VISIBLE_DEVICES=4 python3 ./signet/trainer/ctc_trainer.py --train-data ../dataset/folds/fold3_train.csv --validation-data ../dataset/folds/fold3_valid.csv --data-root ../dataset/npy_data --experiment-name debug