#!/usr/bin/env python
# coding: utf-8

import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.nn import functional as F


def load_data(path, random_state=2025):
    data = np.load(path)
    X_train, X_val, y_train, y_val = train_test_split(
        data['X_train'], data['y_train'], test_size=0.2, random_state=random_state
    )
    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(data['X_test'], dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(-1),
        torch.tensor(data['y_test'], dtype=torch.float32).unsqueeze(-1)
    )

def create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, batch_size, drop_last):
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=drop_last, num_workers=2, pin_memory=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=batch_size, shuffle=False, drop_last=False, num_workers=2)
    test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)
    return train_loader, val_loader, test_loader

class MLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 22), nn.ReLU(),
            nn.Linear(22, 16), nn.ReLU(),
            nn.Linear(16, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)



class ConvModel(nn.Module):
    def __init__(self, input_features=21):
        super(ConvModel, self).__init__()
        
        # First Conv1D block
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.5)
        
        # Second Conv1D block
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.5)
        #
        flattened_size = 640
        # Dense layers
        self.fc1 = nn.Linear(flattened_size, 256)
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout4 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 1)  # Output layer
        
    def forward(self, x):
        # x is expected to have shape (batch_size, input_features)
        # Reshape to (batch_size, 1, input_features) for Conv1D
        x = x.unsqueeze(1)
        
        # First Conv1D block
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Second Conv1D block
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout2(x)
        # Flatten
        x = torch.flatten(x, 1)
        
        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout3(x)
        x = F.relu(self.fc2(x))
        x = self.dropout4(x)
        x = torch.sigmoid(self.fc3(x))
        
        return x


class MasterLightningModel(pl.LightningModule):
    def __init__(self, input_dim, learning_rate, pos_weight= None, model_type='mlp'):
        super().__init__()
        self.save_hyperparameters()
        if model_type == 'mlp':
            self.model = MLP(input_dim=input_dim)
        elif model_type == 'conv':
            self.model = ConvModel(input_features=input_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if pos_weight is None:
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, dtype=torch.float32))

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss = self.loss_fn(self(x), y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.learning_rate)



def main(args):
    X_train, X_val, X_test, y_train, y_val, y_test = load_data(args.data_path)
    train_loader, val_loader, test_loader = create_dataloaders(X_train, X_val, X_test, y_train, y_val, y_test, args.batch_size, args.drop_last)
    
    # pos_weight = (y_train.sum() / (len(y_train) - y_train.sum())).item()
    # print(f"Positive weight: {pos_weight}")


    model = MasterLightningModel(input_dim=args.input_dim, learning_rate=args.learning_rate, model_type=args.model_type)
    
    logger = pl.loggers.TensorBoardLogger(save_dir=args.logdir, name=f'{args.model_type}_{args.learning_rate}')

    checkpoint_callback = ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', dirpath=f'{args.logdir}/checkpoints/', filename=f'best_model_{args.model_type}_{args.learning_rate}')

    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=args.patience, mode='min')
    
    trainer = Trainer(logger=logger, max_epochs=args.epochs, callbacks=[early_stopping_callback, checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--input_dim', type=int, default=21, help='Input feature dimension')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=2048, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    parser.add_argument('--drop_last', action='store_true', help='Drop last batch if incomplete')
    parser.add_argument('--logdir', type=str, default='logdir', help='Directory for logs and checkpoints')
    parser.add_argument('--model_type', type=str, default='mlp', choices=['mlp', 'conv'], help='Model type: mlp or conv')
    args = parser.parse_args()
    main(args)


# python bin/train_dnn.py --data_path data_processed/data1/smote.npz --model_type conv --learning_rate 1e-4 --logdir logdir

# python bin/train_dnn.py --data_path data_processed/data1/smote.npz --model_type mlp --learning_rate 1e-4 --logdir logdir