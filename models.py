import torch
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib
import sys
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, confusion_matrix, average_precision_score, log_loss)
from sklearn.linear_model import LogisticRegression

class Surrogate(nn.Module):
  def __init__(self, d_z, d_hidden, d_out = 1):
    super().__init__()
    self.fc1 = nn.Linear(d_z , d_hidden)
    self.fc2 = nn.Linear(d_hidden, d_out)
    self.relu = nn.ReLU()

  def forward(self, z):
    # x = torch.cat([z, y], dim=1)
    return self.fc2(self.relu(self.fc1(z)))

class Embedder(nn.Module):
  def __init__(self, d_in, d_hidden, d_out):
    super().__init__()
    self.fc1 = nn.Linear(d_in, d_hidden)
    self.fc2 = nn.Linear(d_hidden, d_out)
    self.relu = nn.ReLU()

  def forward(self, x):
    return self.fc2(self.relu(self.fc1(x)))


class Decoder(nn.Module):
    def __init__(self, d_z, d_hidden, d_out):
        super().__init__()
        self.fc1 = nn.Linear(d_z, d_hidden)
        self.fc2 = nn.Linear(d_hidden, d_out)
        self.relu = nn.ReLU()

    def forward(self, z):
        return self.fc2(self.relu(self.fc1(z)))
