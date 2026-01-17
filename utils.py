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
from sklearn.metrics import (accuracy_score, classification_report, roc_auc_score, confusion_matrix, average_precision_score, log_loss, mean_squared_error)
from sklearn.linear_model import LogisticRegression
from config import NUM_THREADS, DEVICE, Z_DIM, BATCH_SIZE, D_HIDDEN

NUM_LOOPS = 5
EPOCHS_SURROGATE = 5
EPOCHS_AE = 5
EPOCHS_IMPROVE = 5

NUMERIC_COLS = [
        'HAS_NOMINATION_ADDED', 'BRANCH_CODE', 'ISPMJDY_ACCOUNT', 'IS_SCSS_ACCOUNT',
        'IS_PMJJY_ACCOUNT', 'IS_CHEQUE_ALLOWED', 'IS_CHEQUEBOOK_AVAILED', 'CHEQUEBOOK_COUNT',
        'IS_PAN_LINKED', 'IS_AADHAAR', 'IS_GST_LINKED', 'IS_PASSPORT_LINKED', 'IS_VOTER_ID_LINKED',
        'KYC_RATING', 'HAS_NETBANKING', 'HAS_MOBILE_BANKING', 'HAS_UPI', 'HAS_DEMAT_ACCOUNT',
        'HAS_MUTUAL_FUND', 'HAS_MOBILE_PASSBOOK', 'HAS_FASTAG', 'IS_SUKANYA_SAMRUDHI_ACCOUNT',
        'IS_PPF_LINKED', 'PPF_AMOUNT', 'HAS_LOCKER', 'HAS_ATM_CARD', 'HAS_CREDIT_CARD',
        'CREDIT_CARDS_COUNT', 'IS_ACCOUNT_FROZEN', 'FREEZE_REASON_CODE', 'IS_KYC_COMPLIANT',
        'BRANCH_PIN', 'IS_RURAL_BRANCH', 'IS_HOTSPOT', 'IS_APY_ACCOUNT', 'HAS_UDHYAM',
        'LIEN_AMOUNT', 'FREEZE_COUNT', 'TXN_AMOUNT_MEAN', 'TXN_AMOUNT_STDDEV', 'LAST_TXN_AMOUNT',
        'TXN_AMOUNT_Z_SCORE', 'TXN_AMOUNT_MEAN_7D', 'TXN_AMOUNT_STDDEV_7D', 'LAST_TXN_AMOUNT_7D',
        'TXN_AMOUNT_Z_SCORE_7D', 'TXN_AMOUNT_MEAN_14D', 'TXN_AMOUNT_STDDEV_14D',
        'LAST_TXN_AMOUNT_14D', 'TXN_AMOUNT_Z_SCORE_14D', 'ACCOUNT_AGE_DAYS', 'DAYS_SINCE_LAST_TXN',
        'UNIQUE_COUNTERPARTY_COUNT_30D', 'NIGHT_TXN_RATIO', 'SERVICE_ADOPTION_SCORE',
        'KYC_COMPLETENESS_SCORE', 'IS_FRAUD'
    ]

def load_data(filepath):
    try:
        valid_cols = set(NUMERIC_COLS)
        df = pd.read_csv(filepath, header=0, usecols=lambda c: c in valid_cols, dtype=np.float32)
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
        sys.exit(0)

    df = df.fillna(0)

    drop_cols = [
        'IS_FRAUD', 'IS_ACCOUNT_FROZEN', 'FREEZE_REASON_CODE', 'FREEZE_COUNT',
        'LAST_TXN_AMOUNT', 'TXN_AMOUNT_Z_SCORE_14D', 'TXN_AMOUNT_STDDEV',
        'TXN_AMOUNT_STDDEV_14D', 'TXN_AMOUNT_MEAN', 'TXN_AMOUNT_Z_SCORE_7D',
        'TXN_AMOUNT_MEAN_7D', 'TXN_AMOUNT_STDDEV_7D', 'TXN_AMOUNT_Z_SCORE',
        'LAST_TXN_AMOUNT_7D', 'LAST_TXN_AMOUNT_14D'
    ]
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['IS_FRAUD'].values
    return X, y


def load_data_tejas(filepath):
    try:
        df = pd.read_csv(filepath, header=0,  dtype=np.float32)
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
        sys.exit(0)

    drop_cols = ['TARGET']
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['TARGET'].values
    return X, y


def load_data_credit(filepath):
    try:
        df = pd.read_csv(filepath, header=0,  dtype=np.float32)
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
        sys.exit(0)

    drop_cols = ['Class']
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df['Class'].values
    return X, y


def load_data_custom(filepath, target):
    try:
        df = pd.read_csv(filepath, header=0,  dtype=np.float32)
    except FileNotFoundError:
        print(f"Error: '{filepath}' not found.")
        sys.exit(0)

    drop_cols = [target]
    X = df.drop(columns=drop_cols, errors='ignore')
    y = df[target].values
    return X, y

def standardize(X, use=None):
    if use is not None:
        stats_source = use
    else:
        stats_source = X

    mean = np.nanmean(stats_source, axis=0)
    s = np.nanstd(stats_source, axis=0)
    s[s == 0] = 1
    X_scaled = (X - mean) / s

    # ✅ CRITICAL FIX: Replace leftover NaNs with 0
    # This prevents the "Loss: nan" error on MPS/GPU
    X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

    return pd.DataFrame(X_scaled, columns=X.columns)




# ===============================


def evaluate_state(stage_name, X_test, y_test, xgb_teacher, embedder, surrogate=None, decoder=None):
    stats = {"Stage": stage_name}
    device = next(embedder.parameters()).device # Get current device

    if decoder:
        embedder.eval()
        decoder.eval()
        tensor_x = torch.FloatTensor(X_test).to(device)
        with torch.no_grad():
            z = embedder(tensor_x)
            x_hat = decoder(z)
            recon_loss = nn.MSELoss()(x_hat, tensor_x).item()
        stats["Reconstruction MSE"] = f"{recon_loss:.5f}"
    else:
        stats["Reconstruction MSE"] = "N/A"

    if surrogate:
        embedder.eval()
        surrogate.eval()
        tensor_x = torch.FloatTensor(X_test).to(device)
        with torch.no_grad():
            z = embedder(tensor_x)
            student_preds = surrogate(z).cpu().numpy().flatten()

        auc = roc_auc_score(y_test, student_preds)
        stats["Student AUC"] = f"{auc:.4f}"

        dtest = xgb.DMatrix(X_test, nthread=8)
        teacher_preds = xgb_teacher.predict(dtest)
        distillation_mse = mean_squared_error(teacher_preds, student_preds)
        stats["Teacher–Surrogate MSE"] = f"{distillation_mse:.5f}"
    else:
        stats["Student AUC"] = "N/A"
        stats["Teacher–Surrogate MSE"] = "N/A"

    return stats


def get_z_features(embedder, X_np):
    """ Extracts Z vectors using parallel processing for CPU """
    embedder.eval()
    all_z = []
    
    # Process in larger chunks for CPU efficiency
    chunk_size = 20000
    with torch.no_grad():
        for i in range(0, len(X_np), chunk_size):
            chunk = X_np[i:i + chunk_size]
            tensor_x = torch.FloatTensor(chunk).to(DEVICE)
            z = embedder(tensor_x)
            all_z.append(z.cpu().numpy())
    
    return np.vstack(all_z)
