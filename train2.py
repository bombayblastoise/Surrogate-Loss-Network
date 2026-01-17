import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models import Surrogate, Embedder, Decoder
from config import Z_DIM, D_HIDDEN, BATCH_SIZE, DEVICE, NUM_THREADS

# We can use huge batches because 24GB RAM is plenty
NUM_LOOPS = 10
EPOCHS_SURROGATE = 8
EPOCHS_AE = 5
EPOCHS_IMPROVE = 10
def freeze(model):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()


def unfreeze(model):
    for p in model.parameters():
        p.requires_grad = True
    model.train()


def build_embedder(input_dim):
    return Embedder(input_dim, D_HIDDEN, Z_DIM).to(DEVICE)

def build_surrogate():
    return Surrogate(Z_DIM, D_HIDDEN, 1).to(DEVICE)

def build_autoencoder(input_dim):
    embedder = build_embedder(input_dim)
    decoder = Decoder(Z_DIM, D_HIDDEN, input_dim).to(DEVICE)
    return embedder, decoder

def get_xgb_preds(xgb_model, X):
    return xgb_model.predict(xgb.DMatrix(X))

def iterate_minibatches(tensor_x, tensor_y=None):
    n = len(tensor_x)
    indices = torch.randperm(n)

    for start in range(0, n, BATCH_SIZE):
        idx = indices[start:start+BATCH_SIZE]
        if tensor_y is None:
            yield tensor_x[idx]
        else:
            yield tensor_x[idx], tensor_y[idx]



def centroid_separation_loss(z, y, margin=1.0, eps=1e-6):
    """
    Fast O(B) contrastive-style loss for fraud
    z: [B, D]
    y: [B] or [B,1] with 0/1 labels
    """

    y = y.view(-1)

    fraud_mask = (y == 1)
    legit_mask = (y == 0)

    # skip batch if only one class present
    if fraud_mask.sum() == 0 or legit_mask.sum() == 0:
        return torch.tensor(0.0, device=z.device)

    z_fraud = z[fraud_mask]
    z_legit = z[legit_mask]

    fraud_centroid = z_fraud.mean(dim=0)
    legit_centroid = z_legit.mean(dim=0)

    # pull fraud together
    pull_loss = ((z_fraud - fraud_centroid) ** 2).mean()

    # push fraud away from legit
    centroid_dist = torch.norm(fraud_centroid - legit_centroid, p=2)
    push_loss = torch.relu(margin - centroid_dist) ** 2

    return pull_loss + push_loss


def variance_loss(z, eps=1e-4):
    std = torch.sqrt(z.var(dim=0) + eps)
    return torch.mean(torch.relu(1.0 - std))



def train_xgb(X_train, y_train, X_test, y_test):
    print("\n" + "="*50)
    print("xgboost training")
    print("="*50)

    xgb_train = xgb.DMatrix(X_train, label=y_train, nthread=NUM_THREADS)
    xgb_test = xgb.DMatrix(X_test, label=y_test, nthread=NUM_THREADS)

    ratio = float(np.sum(y_train == 0)) / np.sum(y_train == 1)

    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'learning_rate': 0.05,
        'nthread': NUM_THREADS,
        'scale_pos_weight': ratio,
        'eval_metric': 'auc',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'max_delta_step': 1,
        'min_child_weight': 5,
        'gamma': 0.1,
        'tree_method': "hist"
    }

    xgb_model = xgb.train(params=params, dtrain=xgb_train, num_boost_round=50)

    return xgb_model

def initialize_embedder(X_train, input_dim):
    embedder, decoder = build_autoencoder(input_dim)

    x = torch.tensor(X_train, device=DEVICE)
    opt = optim.Adam(list(embedder.parameters()) + list(decoder.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(EPOCHS_AE):
        for xb in iterate_minibatches(x):
            z = embedder(xb)
            loss = loss_fn(decoder(z), xb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    freeze(decoder)

    return embedder


def train_surrogate(Z, teacher_labels):
    surrogate = build_surrogate()
    unfreeze(surrogate)

    opt = optim.Adam(surrogate.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    z = torch.tensor(Z, device=DEVICE)
    y = torch.tensor(teacher_labels, device=DEVICE).unsqueeze(1)

    for _ in range(EPOCHS_SURROGATE):
        for zb, yb in iterate_minibatches(z, y):
            loss = loss_fn(surrogate(zb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    freeze(surrogate)
    return surrogate


def movement_loss(z, z_prev):
    return torch.mean(torch.norm(z - z_prev.detach(), dim=1))


def improve_embedder(X, y_true, embedder, surrogate):
    freeze(surrogate)
    unfreeze(embedder)

    opt = optim.Adam(embedder.parameters(), lr=3e-4)
    loss_cls = nn.BCEWithLogitsLoss()

    x = torch.tensor(X, device=DEVICE)
    y = torch.tensor(y_true, device=DEVICE).unsqueeze(1)

    for _ in range(EPOCHS_IMPROVE):
        for xb, yb in iterate_minibatches(x, y):
            z = embedder(xb)
            z = z + 0.05 * torch.randn_like(z)
            preds = surrogate(z)
            preds = torch.clamp(preds, -10, 10)

            loss = loss_cls(preds, yb) + 0.1 * variance_loss(z)


            if yb.sum() > 1 and (yb == 0).sum() > 1:
              loss += 1.0 * centroid_separation_loss(z, yb)


            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(embedder.parameters(), 1.0)
            opt.step()

    freeze(embedder)
    return embedder
