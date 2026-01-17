import matplotlib
matplotlib.use("Agg")

# Import config FIRST to set up threads globally
from config import Z_DIM, D_HIDDEN, DEVICE, NUM_THREADS

import xgboost as xgb
import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import torch.optim as optim
import torch.nn as nn
from models import Decoder, Embedder
from train2 import initialize_embedder, train_surrogate, train_xgb, improve_embedder, freeze
from utils import standardize, load_data_custom, get_z_features
from analysis import final_embedding_analysis, representation_report, plot_embedding, plot_reconstruction

NUM_LOOPS = 5
EPOCHS_SURROGATE = 5
EPOCHS_AE = 5
EPOCHS_IMPROVE = 10

# Thread configuration already set in config.py
torch.set_num_threads(NUM_THREADS)

if __name__ == "__main__":
    fpath = 'data/tejas.csv'
    print(f"Loading data from {fpath}")
    raw_X, raw_y = load_data_custom(fpath, 'TARGET')

    raw_X_train, raw_X_test, y_train, y_test = train_test_split(
        raw_X, raw_y, random_state=42, test_size=0.2, shuffle=True, stratify=raw_y
    )

    std_X_train = standardize(raw_X_train)
    std_X_test = standardize(raw_X_test, raw_X_train)

    X_train_np = std_X_train.values
    X_test_np = std_X_test.values
    input_dim = X_train_np.shape[1]

    # train xgb on raw X
    xgb_teacher = train_xgb(X_train_np, y_train, X_test_np, y_test)

    dtest = xgb.DMatrix(X_test_np, nthread=8)
    teacher_preds = xgb_teacher.predict(dtest)
    teacher_auc = roc_auc_score(y_test, teacher_preds)
    print(f" Teacher (XGBoost) Test AUC: {teacher_auc:.4f}")

    # initialize models
    embedder = initialize_embedder(X_train_np, input_dim)
    ae_embedder = initialize_embedder(X_train_np, input_dim)
    Z_ae_test = get_z_features(ae_embedder, X_test_np)

    # we make this to ensure embedder doesnt give garbage
    decoder = Decoder(Z_DIM, D_HIDDEN, input_dim).to(DEVICE)

    # training
    print("\n" + "="*50)
    print(f"🔄 STARTING TRAINING LOOP ({NUM_LOOPS} Cycles)")
    print("="*50)

    surrogate = None

    Z_prev_train = X_train_np
    Z_prev_test = X_test_np


    for i in range(NUM_LOOPS):


        print("torch.cuda.memory_allocated: %fGB"%(torch.cuda.memory_allocated(0)/1024/1024/1024))
        print("torch.cuda.memory_reserved: %fGB"%(torch.cuda.memory_reserved(0)/1024/1024/1024))
        print("torch.cuda.max_memory_reserved: %fGB"%(torch.cuda.max_memory_reserved(0)/1024/1024/1024))
        print(f"\n--- Cycle {i+1} / {NUM_LOOPS} ---")

        if surrogate is not None:
            freeze(surrogate)


        # 1. Current embeddings
        Z_curr_train = get_z_features(embedder, X_train_np)
        Z_curr_test  = get_z_features(embedder, X_test_np)

        # 2. Teacher input = [Z_{k-1} | Z_k]
        X_train_xgb = np.hstack([Z_prev_train, Z_curr_train])
        X_test_xgb  = np.hstack([Z_prev_test,  Z_curr_test])

        # 3. TRAIN XGBOOST (On Augmented Data)
        print(f"     Training XGBoost on {X_train_xgb.shape[1]} features...")
        xgb_teacher = train_xgb(X_train_xgb, y_train, X_test_xgb, y_test)
        dtest = xgb.DMatrix(X_test_xgb, nthread=8)
        xgb_auc = roc_auc_score(y_test, xgb_teacher.predict(dtest))
        print(f"     XGBoost AUC: {xgb_auc:.4f}")

        # 4. Teacher labels (FULL VIEW)
        teacher_labels = xgb_teacher.predict(xgb.DMatrix(X_train_xgb))

        # 5. TRAIN SURROGATE
        # Input: Z_{k} | Target: New XGB Labels
        surrogate = train_surrogate(Z_curr_train, teacher_labels)

        # 6. IMPROVE EMBEDDER
        # Input: Raw Data | Target: New XGB Labels
        embedder = improve_embedder(X_train_np, y_train, embedder, surrogate)

        # 7. Store Zk for next round
        Z_prev_train = Z_curr_train.copy()
        Z_prev_test  = Z_curr_test.copy()


        # 8. stats
        representation_report(
            cycle=i+1,
            embedder=embedder,
            surrogate=surrogate,
            xgb_teacher=xgb_teacher,
            Z_prev_train=Z_prev_train,
            Z_curr_train=Z_curr_train,
            Z_curr_test=Z_curr_test,
            X_test_xgb=X_test_xgb,
            y_train=y_train,
            y_test=y_test,
            device=DEVICE
        )



    # --- 5. REPORT ---
    print("\n" + "="*80)
    print(f"{'TRAINING PROGRESS REPORT':^80}")
    print("="*80)
    print(f"{'Stage':<25} | {'Distill MSE':<15} | {'Student AUC':<10}")
    print("-" * 80)

    print(f"{'0. XGBoost Teacher':<25} | {'0.00000':<15} | {teacher_auc:.4f}")
    # --- 6. ANALYSIS ---
    final_embedding_analysis(
    embedder=embedder,
    decoder=decoder,
    X_train=X_train_np,
    X_test=X_test_np,
    y_train=y_train,
    y_test=y_test,
    device=DEVICE
)

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    import numpy as np

    def linear_probe_auc(Z, y):
        clf = LogisticRegression(max_iter=3000)
        clf.fit(Z, y)
        return roc_auc_score(y, clf.predict_proba(Z)[:, 1])


    def centroid_distance(Z, y):
        fraud = Z[y == 1]
        legit = Z[y == 0]
        if len(fraud) == 0 or len(legit) == 0:
            return 0.0
        return np.linalg.norm(fraud.mean(0) - legit.mean(0))


    # ============================================================
    # FINAL EMBEDDING COMPARISON (CORRECT & FAIR)
    # ============================================================

    from sklearn.decomposition import PCA
    from sklearn.metrics import mean_squared_error
    from sklearn.linear_model import LogisticRegression
    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    os.makedirs("plots", exist_ok=True)

    # -------------------------
    # Helper metrics
    # -------------------------
    def linear_probe_auc(Z, y):
        clf = LogisticRegression(max_iter=3000)
        clf.fit(Z, y)
        return roc_auc_score(y, clf.predict_proba(Z)[:, 1])

    def centroid_distance(Z, y):
        f = Z[y == 1]
        l = Z[y == 0]
        return np.linalg.norm(f.mean(0) - l.mean(0))

    # -------------------------
    # Helper plots
    # -------------------------
    def scatter_plot(Z2, y, name):
        plt.figure(figsize=(4,3))
        plt.scatter(Z2[:,0], Z2[:,1], c=y, s=6, cmap="coolwarm")
        plt.tight_layout()
        plt.savefig(f"plots/{name}.png", dpi=150)
        plt.close()

    # ============================================================
    # 1️⃣ YOUR EMBEDDING (with post-hoc decoder)
    # ============================================================

    Z_yours = get_z_features(embedder, X_test_np)

    print("\nYOUR EMBEDDING")
    print("AUC:", linear_probe_auc(Z_yours, y_test))
    print("Centroid:", centroid_distance(Z_yours, y_test))

    # ---- Train decoder ONLY for your embedding
    task_decoder = Decoder(Z_DIM, D_HIDDEN, input_dim).to(DEVICE)
    for p in embedder.parameters(): p.requires_grad = False

    opt = optim.Adam(task_decoder.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train_np, device=DEVICE)
    for _ in range(5):
        for i in range(0, len(X_train_t), 2048):
            xb = X_train_t[i:i+2048]
            with torch.no_grad():
                z = embedder(xb)
            loss = loss_fn(task_decoder(z), xb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    with torch.no_grad():
        X_hat = task_decoder(torch.tensor(Z_yours, device=DEVICE)).cpu().numpy()

    print("Recon MSE:", mean_squared_error(X_test_np, X_hat))

    # PCA projection plot
    scatter_plot(PCA(2).fit_transform(Z_yours), y_test, "yours_pca")

    # Reconstruction scatter
    f = np.random.randint(0, X_test_np.shape[1])
    plt.figure(figsize=(4,3))
    plt.scatter(X_test_np[:,f], X_hat[:,f], s=5)
    plt.tight_layout()
    plt.savefig("plots/yours_recon.png", dpi=150)
    plt.close()

    # ============================================================
    # 3️⃣ AUTOENCODER BASELINE (own encoder + own decoder)
    # ============================================================

    ae_embedder = Embedder(input_dim, D_HIDDEN, Z_DIM).to(DEVICE)
    ae_decoder  = Decoder(Z_DIM, D_HIDDEN, input_dim).to(DEVICE)

    opt = optim.Adam(list(ae_embedder.parameters()) + list(ae_decoder.parameters()), lr=1e-3)
    loss_fn = nn.MSELoss()

    X_train_t = torch.tensor(X_train_np, device=DEVICE)
    for _ in range(5):
        for i in range(0, len(X_train_t), 2048):
            xb = X_train_t[i:i+2048]
            z = ae_embedder(xb)
            loss = loss_fn(ae_decoder(z), xb)
            opt.zero_grad()
            loss.backward()
            opt.step()

    Z_ae = ae_embedder(torch.tensor(X_test_np, device=DEVICE)).detach().cpu().numpy()
    X_hat_ae = ae_decoder(torch.tensor(Z_ae, device=DEVICE)).detach().cpu().numpy()

    print("\nAUTOENCODER BASELINE")
    print("AUC:", linear_probe_auc(Z_ae, y_test))
    print("Centroid:", centroid_distance(Z_ae, y_test))
    print("Recon MSE:", mean_squared_error(X_test_np, X_hat_ae))

    scatter_plot(PCA(2).fit_transform(Z_ae), y_test, "ae_pca")

    f = np.random.randint(0, X_test_np.shape[1])
    plt.figure(figsize=(4,3))
    plt.scatter(X_test_np[:,f], X_hat_ae[:,f], s=5)
    plt.tight_layout()
    plt.savefig("plots/ae_recon.png", dpi=150)
    plt.close()

