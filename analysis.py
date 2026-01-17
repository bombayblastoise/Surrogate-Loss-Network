import os
os.makedirs("plots", exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import xgboost as xgb
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.manifold import TSNE
import umap
from sklearn.linear_model import LogisticRegression
from config import DEVICE

# Ensure sklearn uses all cores
import sklearn
sklearn.config_context(assume_finite=True)


def final_embedding_analysis(
    *,
    embedder,
    decoder,
    X_train,
    X_test,
    y_train,
    y_test,
    device,
    max_points=5000,
    n_jobs=-1
):
    """Use n_jobs for parallel sklearn operations"""
    embedder.eval()
    if decoder is not None:
        decoder.eval()

    # ---------- 1. Get embeddings ----------
    with torch.no_grad():
        Z_train = embedder(torch.tensor(X_train, device=device)).cpu().numpy()
        Z_test  = embedder(torch.tensor(X_test,  device=device)).cpu().numpy()

    # subsample for plots
    idx = np.random.choice(len(Z_test), min(max_points, len(Z_test)), replace=False)
    Zp = Z_test[idx]
    yp = y_test[idx]
    Xp = X_test[idx]

    print("\n🔍 FINAL EMBEDDING ANALYSIS")
    print("=" * 60)

    # ---------- 2. Linear probe (truth test) ----------
    probe = LogisticRegression(max_iter=1000, n_jobs=n_jobs)
    probe.fit(Z_train, y_train)
    probe_preds = probe.predict_proba(Z_test)[:, 1]
    probe_auc = roc_auc_score(y_test, probe_preds)

    print(f"📈 Z Linear Probe AUC: {probe_auc:.4f}")

        # ---------- 3. PCA plot ----------
    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(Zp)

    plt.figure(figsize=(5, 4))
    plt.scatter(Z_pca[:, 0], Z_pca[:, 1], c=yp, s=6, cmap="coolwarm")
    plt.title("PCA(Z)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.savefig("plots/pca_z.png", dpi=600, bbox_inches="tight")
    plt.close()


    # ---------- 4. t-SNE ----------
    tsne = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto", n_jobs=n_jobs)
    Z_tsne = tsne.fit_transform(Zp)

    plt.figure(figsize=(5, 4))
    plt.scatter(Z_tsne[:, 0], Z_tsne[:, 1], c=yp, s=6, cmap="coolwarm")
    plt.title("t-SNE(Z)")
    plt.tight_layout()
    plt.savefig("plots/tsne_z.png", dpi=600, bbox_inches="tight")
    plt.close()


    # ---------- 5. UMAP ----------
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
    Z_umap = reducer.fit_transform(Zp)

    plt.figure(figsize=(5, 4))
    plt.scatter(Z_umap[:, 0], Z_umap[:, 1], c=yp, s=6, cmap="coolwarm")
    plt.title("UMAP(Z)")
    plt.tight_layout()
    plt.savefig("plots/umap_z.png", dpi=600, bbox_inches="tight")
    plt.close()


    # ---------- 6. Geometry check ----------
    fraud = Zp[yp == 1]
    legit = Zp[yp == 0]

    if len(fraud) > 0 and len(legit) > 0:
        c_fraud = fraud.mean(0)
        c_legit = legit.mean(0)
        centroid_dist = np.linalg.norm(c_fraud - c_legit)
        print(f"📐 Centroid distance (fraud vs legit): {centroid_dist:.4f}")


# ---------- 7. Reconstruction (what did Z keep?) ----------
    if decoder is not None:
      with torch.no_grad():
          X_hat = decoder(torch.tensor(Zp, device=device)).cpu().numpy()

      recon_mse = mean_squared_error(Xp, X_hat)
      print(f"🔁 Reconstruction MSE: {recon_mse:.4f}")

      # pick 5 random features
      num_feats = min(5, Xp.shape[1])
      feat_idx = np.random.choice(Xp.shape[1], size=num_feats, replace=False)

      for f in feat_idx:
          plt.figure(figsize=(4, 3))
          plt.scatter(Xp[:, f], X_hat[:, f], s=5, alpha=0.6)
          plt.xlabel("Original")
          plt.ylabel("Reconstructed")
          plt.title(f"Feature {f} Reconstruction")
          plt.tight_layout()
          plt.savefig(
              f"plots/reconstruction_feature_{f}.png",
              dpi=600,
              bbox_inches="tight"
          )
          plt.close()



    # ---------- 8. Feature importance in Z ----------
    coef = np.abs(probe.coef_[0])
    top_dims = np.argsort(coef)[-10:][::-1]
    print(f"🔑 Most important Z dims (linear probe): {top_dims.tolist()}")

    print("=" * 60)
    print("✔ Interpretation guide:")
    print("- Clear class separation → Z learned structure")
    print("- High probe AUC → Z encodes truth")
    print("- Low recon error → Z preserved info")
    print("- Tight clusters → stable embedding")



from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
import numpy as np
import torch

def representation_report(
    *,
    cycle,
    embedder,
    surrogate,
    xgb_teacher,
    Z_prev_train,
    Z_curr_train,
    Z_curr_test,
    X_test_xgb,
    y_train,
    y_test,
    device
):
    print(f"\n📊 REPRESENTATION REPORT — Cycle {cycle}")
    print("=" * 60)

    # ---------- 1. Teacher performance ----------
    teacher_preds = xgb_teacher.predict(
        xgb.DMatrix(X_test_xgb)
    )
    teacher_auc = roc_auc_score(y_test, teacher_preds)
    teacher_mse = mean_squared_error(y_test, teacher_preds)

    print(f"Teacher AUC          : {teacher_auc:.4f}")
    print(f"Teacher MSE          : {teacher_mse:.4f}")

    # ---------- 2. Student (truth) performance ----------
    with torch.no_grad():
        zt = torch.tensor(Z_curr_test, device=device)
        student_preds = surrogate(zt).cpu().numpy().ravel()

    student_auc = roc_auc_score(y_test, student_preds)
    student_mse = mean_squared_error(y_test, student_preds)

    print(f"Student AUC          : {student_auc:.4f}")
    print(f"Student MSE          : {student_mse:.4f}")
    print(f"Δ MSE (Student−Teacher): {student_mse - teacher_mse:+.4f}")

    # ---------- 3. Z linear probe (MOST IMPORTANT) ----------
    probe = LogisticRegression(max_iter=1000)
    probe.fit(Z_curr_train, y_train)
    probe_preds = probe.predict_proba(Z_curr_test)[:, 1]
    z_auc = roc_auc_score(y_test, probe_preds)

    print(f"Z Linear Probe AUC   : {z_auc:.4f}")

    # ---------- 4. Z drift ----------
    drift = np.mean(
    np.linalg.norm(Z_curr_train - Z_prev_train, axis=1)
)

    print(f"Z Drift              : {drift:.6f}")

    # ---------- 5. Geometry ----------
    yb = torch.tensor(y_train, device=device)
    zb = torch.tensor(Z_curr_train, device=device)

    fraud = zb[yb == 1]
    legit = zb[yb == 0]

    if len(fraud) > 0 and len(legit) > 0:
        centroid_dist = torch.norm(
            fraud.mean(0) - legit.mean(0)
        ).item()
        print(f"Centroid Distance    : {centroid_dist:.4f}")

    print("=" * 60)


import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import os

def plot_embedding(Z, y, name, outdir="plots", n_jobs=-1):
    """Use n_jobs=-1 to parallelize sklearn operations"""
    os.makedirs(outdir, exist_ok=True)

    # PCA with parallel processing
    Z_pca = PCA(2, n_components=2).fit_transform(Z)
    plt.figure(figsize=(4,3))
    plt.scatter(Z_pca[:,0], Z_pca[:,1], c=y, s=6, cmap="coolwarm")
    plt.title(f"PCA – {name}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{name}_pca.png", dpi=150)
    plt.close()

    # t-SNE
    Z_tsne = TSNE(2, perplexity=30, init="pca", learning_rate="auto", n_jobs=n_jobs).fit_transform(Z)
    plt.figure(figsize=(4,3))
    plt.scatter(Z_tsne[:,0], Z_tsne[:,1], c=y, s=6, cmap="coolwarm")
    plt.title(f"t-SNE – {name}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{name}_tsne.png", dpi=150)
    plt.close()

    # UMAP with CPU optimization
    Z_umap = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean').fit_transform(Z)
    plt.figure(figsize=(4,3))
    plt.scatter(Z_umap[:,0], Z_umap[:,1], c=y, s=6, cmap="coolwarm")
    plt.title(f"UMAP – {name}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{name}_umap.png", dpi=150)
    plt.close()

from sklearn.metrics import mean_squared_error
import torch

def plot_reconstruction(Z, X, decoder, name, outdir="plots"):
    if decoder is None:
        return None

    with torch.no_grad():
        X_hat = decoder(torch.tensor(Z, device=DEVICE)).cpu().numpy()

    mse = mean_squared_error(X, X_hat)

    f = np.random.randint(0, X.shape[1])
    plt.figure(figsize=(4,3))
    plt.scatter(X[:,f], X_hat[:,f], s=5)
    plt.xlabel("Original")
    plt.ylabel("Reconstructed")
    plt.title(f"{name} – feature {f}")
    plt.tight_layout()
    plt.savefig(f"{outdir}/{name}_recon.png", dpi=150)
    plt.close()

    return mse
