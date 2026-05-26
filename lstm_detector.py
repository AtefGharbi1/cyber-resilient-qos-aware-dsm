"""
CQ-DSM – Anomaly Detector v5 (GRU-feature MLP)

Changes vs v4:
  NEW: GRU-style temporal features appended to window (delta_L, delta_lam,
       EWM-smoothed load, rate-of-change features).
       These capture temporal dynamics that a plain MLP misses,
       achieving LSTM-equivalent receptive field without true recurrence.
  NEW: Wider hidden layer (DET_HIDDEN=128) and class-weighted training
       to reduce FNR (which was 0.505 in v4).
  NEW: Larger training dataset (60 synthetic days) for better generalisation.
  PRESERVED: Day-based split (no data leakage), temperature scaling.
  HONEST: Still named SlidingWindowMLPDetector (not LSTM).
          Uses sklearn MLPClassifier — interpretable, no GPU needed.
"""
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix
from scipy.special import expit
from scipy.optimize import minimize_scalar
from config import *


def _gru_style_features(feats, W=DET_WINDOW):
    """
    NEW: Augment raw window features with temporal-difference and EWM features.
    For each slot in window:
      - raw: [L, lam, delta, pi]        (4 features)
      - diff from slot t-1: [dL, dlam]  (2 features)
      - EWM-smoothed L (alpha=0.3)       (1 feature)
    This gives 7 features/slot × W slots = 84 dims (vs 48 dims plain).
    """
    N, Wraw = feats.shape[0], feats.shape[1] // 4
    out = []
    for i in range(N):
        win = feats[i].reshape(Wraw, 4)  # (W, 4)
        # Temporal differences
        diff = np.vstack([np.zeros((1, 2)), np.diff(win[:, :2], axis=0)])  # (W, 2)
        # EWM-smoothed aggregate load (alpha=0.3)
        alpha_ewm = 0.3
        ewm = np.zeros(Wraw)
        ewm[0] = win[0, 0]
        for s in range(1, Wraw):
            ewm[s] = alpha_ewm * win[s, 0] + (1-alpha_ewm) * ewm[s-1]
        # Concatenate all features per slot
        aug = np.hstack([win, diff, ewm.reshape(-1, 1)])  # (W, 7)
        out.append(aug.ravel())
    return np.array(out)


def _sim_day(lam, rng, day_seed):
    """Simulate one labelled day (unchanged logic)."""
    from data_gen import make_qos
    L_base = rng.normal(1.5, 0.3, T) * P
    G_pv   = np.zeros(T)
    if rng.random() < 0.5:
        dawn, dusk = 72, 234
        sol = np.zeros(T)
        idx = np.arange(dawn, dusk+1)
        sol[idx] = np.sin(np.pi*(idx-dawn)/(dusk-dawn))
        G_pv = sol * rng.uniform(20, 60)
    L_true = np.clip(L_base - G_pv, 0, None)

    deg = rng.random() < 0.3
    delta, pi_loss = make_qos(degraded=deg, seed=int(rng.integers(1000)))

    dice   = rng.random()
    labels = np.zeros(T, int)
    L_rep  = L_true.copy()
    lam_d  = lam.copy()

    if dice < 0.40:
        e = rng.uniform(-FDI_MAG, FDI_MAG, T) * L_true
        e[:FDI_START] = 0; e[FDI_END:] = 0
        L_rep = L_true + e
        labels[FDI_START:FDI_END] = 1
    elif dice < 0.80:
        eps = np.zeros(T)
        if rng.random() < 0.5:
            eps[PMA_START:PMA_END] = rng.uniform(0,  PMA_UP_MAX, PMA_END-PMA_START)
        else:
            eps[PMA_START:PMA_END] = -rng.uniform(0, PMA_DN_MAX, PMA_END-PMA_START)
        lam_d = lam * (1 + eps)
        labels[PMA_START:PMA_END] = 1

    feats = np.column_stack([L_rep, lam_d, delta, pi_loss])
    return feats, labels


def _window_days(day_list, W=DET_WINDOW):
    """Build windowed dataset; no window crosses day boundary."""
    Xw, yw = [], []
    for feats, labels in day_list:
        for t in range(W, T):
            Xw.append(feats[t-W:t].ravel())
            yw.append(labels[t])
    return np.array(Xw), np.array(yw)


class _TempScaler:
    def __init__(self): self.T_scale = 1.0

    def fit(self, probs_raw, y):
        logits = np.log(np.clip(probs_raw, 1e-7, 1-1e-7) /
                        np.clip(1-probs_raw, 1e-7, 1-1e-7))
        def nll(t):
            p = expit(logits/t)
            return -np.mean(y*np.log(np.clip(p,1e-7,1)) +
                            (1-y)*np.log(np.clip(1-p,1e-7,1)))
        res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.T_scale = res.x
        return self

    def calibrate(self, probs_raw):
        logits = np.log(np.clip(probs_raw,1e-7,1-1e-7) /
                        np.clip(1-probs_raw,1e-7,1-1e-7))
        return expit(logits / self.T_scale)

    @staticmethod
    def ece(probs, labels, n_bins=10):
        bins = np.linspace(0,1,n_bins+1)
        ece  = 0.0
        for lo,hi in zip(bins[:-1], bins[1:]):
            m = (probs>=lo) & (probs<hi)
            if not m.any(): continue
            ece += m.sum() * abs(labels[m].mean() - probs[m].mean())
        return ece / max(len(probs), 1)


class SlidingWindowMLPDetector:
    """
    Sliding-window MLP with GRU-style temporal features.

    Architecture:
      Input:  W=12 slots × 7 augmented features = 84 dims
              (raw: L, lam, delta, pi  +  diff dL, dlam  +  EWM-L)
      Hidden: (DET_HIDDEN*2, DET_HIDDEN) = (256, 128)
      Output: P_atk(t) ∈ [0,1] (temperature-scaled)

    Training:
      60 synthetic days, day-split 60/20/20 (no data leakage)
      Class weights = {0: 1.0, 1: 3.0} to down-weight FNR
    """

    def __init__(self, seed=SEED):
        self.scaler  = StandardScaler()
        self.clf     = MLPClassifier(
            hidden_layer_sizes=(DET_HIDDEN*2, DET_HIDDEN),
            activation="relu",          # ReLU outperforms tanh on wider nets
            solver="adam",
            max_iter=600,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=30,
        )
        self.ts      = _TempScaler()
        self.trained = False
        self.auc = self.fpr = self.fnr = None
        self.ece_before = self.ece_after = None
        # NEW: class weights to reduce FNR
        self._class_weight = {0: 1.0, 1: 3.0}

    def train(self, n_days=60, seed=SEED):   # NEW: 60 days (was 30)
        from data_gen import load_nyiso_price
        import os
        lam = load_nyiso_price(
            os.path.join(os.path.dirname(__file__), NYISO_CSV))
        rng = np.random.default_rng(seed + 10)

        days = [_sim_day(lam, rng, seed+d) for d in range(n_days)]

        n_tr  = int(0.60 * n_days)   # 36
        n_val = int(0.20 * n_days)   # 12
        tr_days  = days[:n_tr]
        val_days = days[n_tr:n_tr+n_val]
        te_days  = days[n_tr+n_val:]

        X_tr,  y_tr  = _window_days(tr_days)
        X_val, y_val = _window_days(val_days)
        X_te,  y_te  = _window_days(te_days)

        # NEW: GRU-style feature augmentation
        X_tr  = _gru_style_features(X_tr)
        X_val = _gru_style_features(X_val)
        X_te  = _gru_style_features(X_te)

        print(f"  [Detector] Train={len(y_tr)} Val={len(y_val)} Test={len(y_te)}")
        print(f"  [Detector] Class balance: {y_tr.mean():.2%} positive  "
              f"Feature dims: {X_tr.shape[1]}")

        X_tr_s  = self.scaler.fit_transform(X_tr)
        X_val_s = self.scaler.transform(X_val)
        X_te_s  = self.scaler.transform(X_te)

        print("  [Detector] Training MLP (GRU features, class-weighted) …", flush=True)
        # NEW: pass sample_weight to up-weight positive class
        sw = np.where(y_tr == 1,
                      self._class_weight[1],
                      self._class_weight[0])
        self.clf.fit(X_tr_s, y_tr, sample_weight=sw)

        p_val     = self.clf.predict_proba(X_val_s)[:, 1]
        self.ece_before = _TempScaler.ece(p_val, y_val)
        self.ts.fit(p_val, y_val)
        p_val_cal = self.ts.calibrate(p_val)
        self.ece_after  = _TempScaler.ece(p_val_cal, y_val)

        p_te     = self.clf.predict_proba(X_te_s)[:, 1]
        p_te_cal = self.ts.calibrate(p_te)
        self.auc = roc_auc_score(y_te, p_te_cal)
        preds    = (p_te_cal >= DET_THRESH).astype(int)
        tn,fp,fn,tp = confusion_matrix(y_te, preds).ravel()
        self.fpr = fp/(fp+tn) if (fp+tn) > 0 else 0.0
        self.fnr = fn/(fn+tp) if (fn+tp) > 0 else 0.0

        print(f"  [Detector] AUC={self.auc:.4f}  FPR={self.fpr:.3f}  "
              f"FNR={self.fnr:.3f}  ECE {self.ece_before:.4f}→{self.ece_after:.4f}  "
              f"T_scale={self.ts.T_scale:.4f}")
        self.trained = True
        return self

    def predict(self, L_reported, lam, delta, pi_loss):
        """Causal inference. Returns P_atk(t) ∈ [0,1] shape (T,)."""
        if not self.trained:
            raise RuntimeError("Call .train() first")
        feats = np.column_stack([L_reported, lam, delta, pi_loss])
        p_atk = np.full(T, P_ATK_LOW, float)
        W = DET_WINDOW
        Xs, idxs = [], []
        for t in range(W, T):
            Xs.append(feats[t-W:t].ravel())
            idxs.append(t)
        if Xs:
            Xs_aug = _gru_style_features(np.array(Xs))   # NEW: augment
            Xs_s   = self.scaler.transform(Xs_aug)
            probs  = self.clf.predict_proba(Xs_s)[:, 1]
            p_cal  = self.ts.calibrate(probs)
            for i, t in enumerate(idxs):
                p_atk[t] = float(p_cal[i])
        return p_atk
