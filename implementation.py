#!/usr/bin/env python3
"""
Implementation script for PoC FRC forecasting and compliance classification

- Loads CSV dataset from a data directory containing:
  sudan_2013.csv, jordan_2014.csv, jordan_2015.csv, rwanda_2015.csv
- Builds simple field-style features from se1_* + elapsed time + time encodings
- Trains compact PyTorch MLP for regression (PoC FRC) and classification (≥ 0.2 mg/L)
- Uses 80/10/10 (train/val/test) split with StandardScaler fit on train only
- Saves metrics and plots; optional small hyperparameter grid

CLI examples:
  python implementation.py --data-dir dataset/data --out-dir .
  python implementation.py --data-dir /path/to/data --run-grid
"""

import os
import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    mean_absolute_error, mean_squared_error,
)
from sklearn.linear_model import LogisticRegression
import torch
import torch.nn as nn

SEED = 42
np.random.seed(SEED)
try:
    torch.manual_seed(SEED)
except Exception:
    pass

FRC_THRESHOLD = 0.2


def _to_ts(date_val, time_val):
    """Combine date+time robustly."""
    try:
        if pd.isna(date_val) and not pd.isna(time_val):
            return pd.to_datetime(time_val, errors='coerce')
        if pd.isna(time_val):
            return pd.to_datetime(date_val, errors='coerce')
        d = pd.to_datetime(date_val, errors='coerce')
        t = pd.to_datetime(time_val, errors='coerce')
        if pd.isna(d) and not pd.isna(t):
            return t
        if not pd.isna(d) and not pd.isna(t):
            return pd.to_datetime(str(getattr(d, 'date', lambda: d)()) + ' ' + str(getattr(t, 'time', lambda: t)()))
        return pd.to_datetime(date_val, errors='coerce')
    except Exception:
        return pd.NaT


from typing import Optional

def resolve_csv_dir(explicit: Optional[str] = None):
    if explicit:
        p = Path(explicit)
        if p.exists():
            return str(p)
        raise FileNotFoundError(f'Data directory not found: {explicit}')
    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()
    p = script_dir / 'dataset' / 'data'
    if p.exists():
        return str(p)
    raise FileNotFoundError('Could not locate dataset/data folder. Pass --data-dir /path/to/data')

SITE_CSV_FILES = {
    'South Sudan 2013': 'sudan_2013.csv',
    'Jordan 2014': 'jordan_2014.csv',
    'Jordan 2015': 'jordan_2015.csv',
    'Rwanda 2015': 'rwanda_2015.csv',
}


def load_site_csv_dataset(csv_dir: str, site_files: dict):
    X_rows, y_reg, y_cls, sites, frames = [], [], [], [], []
    for site, fname in site_files.items():
        path = os.path.join(csv_dir, fname)
        df = pd.read_csv(path, encoding='utf-8-sig')
        t1 = df.apply(lambda r: _to_ts(r.get('se1_date'), r.get('se1_time')), axis=1)
        t4 = df.apply(lambda r: _to_ts(r.get('se4_date'), r.get('se4_time')), axis=1)
        elapsed_h = (t4 - t1).dt.total_seconds() / 3600.0

        tt = pd.to_datetime(df.get('se1_time', pd.NaT), errors='coerce')
        hod = tt.dt.hour + tt.dt.minute / 60.0
        theta = 2 * np.pi * (hod / 24.0)
        sin_hod = np.sin(theta)
        cos_hod = np.cos(theta)
        wday = pd.to_datetime(df.get('se1_date', pd.NaT), errors='coerce').dt.weekday

        feats = pd.DataFrame({
            'se1_frc': df.get('se1_frc'),
            'se1_trc': df.get('se1_trc'),
            'se1_turb': df.get('se1_turb'),
            'se1_wattemp': df.get('se1_wattemp'),
            'se1_cond': df.get('se1_cond'),
            'se1_ph': df.get('se1_ph'),
            'se1_orp': df.get('se1_orp'),
            'elapsed_h': elapsed_h,
            'sin_hod': sin_hod,
            'cos_hod': cos_hod,
            'weekday': wday,
        })
        y = df.get('se4_frc')
        if 'se5_frc' in df.columns:
            y = y.fillna(df.get('se5_frc'))

        m = (~y.isna()) & (feats.notna().any(axis=1))
        feats = feats.loc[m].apply(pd.to_numeric, errors='coerce')
        y = y.loc[m].astype(float)

        med = feats.median().fillna(0)
        feats = feats.fillna(med).fillna(0.0)
        vals = feats.values.astype(np.float32)
        vals[~np.isfinite(vals)] = 0.0

        X_rows.append(vals)
        y_reg.append(y.values.astype(np.float32))
        y_cls.append((y.values >= FRC_THRESHOLD).astype(np.int64))
        sites += [site] * len(y)
        frames.append(feats.assign(site=site, y=y.values))

    X = np.vstack(X_rows)
    yr = np.concatenate(y_reg)
    yc = np.concatenate(y_cls)
    site_ids = np.array(sites)
    df_all = pd.concat(frames, ignore_index=True)
    X[~np.isfinite(X)] = 0.0
    print('Loaded CSV dataset:', X.shape, '| regression targets:', yr.shape, '| classification:', yc.shape)
    print('Sites:', {s: int((site_ids == s).sum()) for s in np.unique(site_ids)})
    print('NaN count in X:', int(np.isnan(X).sum()))
    return X, yr, yc, site_ids, df_all


def build_splits_80_10_10(X, y, seed=42, stratify=None):
    X_tv, X_test, y_tv, y_test = train_test_split(X, y, test_size=0.10, random_state=seed, stratify=stratify)
    strat2 = y_tv if stratify is not None else None
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv, test_size=0.111111, random_state=seed, stratify=strat2)
    scaler = StandardScaler().fit(X_train)
    return (
        scaler.transform(X_train).astype('float32'), y_train,
        scaler.transform(X_val).astype('float32'), y_val,
        scaler.transform(X_test).astype('float32'), y_test, scaler,
    )



class MLPReg(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32), p_drop=0.2):
        super().__init__()
        layers = []; d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_regressor(X_tr, y_tr, X_val, y_val, seed=0, epochs=200, patience=15, hidden=(64,32), lr=1e-3, p_drop=0.2):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = MLPReg(in_dim=X_tr.shape[1], hidden=hidden, p_drop=p_drop).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.MSELoss(); best=(1e9,None); wait=0
    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device); ytr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=device); yva = torch.tensor(y_val, dtype=torch.float32, device=device)
    for ep in range(1, epochs+1):
        m.train(); opt.zero_grad(); pred = torch.clamp(m(Xtr), 0, 4); loss = crit(pred, torch.clamp(ytr,0,4)); loss.backward(); opt.step()
        m.eval(); 
        with torch.no_grad(): pv = torch.clamp(m(Xva),0,4); vloss = crit(pv, torch.clamp(yva,0,4)).item()
        if vloss<best[0]: best=(vloss,{k:v.cpu().clone() for k,v in m.state_dict().items()}); wait=0
        else: wait+=1
        if ep%20==0: print(f'Epoch {ep:03d} val MSE={vloss:.4f}')
        if wait>=patience: print('Early stopping'); break
    if best[1] is not None: m.load_state_dict({k:v.to(device) for k,v in best[1].items()})
    return m


def eval_reg(y_true, y_pred):
    y_true = np.clip(y_true, 0, 4); y_pred = np.clip(y_pred, 0, 4)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred)); mae = mean_absolute_error(y_true, y_pred)
    print(f'RMSE={rmse:.4f}, MAE={mae:.4f}')
    return {'rmse': rmse, 'mae': mae}


class MLPCls(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32), p_drop=0.2):
        super().__init__()
        layers=[]; d=in_dim
        for h in hidden:
            layers += [nn.Linear(d,h), nn.ReLU(), nn.Dropout(p_drop)]
            d=h
        layers += [nn.Linear(d,1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_mlp_cls(X_train, y_train, X_val, y_val, seed=0, epochs=100, patience=12, hidden=(64,32), lr=1e-3, p_drop=0.2, pos_weight_scale=1.0):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = MLPCls(in_dim=X_train.shape[1], hidden=hidden, p_drop=p_drop).to(device)
    pos = max(int(y_train.sum()), 1); neg = max(len(y_train)-int(y_train.sum()), 1)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((neg/pos)*pos_weight_scale, device=device))
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    Xtr = torch.tensor(X_train, dtype=torch.float32, device=device); ytr = torch.tensor(y_train, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=device); yva = torch.tensor(y_val, dtype=torch.float32, device=device)
    best=(-1,None); wait=0
    for ep in range(1,epochs+1):
        m.train(); opt.zero_grad(); loss = crit(m(Xtr), ytr); loss.backward(); opt.step()
        m.eval(); 
        with torch.no_grad(): pv = torch.sigmoid(m(Xva)).cpu().numpy(); bal = balanced_accuracy_score(y_val.astype(int), (pv>=0.5).astype(int))
        if bal>best[0]: best=(bal,{k:v.cpu().clone() for k,v in m.state_dict().items()}); wait=0
        else: wait+=1
        if ep%10==0: print(f'Epoch {ep:03d} val bal acc={bal:.4f}')
        if wait>=patience: print('Early stopping'); break
    if best[1] is not None: m.load_state_dict({k:v.to(device) for k,v in best[1].items()})
    return m


def evaluate_cls(y_true, y_prob, threshold=0.5, label='Eval'):
    y_pred = (y_prob >= threshold).astype(int)
    out = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true))==2 else float('nan'),
        'avg_precision': average_precision_score(y_true, y_prob),
        'confusion_matrix': confusion_matrix(y_true, y_pred)
    }
    print(f"{label}: " + ", ".join([f"{k}={v:.4f}" for k,v in out.items() if k!='confusion_matrix']))
    print('  confusion_matrix=\n', out['confusion_matrix'])
    return out



class MLPQuantReg(nn.Module):
    def __init__(self, in_dim, hidden=(64, 32), p_drop=0.2):
        super().__init__()
        layers = []; d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p_drop)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def pinball_loss(y_pred, y_true, q):
    e = y_true - y_pred
    return torch.mean(torch.maximum(q * e, (q - 1) * e))

def train_quantile_model(X_tr, y_tr, X_val, y_val, q=0.5, seed=0, epochs=200, patience=15, lr=1e-3):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    m = MLPQuantReg(in_dim=X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-4)
    Xtr = torch.tensor(X_tr, dtype=torch.float32, device=device)
    ytr = torch.tensor(y_tr, dtype=torch.float32, device=device)
    Xva = torch.tensor(X_val, dtype=torch.float32, device=device)
    yva = torch.tensor(y_val, dtype=torch.float32, device=device)
    best=(1e9,None); wait=0
    for ep in range(1,epochs+1):
        m.train(); opt.zero_grad(); pred = torch.clamp(m(Xtr),0,4); loss = pinball_loss(pred, torch.clamp(ytr,0,4), q)
        loss.backward(); opt.step()
        m.eval();
        with torch.no_grad(): pv = torch.clamp(m(Xva),0,4); vloss = pinball_loss(pv, torch.clamp(yva,0,4), q).item()
        if vloss<best[0]: best=(vloss,{k:v.cpu().clone() for k,v in m.state_dict().items()}); wait=0
        else: wait+=1
        if wait>=patience: break
    if best[1] is not None: m.load_state_dict({k:v.to(device) for k,v in best[1].items()})
    return m

def ci_reliability(levels, X_tr, y_tr, X_val, y_val, X_te, y_te, seed=0):
    caps = []
    for alpha in levels:
        p_lo = (1 - alpha/100)/2
        p_hi = 1 - p_lo
        m_lo = train_quantile_model(X_tr, y_tr, X_val, y_val, q=p_lo, seed=seed)
        m_hi = train_quantile_model(X_tr, y_tr, X_val, y_val, q=p_hi, seed=seed)
        with torch.no_grad():
            qlo = m_lo(torch.tensor(X_te, dtype=torch.float32)).cpu().numpy()
            qhi = m_hi(torch.tensor(X_te, dtype=torch.float32)).cpu().numpy()
        y_true = np.clip(y_te, 0, 4)
        inside = (y_true >= np.minimum(qlo, qhi)) & (y_true <= np.maximum(qlo, qhi))
        caps.append(100*np.mean(inside))
    return np.array(caps)


def parse_args():
    ap = argparse.ArgumentParser(description='PoC FRC forecasting and compliance classification (80/10/10).')
    ap.add_argument('--data-dir', type=str, default=None, help='Directory with CSV files (sudan_2013.csv, jordan_2014.csv, jordan_2015.csv, rwanda_2015.csv). Default: dataset/data')
    ap.add_argument('--out-dir', type=str, default=None, help='Directory to write outputs (plots/CSVs). Default: script directory')
    ap.add_argument('--seed', type=int, default=SEED, help='Random seed (default: 42)')
    ap.add_argument('--run-grid', action='store_true', help='Run small hyperparameter grid and save results')
    return ap.parse_args()


def main():
    args = parse_args()
    global SEED
    SEED = int(args.seed)
    np.random.seed(SEED)
    try:
        torch.manual_seed(SEED)
    except Exception:
        pass

    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()
    csv_dir = resolve_csv_dir(args.data_dir)
    print('Using CSV_DIR =', csv_dir)
    out_dir = Path(args.out_dir) if args.out_dir else script_dir


    X_all, y_reg_all, y_cls_all, site_ids, df_sites = load_site_csv_dataset(csv_dir, SITE_CSV_FILES)
    X_all = X_all.astype('float32'); X_all[~np.isfinite(X_all)] = 0.0

    print('\n[Regression] 80/10/10 split')
    Xtr, ytr, Xva, yva, Xte, yte, _ = build_splits_80_10_10(X_all, y_reg_all, seed=SEED, stratify=None)
    
    reg = train_regressor(Xtr, ytr, Xva, yva, seed=0, hidden=(64,64,32), lr=1e-3, p_drop=0.0)
    reg.eval();
    with torch.no_grad(): y_pred_test = reg(torch.tensor(Xte, dtype=torch.float32)).cpu().numpy()
    _ = eval_reg(yte, y_pred_test)

    print('\n[Classification] 80/10/10 split (stratified)')
    Xtr_c, ytr_c, Xva_c, yva_c, Xte_c, yte_c, _ = build_splits_80_10_10(X_all, y_cls_all, seed=SEED, stratify=y_cls_all)
    lr = LogisticRegression(class_weight='balanced', max_iter=2000, solver='lbfgs')
    lr.fit(Xtr_c, ytr_c)
    p_val_lr = lr.predict_proba(Xva_c)[:,1]
    val_bal_lr = balanced_accuracy_score(yva_c, (p_val_lr>=0.5).astype(int))
    print(f'Val (LogReg) bal acc = {val_bal_lr:.4f}')
    p_te_lr = lr.predict_proba(Xte_c)[:,1]
    _ = evaluate_cls(yte_c, p_te_lr, label='Test (LogReg)')

    mcls = train_mlp_cls(Xtr_c, ytr_c, Xva_c, yva_c, seed=0, hidden=(64,32), lr=1e-3, p_drop=0.2, pos_weight_scale=1.5)
    with torch.no_grad():
        p_val_mlp = torch.sigmoid(mcls(torch.tensor(Xva_c, dtype=torch.float32))).cpu().numpy()
        p_te_mlp = torch.sigmoid(mcls(torch.tensor(Xte_c, dtype=torch.float32))).cpu().numpy()
    val_bal_mlp = balanced_accuracy_score(yva_c, (p_val_mlp>=0.5).astype(int))
    print(f'Val (MLP) bal acc = {val_bal_mlp:.4f}')
    _ = evaluate_cls(yte_c, p_te_mlp, label='Test (MLP)')



    # Save 80/10/10 outputs: metrics CSVs and figures
    import csv
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(yte_c, (p_te_mlp >= 0.5).astype(int))
    cls_metrics = {
        'accuracy': accuracy_score(yte_c, (p_te_mlp >= 0.5).astype(int)),
        'balanced_accuracy': balanced_accuracy_score(yte_c, (p_te_mlp >= 0.5).astype(int)),
        'f1': f1_score(yte_c, (p_te_mlp >= 0.5).astype(int), zero_division=0),
        'roc_auc': roc_auc_score(yte_c, p_te_mlp) if len(np.unique(yte_c)) == 2 else float('nan'),
        'avg_precision': average_precision_score(yte_c, p_te_mlp),
        'tn': int(cm[0,0]), 'fp': int(cm[0,1]), 'fn': int(cm[1,0]), 'tp': int(cm[1,1])
    }
    cls_csv = out_dir / 'classification_metrics_80_10_10.csv'
    with open(cls_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cls_metrics.keys())
        w.writerow(cls_metrics.values())
    print('Saved:', cls_csv)

    # Confusion matrix figure
    import matplotlib.pyplot as plt
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap='Blues')
    plt.title('Confusion Matrix (Test)')
    plt.xlabel('Predicted'); plt.ylabel('True')
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i,j], ha='center', va='center', color='black')
    plt.xticks([0,1], ['<0.2','≥0.2'])
    plt.yticks([0,1], ['<0.2','≥0.2'])
    plt.tight_layout()
    cm_png = out_dir / 'confusion_matrix.png'
    plt.savefig(cm_png, dpi=150)
    plt.close()
    print('Saved plot:', cm_png)

    # Per-site classification metrics (by site on classification test split)
    try:
        n_all = len(y_cls_all)
        all_idx_cls = np.arange(n_all)
        idx_tv_c, idx_test_c = train_test_split(all_idx_cls, test_size=0.10, random_state=SEED, stratify=y_cls_all)
        site_test_c = site_ids[idx_test_c]
        rows_cls = []
        for s in sorted(set(site_test_c.tolist())):
            m = (site_test_c == s)
            if not np.any(m):
                continue
            y_true_s = yte_c[m].astype(int)
            y_prob_s = p_te_mlp[m]
            y_pred_s = (y_prob_s >= 0.5).astype(int)
            cm_s = confusion_matrix(y_true_s, y_pred_s)
            tn = int(cm_s[0,0]) if cm_s.shape==(2,2) else 0
            fp = int(cm_s[0,1]) if cm_s.shape==(2,2) else 0
            fn = int(cm_s[1,0]) if cm_s.shape==(2,2) else 0
            tp = int(cm_s[1,1]) if cm_s.shape==(2,2) else 0
            acc = accuracy_score(y_true_s, y_pred_s)
            bal = balanced_accuracy_score(y_true_s, y_pred_s)
            f1s = f1_score(y_true_s, y_pred_s, zero_division=0)
            auc = roc_auc_score(y_true_s, y_prob_s) if len(np.unique(y_true_s))==2 else float('nan')
            ap = average_precision_score(y_true_s, y_prob_s)
            rows_cls.append({'site': str(s), 'n': int(m.sum()), 'accuracy': acc, 'balanced_accuracy': bal,
                             'f1': f1s, 'roc_auc': auc, 'avg_precision': ap,
                             'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp})
        if rows_cls:
            import pandas as pd
            df_cls_sites = pd.DataFrame(rows_cls).sort_values('site')
            bysite_cls_csv = out_dir / 'classification_metrics_by_site_80_10_10.csv'
            df_cls_sites.to_csv(bysite_cls_csv, index=False)
            print('Saved:', bysite_cls_csv)
            # Bar chart: balanced accuracy and F1 by site
            try:
                cats = df_cls_sites['site'].tolist()
                bal_vals = df_cls_sites['balanced_accuracy'].astype(float).tolist()
                f1_vals = df_cls_sites['f1'].astype(float).tolist()
                x = np.arange(len(cats))
                width = 0.35
                plt.figure(figsize=(10,4))
                # Balanced accuracy subplot
                plt.subplot(1,2,1)
                plt.bar(x, bal_vals, width=0.6, color='#4C78A8')
                plt.xticks(x, cats, rotation=30, ha='right')
                plt.ylim(0,1); plt.ylabel('Balanced accuracy')
                plt.title('Balanced Accuracy by Site')
                plt.grid(True, axis='y', alpha=0.3)
                # F1 subplot
                plt.subplot(1,2,2)
                plt.bar(x, f1_vals, width=0.6, color='#F58518')
                plt.xticks(x, cats, rotation=30, ha='right')
                plt.ylim(0,1); plt.ylabel('F1 score')
                plt.title('F1 by Site')
                plt.grid(True, axis='y', alpha=0.3)
                plt.tight_layout()
                cls_by_site_png = out_dir / 'classification_by_site_balacc_f1.png'
                plt.savefig(cls_by_site_png, dpi=150)
                plt.close()
                print('Saved plot:', cls_by_site_png)
            except Exception as e:
                print('By-site classification bar chart not generated:', e)
    except Exception as e:
        print('By-site classification metrics not generated:', e)

    # Regression metrics CSV
    reg_csv = out_dir / 'regression_metrics_80_10_10.csv'
    wr = eval_reg(yte, y_pred_test)
    with open(reg_csv, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['rmse','mae'])
        w.writerow([wr['rmse'], wr['mae']])
    print('Saved:', reg_csv)

    # Regression scatter plot
    plt.figure(figsize=(4.5,4.5))
    plt.scatter(yte, y_pred_test, s=18, alpha=0.7)
    lims = [0, max(2.0, float(max(np.max(yte), np.max(y_pred_test))))]
    plt.plot(lims, lims, 'k--', alpha=0.6)
    plt.xlim(lims); plt.ylim(lims)
    plt.xlabel('True PoC FRC (mg/L)'); plt.ylabel('Predicted PoC FRC (mg/L)')
    plt.title('Test: True vs Predicted')
    plt.grid(True, alpha=0.3); plt.tight_layout()
    reg_png = out_dir / 'regression_scatter.png'
    plt.savefig(reg_png, dpi=150)
    plt.close()
    print('Saved plot:', reg_png)

    # CI reliability via quantile models on the 80/10/10 split
    # Extra: regression scatter by location (site)
    try:
        # Recreate the exact test indices using the same split protocol
        n = len(y_reg_all)
        all_idx = np.arange(n)
        idx_tv, idx_test = train_test_split(all_idx, test_size=0.10, random_state=SEED, stratify=None)
        # Consistency check is optional; we trust same random_state reproduces the split
        site_test = site_ids[idx_test]
        # Shared limits across panels
        lim_hi = max(2.0, float(max(np.max(yte), np.max(y_pred_test))))
        lims = [0, lim_hi]
        # Faceted 2x2 by site
        sites_unique = list(dict.fromkeys(site_test))  # preserve order of appearance
        cols = 2
        rows = int(np.ceil(len(sites_unique)/cols))
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows*cols == 1:
            axes = np.array([[axes]])
        axes = axes.reshape(rows, cols)
        for i, s in enumerate(sites_unique):
            r = i//cols; c = i%cols
            ax = axes[r, c]
            m = (site_test == s)
            ax.scatter(yte[m], y_pred_test[m], s=18, alpha=0.7)
            ax.plot(lims, lims, 'k--', alpha=0.6)
            ax.set_xlim(lims); ax.set_ylim(lims)
            ax.set_title(str(s))
            ax.set_xlabel('True PoC FRC (mg/L)'); ax.set_ylabel('Predicted (mg/L)')
            ax.grid(True, alpha=0.3)
        # Hide unused subplots
        for j in range(i+1, rows*cols):
            r = j//cols; c = j%cols
            axes[r, c].axis('off')
        plt.tight_layout()
        bysite_png = out_dir / 'regression_scatter_by_site.png'
        plt.savefig(bysite_png, dpi=150)
        plt.close()
        print('Saved plot:', bysite_png)
        # Single plot colored by site
        plt.figure(figsize=(6,5))
        cmap = plt.get_cmap('tab10')
        for i, s in enumerate(sites_unique):
            m = (site_test == s)
            plt.scatter(yte[m], y_pred_test[m], s=18, alpha=0.75, color=cmap(i%10), label=str(s))
        plt.plot(lims, lims, 'k--', alpha=0.6)
        plt.xlim(lims); plt.ylim(lims)
        plt.xlabel('True PoC FRC (mg/L)'); plt.ylabel('Predicted PoC FRC (mg/L)')
        plt.title('Test: True vs Predicted by Site')
        plt.legend(frameon=False)
        plt.grid(True, alpha=0.3); plt.tight_layout()
        colored_png = out_dir / 'regression_scatter_colored_by_site.png'
        plt.savefig(colored_png, dpi=150)
        plt.close()
        print('Saved plot:', colored_png)

        # Per-site test RMSE/MAE CSV
        rows = []
        from sklearn.metrics import mean_squared_error, mean_absolute_error
        for s in sites_unique:
            m = (site_test == s)
            if not np.any(m):
                continue
            yt = np.clip(yte[m], 0, 4)
            yp = np.clip(y_pred_test[m], 0, 4)
            rmse_s = float(np.sqrt(mean_squared_error(yt, yp)))
            mae_s = float(mean_absolute_error(yt, yp))
            rows.append({'site': str(s), 'n': int(m.sum()), 'rmse': rmse_s, 'mae': mae_s})
        if rows:
            import pandas as pd
            df_sites_metrics = pd.DataFrame(rows).sort_values('site')
            bysite_csv = out_dir / 'regression_metrics_by_site_80_10_10.csv'
            df_sites_metrics.to_csv(bysite_csv, index=False)
            print('Saved:', bysite_csv)
            # Bar chart: RMSE and MAE by site
            try:
                cats = df_sites_metrics['site'].tolist()
                rmse_vals = df_sites_metrics['rmse'].astype(float).tolist()
                mae_vals = df_sites_metrics['mae'].astype(float).tolist()
                x = np.arange(len(cats))
                plt.figure(figsize=(10,4))
                # RMSE subplot
                plt.subplot(1,2,1)
                plt.bar(x, rmse_vals, width=0.6, color='#54A24B')
                plt.xticks(x, cats, rotation=30, ha='right')
                plt.ylabel('RMSE (mg/L)')
                plt.title('RMSE by Site')
                plt.grid(True, axis='y', alpha=0.3)
                # MAE subplot
                plt.subplot(1,2,2)
                plt.bar(x, mae_vals, width=0.6, color='#E45756')
                plt.xticks(x, cats, rotation=30, ha='right')
                plt.ylabel('MAE (mg/L)')
                plt.title('MAE by Site')
                plt.grid(True, axis='y', alpha=0.3)
                plt.tight_layout()
                reg_by_site_png = out_dir / 'regression_by_site_rmse_mae.png'
                plt.savefig(reg_by_site_png, dpi=150)
                plt.close()
                print('Saved plot:', reg_by_site_png)
            except Exception as e:
                print('By-site regression bar chart not generated:', e)
    except Exception as e:
        print('By-site scatter not generated:', e)

    # CI reliability via quantile models on the 80/10/10 split
    levels = [50, 60, 70, 80, 90]
    caps = ci_reliability(levels, Xtr, ytr, Xva, yva, Xte, yte, seed=0)
    print('\n[Calibration] CI reliability via quantile models (80/10/10 test)')
    plt.figure(figsize=(5, 4))
    plt.plot(levels, levels, 'k--', label='Ideal 1:1')
    plt.plot(levels, caps, 'o-', label='Empirical (quantile MLP)')
    plt.xlabel('Nominal CI (%)'); plt.ylabel('Empirical capture (%)')
    plt.title('CI Reliability (Test)')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    rel_png = out_dir / 'ci_reliability.png'
    plt.savefig(rel_png, dpi=150)
    plt.close()
    print('Saved plot:', rel_png)

    # Optional hyperparameter grid (enable with --run-grid)
    if hasattr(sys, 'argv'):
        import argparse as _argp
        # Quick check to avoid heavy grid unless requested
        if any(a == '--run-grid' for a in sys.argv[1:]):
            print('\n[Search] Classification MLP hyperparameter grid (select by validation balanced accuracy)')
            cls_grid = []
            hidden_grid = [(64,32), (128,64), (64,64,32)]
            lr_grid = [1e-3, 3e-4]
            pdrop_grid = [0.2, 0.0]
            wscale_grid = [1.0, 1.5, 2.0]
            best = (-1.0, None)
            for h in hidden_grid:
                for lr_ in lr_grid:
                    for pd_ in pdrop_grid:
                        for ws in wscale_grid:
                            mtmp = train_mlp_cls(Xtr_c, ytr_c, Xva_c, yva_c, seed=0, hidden=h, lr=lr_, p_drop=pd_, pos_weight_scale=ws)
                            with torch.no_grad():
                                pv = torch.sigmoid(mtmp(torch.tensor(Xva_c, dtype=torch.float32))).cpu().numpy()
                            bal = balanced_accuracy_score(yva_c.astype(int), (pv>=0.5).astype(int))
                            cls_grid.append({'hidden':h,'lr':lr_,'p_drop':pd_,'pos_weight_scale':ws,'val_bal_acc':float(bal)})
                            if bal > best[0]: best = (bal, (h, lr_, pd_, ws))
            import pandas as pd
            cls_grid_df = pd.DataFrame(cls_grid)
            cls_grid_csv = out_dir / 'classification_grid_results.csv'
            cls_grid_df.sort_values('val_bal_acc', ascending=False).to_csv(cls_grid_csv, index=False)
            print('Saved:', cls_grid_csv)
            if best[1] is not None:
                h, lr_, pd_, ws = best[1]
                print(f"Best CLS config: hidden={h}, lr={lr_}, p_drop={pd_}, pos_weight_scale={ws}, val_bal_acc={best[0]:.4f}")
                mbest = train_mlp_cls(Xtr_c, ytr_c, Xva_c, yva_c, seed=0, hidden=h, lr=lr_, p_drop=pd_, pos_weight_scale=ws)
                with torch.no_grad():
                    p_te = torch.sigmoid(mbest(torch.tensor(Xte_c, dtype=torch.float32))).cpu().numpy()
                
                test_bal = balanced_accuracy_score(yte_c.astype(int), (p_te>=0.5).astype(int))
                test_auc = roc_auc_score(yte_c, p_te) if len(np.unique(yte_c))==2 else float('nan')
                test_ap = average_precision_score(yte_c, p_te)
                test_acc = accuracy_score(yte_c, (p_te>=0.5).astype(int))
                test_f1 = f1_score(yte_c, (p_te>=0.5).astype(int), zero_division=0)
                best_cls_csv = out_dir / 'classification_best_model_metrics.csv'
                pd.DataFrame([{'hidden':h,'lr':lr_,'p_drop':pd_,'pos_weight_scale':ws,
                              'val_bal_acc':float(best[0]), 'test_bal_acc':float(test_bal),
                              'test_auc':float(test_auc), 'test_ap':float(test_ap),
                              'test_accuracy':float(test_acc), 'test_f1':float(test_f1)}]).to_csv(best_cls_csv, index=False)
                print('Saved:', best_cls_csv)

            print('\n[Search] Regression MLP hyperparameter grid (select by validation MSE)')
            reg_grid = []
            hidden_grid_r = [(64,32), (128,64), (64,64,32)]
            lr_grid_r = [1e-3, 3e-4]
            pdrop_grid_r = [0.2, 0.0]
            best_r = (1e9, None)
            for h in hidden_grid_r:
                for lr_ in lr_grid_r:
                    for pd_ in pdrop_grid_r:
                        mtmp = train_regressor(Xtr, ytr, Xva, yva, seed=0, hidden=h, lr=lr_, p_drop=pd_)
                        with torch.no_grad():
                            pv = mtmp(torch.tensor(Xva, dtype=torch.float32)).cpu().numpy()
                        
                        val_mse = mean_squared_error(np.clip(yva,0,4), np.clip(pv,0,4))
                        reg_grid.append({'hidden':h,'lr':lr_,'p_drop':pd_,'val_mse':float(val_mse)})
                        if val_mse < best_r[0]: best_r = (val_mse, (h, lr_, pd_))
            reg_grid_df = pd.DataFrame(reg_grid)
            reg_grid_csv = out_dir / 'regression_grid_results.csv'
            reg_grid_df.sort_values('val_mse', ascending=True).to_csv(reg_grid_csv, index=False)
            print('Saved:', reg_grid_csv)
            if best_r[1] is not None:
                h, lr_, pd_ = best_r[1]
                print(f"Best REG config: hidden={h}, lr={lr_}, p_drop={pd_}, val_mse={best_r[0]:.4f}")
                mbest = train_regressor(Xtr, ytr, Xva, yva, seed=0, hidden=h, lr=lr_, p_drop=pd_)
                with torch.no_grad(): y_te_pred = mbest(torch.tensor(Xte, dtype=torch.float32)).cpu().numpy()
                
                rmse = float(np.sqrt(mean_squared_error(np.clip(yte,0,4), np.clip(y_te_pred,0,4))))
                mae = float(mean_absolute_error(np.clip(yte,0,4), np.clip(y_te_pred,0,4)))
                best_reg_csv = out_dir / 'regression_best_model_metrics.csv'
                pd.DataFrame([{'hidden':h,'lr':lr_,'p_drop':pd_,'val_mse':float(best_r[0]),'test_rmse':rmse,'test_mae':mae}]).to_csv(best_reg_csv, index=False)
                print('Saved:', best_reg_csv)




if __name__ == '__main__':
    main()
